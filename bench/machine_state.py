#!/usr/bin/env python3
"""bench/machine_state.py -- HARDWARE.md sections 5.1 to 5.5 as reusable code.

Later stages re-run this fingerprint verification before their first timed run,
so it is a tool rather than a one-off script.

  5.1  clock and power state query
  5.3  sustained-load thermal and boost logging (GPU and CPU together)
  5.4  the four stability checks: VRAM integrity, compute determinism,
       timing reproducibility, background load enumeration
  5.5  environment fingerprint capture, and a verify mode that reports every
       field that differs from a stored fingerprint

Section 5.2 is the operator's clock-regime decision and is recorded in
HARDWARE.md by hand, not queried.

A value that cannot be obtained is recorded as null together with the reason.
Nothing here substitutes a plausible figure for a missing one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
FINGERPRINT_PATH = RESULTS_DIR / "machine_fingerprint.json"

SMI = shutil.which("nvidia-smi") or "nvidia-smi"


# ---------------------------------------------------------------- helpers ---
def _run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError as e:
        return 127, "", f"not found: {e}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def _smi_query(fields, timeout=30):
    """Returns {field: value-or-None}. A field the driver refuses is None with
    its literal reply preserved in the *_raw entry."""
    rc, out, err = _run([SMI, f"--query-gpu={','.join(fields)}",
                         "--format=csv,noheader"], timeout=timeout)
    if rc != 0 or not out:
        return {f: None for f in fields} | {"_error": err or f"rc={rc}"}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    result = {}
    for f, raw in zip(fields, parts):
        result[f] = None if raw in ("[N/A]", "N/A", "[Not Supported]", "") else raw
        result[f + "_raw"] = raw
    return result


def _powershell(script, timeout=60):
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                timeout=timeout)


# ------------------------------------------------------ 5.1 clock / power ---
def query_clock_power_state() -> dict:
    """Everything HARDWARE.md 5.1 asks for. Fields the machine cannot answer
    are recorded as unobtainable WITH the reason -- in particular this GPU
    reports no clock-offset field at all, so whether the vendor utility has
    applied an offset cannot be closed by query."""
    g = _smi_query([
        "name", "driver_version", "vbios_version", "compute_cap", "pstate",
        "clocks.sm", "clocks.mem", "clocks.gr",
        "clocks.max.sm", "clocks.max.mem", "clocks.max.gr",
        "power.draw", "power.limit", "power.default_limit",
        "power.min_limit", "power.max_limit", "enforced.power.limit",
        "temperature.gpu", "memory.total", "memory.used",
        "utilization.gpu", "utilization.memory",
    ])
    throttle = _smi_query([
        "clocks_throttle_reasons.active",
        "clocks_throttle_reasons.gpu_idle",
        "clocks_throttle_reasons.sw_power_cap",
        "clocks_throttle_reasons.hw_slowdown",
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.hw_power_brake_slowdown",
        "clocks_throttle_reasons.sw_thermal_slowdown",
    ])

    rc, full, _ = _run([SMI, "-q"], timeout=60)
    offset_fields = [ln.strip() for ln in full.splitlines() if "Offset" in ln]
    temp_limits = {}
    for ln in full.splitlines():
        for key in ("GPU Shutdown Temp", "GPU Slowdown Temp", "GPU Max Operating Temp"):
            if key in ln and ":" in ln:
                temp_limits[key] = ln.split(":", 1)[1].strip()

    rc_pp, pp, _ = _run(["powercfg", "/getactivescheme"])
    rc_hw, hw, _ = _powershell(
        "(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers'"
        " -Name HwSchMode -ErrorAction SilentlyContinue).HwSchMode")
    rc_oc, oc, _ = _powershell(
        "$p = Get-Process -Name MSIAfterburner,RTSS,EVGAPrecisionX1,GPUTweak"
        " -ErrorAction SilentlyContinue; if ($p) { ($p.Name -join ',') } else { 'none running' }")
    oc_installed = Path(r"C:\Program Files (x86)\MSI Afterburner").exists()

    return {
        "gpu": g,
        "throttle_reasons": throttle,
        "temperature_limits": temp_limits or None,
        "core_clock_offset_vs_stock": None,
        "memory_clock_offset_vs_stock": None,
        "clock_offset_unobtainable_reason":
            ("nvidia-smi on this machine exposes no clock-offset field "
             f"(matching lines in `nvidia-smi -q`: {offset_fields or 'none'}). "
             "Vendor-applied offsets are therefore invisible to query and "
             "whether the GPU is at stock is UNVERIFIED, not zero."),
        "overclocking_utility_installed": oc_installed,
        "overclocking_utility_running": oc.strip() if rc_oc == 0 else None,
        "performance_or_shift_mode_profile": None,
        "performance_mode_unobtainable_reason":
            "the vendor WMI class (root\\WMI MSI_VGA) returns Access denied "
            "without elevation; the active user scenario must be read from the "
            "vendor utility's own UI by the operator",
        "fan_profile": None,
        "fan_profile_unobtainable_reason":
            "same vendor WMI access restriction as the performance mode",
        "windows_power_plan": pp.strip() if rc_pp == 0 else None,
        "hardware_accelerated_gpu_scheduling": hw.strip() if rc_hw == 0 else None,
        "clocks_lockable": check_clock_lock_support(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _is_elevated() -> bool:
    if os.name != "nt":
        return hasattr(os, "geteuid") and os.geteuid() == 0
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:       # noqa: BLE001 - treated as "unknown, assume elevated"
        return True


def check_clock_lock_support() -> dict:
    """Determines whether -lgc and -lmc are available.

    The probe works by asking the driver to apply a lock and reading what it
    says: an unelevated process is refused before anything changes, which is
    exactly the signal wanted. From an ELEVATED process the same call would
    really apply a lock, so it is not run there -- device state is the
    operator's to change, deliberately, not a side effect of a query.
    """
    if _is_elevated():
        msg = ("not probed: this process is elevated, and the probe would "
               "actually apply a clock lock. Run the query unelevated, or take "
               "the answer from the operator's own nvidia-smi -lgc / -lmc run.")
        return {"graphics_clock_lock": {"supported_by_device": None,
                                        "blocked_by_permission": None, "message": msg},
                "memory_clock_lock": {"supported_by_device": None,
                                      "blocked_by_permission": None, "message": msg}}

    rc_g, out_g, err_g = _run([SMI, "-lgc", "0"], timeout=30)
    rc_m, out_m, err_m = _run([SMI, "-lmc", "0"], timeout=30)
    txt_g, txt_m = (out_g + err_g), (out_m + err_m)
    return {
        "graphics_clock_lock": {
            "supported_by_device": "not supported" not in txt_g.lower(),
            "blocked_by_permission": "permission" in txt_g.lower(),
            "message": txt_g.strip() or None,
        },
        "memory_clock_lock": {
            "supported_by_device": "not supported" not in txt_m.lower(),
            "blocked_by_permission": "permission" in txt_m.lower(),
            "message": txt_m.strip() or None,
        },
    }


def verify_clock_lock(expected_mhz: int, samples: int = 10,
                      tolerance_mhz: int = 15, load_cmd: list[str] | None = None) -> dict:
    """Confirms a graphics-clock lock is ACTUALLY in effect rather than assuming
    the operator's command took. Clocks idle down when nothing is running, so a
    load is put on the device while the clock is sampled."""
    proc = None
    if load_cmd:
        proc = subprocess.Popen(load_cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    observed = []
    try:
        for _ in range(samples):
            g = _smi_query(["clocks.sm", "utilization.gpu"])
            raw = g.get("clocks.sm")
            if raw:
                observed.append(int(raw.split()[0]))
            time.sleep(0.3)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
    if not observed:
        return {"locked": False, "observed_mhz": [],
                "reason": "no clock samples could be read from nvidia-smi"}
    off = [m for m in observed if abs(m - expected_mhz) > tolerance_mhz]
    return {
        "locked": not off,
        "expected_mhz": expected_mhz,
        "observed_mhz": observed,
        "off_target_samples": off,
        "reason": None if not off else
                  f"{len(off)} of {len(observed)} samples were more than "
                  f"{tolerance_mhz} MHz from the requested {expected_mhz} MHz",
    }


# ------------------------------------------ 5.3 sustained-load thermal log ---
def _cpu_sample() -> dict:
    """CPU frequency and package temperature, logged alongside the GPU series.
    This machine is a laptop whose CPU and GPU share one thermal solution, so
    the CPU series is part of the GPU's story, not a separate curiosity."""
    rc, out, _ = _powershell(
        "$p=(Get-CimInstance Win32_Processor);"
        "$t=$null;"
        "try{$t=(Get-CimInstance -Namespace root/WMI -ClassName MSAcpi_ThermalZoneTemperature"
        " -ErrorAction Stop | Measure-Object -Property CurrentTemperature -Maximum).Maximum}catch{}"
        "[pscustomobject]@{mhz=$p.CurrentClockSpeed;load=$p.LoadPercentage;tz=$t}|ConvertTo-Json -Compress")
    if rc != 0 or not out:
        return {"cpu_mhz": None, "cpu_load_pct": None, "cpu_temp_c": None,
                "cpu_temp_reason": "WMI query failed"}
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        return {"cpu_mhz": None, "cpu_load_pct": None, "cpu_temp_c": None,
                "cpu_temp_reason": f"unparseable WMI reply {out!r}"}
    tz = d.get("tz")
    return {
        "cpu_mhz": d.get("mhz"),
        "cpu_load_pct": d.get("load"),
        # MSAcpi_ThermalZoneTemperature reports decikelvin
        "cpu_temp_c": (tz / 10.0 - 273.15) if isinstance(tz, (int, float)) else None,
        "cpu_temp_reason": None if tz is not None else
                           "MSAcpi_ThermalZoneTemperature unavailable on this machine",
    }


def log_sustained_load(duration_s: int = 300, interval_s: float = 2.0,
                       load_cmd: list[str] | None = None) -> dict:
    """Section 5.3. Logs GPU clock, temperature, power and throttle reason, plus
    CPU frequency and package temperature, for the full duration."""
    idle = {"gpu": _smi_query(["clocks.sm", "temperature.gpu", "power.draw",
                               "clocks_throttle_reasons.active", "utilization.gpu"]),
            "cpu": _cpu_sample()}

    proc = None
    if load_cmd:
        proc = subprocess.Popen(load_cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

    series = []
    t0 = time.time()
    try:
        while time.time() - t0 < duration_s:
            g = _smi_query(["clocks.sm", "temperature.gpu", "power.draw",
                            "clocks_throttle_reasons.active", "utilization.gpu"])
            row = {"t_s": round(time.time() - t0, 3)}
            row.update({k: v for k, v in g.items() if not k.endswith("_raw")})
            row.update(_cpu_sample())
            series.append(row)
            time.sleep(interval_s)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()

    def _mhz(row):
        v = row.get("clocks.sm")
        return int(v.split()[0]) if v else None

    def _temp(row):
        v = row.get("temperature.gpu")
        return float(v.split()[0]) if v else None

    def at(t):
        for row in series:
            if row["t_s"] >= t:
                return row
        return series[-1] if series else None

    temps = [x for x in (_temp(r) for r in series) if x is not None]
    throttles = sorted({r.get("clocks_throttle_reasons.active")
                        for r in series if r.get("clocks_throttle_reasons.active")})
    stab = compute_stabilization_time(series)

    return {
        "idle": idle,
        "series": series,
        "at_30s": at(30),
        "at_300s": at(300),
        "peak_temperature_c": max(temps) if temps else None,
        "throttle_reasons_seen": throttles,
        "stabilization": stab,
        "max_safe_continuous_benchmark_s": _derive_safe_duration(series, stab),
        "duration_requested_s": duration_s,
        "interval_s": interval_s,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def compute_stabilization_time(series: list[dict], window: int = 5,
                               band_mhz: int = 30) -> dict:
    """Time until the SM clock settles, DERIVED from the logged series.

    The clock is stable from the first sample whose following `window` samples
    all sit inside +/- band_mhz of it. Returns None with a reason when the
    series never settles -- never a default value standing in for a
    measurement that did not happen.
    """
    mhz = []
    for r in series:
        v = r.get("clocks.sm")
        mhz.append((r["t_s"], int(v.split()[0])) if v else (r["t_s"], None))
    usable = [(t, m) for t, m in mhz if m is not None]
    if len(usable) < window + 1:
        return {"seconds": None,
                "reason": f"only {len(usable)} usable clock samples; need at least {window + 1}"}
    for i in range(len(usable) - window):
        t, m = usable[i]
        if all(abs(usable[j][1] - m) <= band_mhz for j in range(i + 1, i + 1 + window)):
            return {"seconds": t, "settled_mhz": m,
                    "criterion": f"{window} consecutive samples within +/-{band_mhz} MHz",
                    "reason": None}
    return {"seconds": None,
            "reason": f"the SM clock never held within +/-{band_mhz} MHz for "
                      f"{window} consecutive samples over the whole run"}


def _derive_safe_duration(series, stab) -> dict:
    """Longest run before a throttle reason appears, derived from the series."""
    for r in series:
        active = r.get("clocks_throttle_reasons.active")
        if active and active not in ("0x0000000000000000", "0x0000000000000001"):
            return {"seconds": r["t_s"],
                    "reason": f"throttle reason {active} appeared at t={r['t_s']}s"}
    if not series:
        return {"seconds": None, "reason": "no samples logged"}
    return {"seconds": series[-1]["t_s"],
            "reason": "no throttle reason appeared; the observed run length is "
                      "a lower bound, not a measured limit"}


# ------------------------------- Stage 0b: CPU telemetry source probing -----
# HARDWARE.md 5.3 records that Stage 0's CPU telemetry was static and unusable:
# Win32_Processor.CurrentClockSpeed read a constant 2496 MHz and
# MSAcpi_ThermalZoneTemperature a constant value across every sample of both
# runs while CPU load varied between 4% and 36%. Two series that do not move
# while their driver does are static nominal reads, not live measurements.
#
# Nothing here is used as telemetry until it has been shown to MOVE under a
# load applied on purpose and to return toward idle when that load stops, and
# until its per-probe cost has been measured. A source that fails either test is
# reported as static, or as too expensive -- never emitted as a series that
# looks like a measurement.
#
# EVERY sampler here runs OUTSIDE any timed bracket. The C microbenchmarks own
# their timed regions and contain no telemetry call at all;
# tests/test_machine_state.py asserts that by scanning their sources between t0
# and t1. When a run is instrumented, the sampler runs in a separate process.

PDH_FMT_DOUBLE = 0x00000200

# Candidate live sources, in the order they are probed.
TELEMETRY_SOURCES = [
    {
        "name": "pdh_processor_performance_total",
        "path": "\\Processor Information(_Total)\\% Processor Performance",
        "quantity": "cpu_frequency",
        "unit": "percent of nominal (x nominal MHz gives MHz)",
        "note": "APERF/MPERF derived; exceeds 100 when the core is in turbo",
    },
    {
        "name": "pdh_performance_limit_total",
        "path": "\\Processor Information(_Total)\\% Performance Limit",
        "quantity": "cpu_frequency_limit",
        "unit": "percent",
        "note": "the ceiling the platform is currently imposing, if any",
    },
    {
        "name": "pdh_processor_frequency_total",
        "path": "\\Processor Information(_Total)\\Processor Frequency",
        "quantity": "cpu_frequency",
        "unit": "MHz",
        "note": "nominal frequency on this platform; probed to show it is static",
    },
]


def logical_core_mapping() -> dict:
    """This CPU's logical-to-physical core mapping, QUERIED not assumed.

    GetLogicalProcessorInformationEx(RelationProcessorCore) returns one record
    per physical core carrying the affinity mask of its logical processors. The
    conventional interleaving (0,1 on core 0; 2,3 on core 1; ...) is what this
    machine happens to use, but it is a convention, not a guarantee, and the
    sampler's affinity depends on getting it right -- so it is read rather than
    assumed.
    """
    if os.name != "nt":
        return {"available": False,
                "reason": "GetLogicalProcessorInformationEx is a Windows interface"}
    import ctypes
    import ctypes.wintypes as wt

    RELATION_PROCESSOR_CORE = 0
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = wt.DWORD(0)
    k32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None,
                                         ctypes.byref(size))
    if size.value == 0:
        return {"available": False,
                "reason": "GetLogicalProcessorInformationEx reported a zero buffer size"}
    buf = (ctypes.c_ubyte * size.value)()
    if not k32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf,
                                                ctypes.byref(size)):
        return {"available": False,
                "reason": f"GetLogicalProcessorInformationEx failed, "
                          f"GetLastError={ctypes.get_last_error()}"}

    # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX:
    #   Relationship DWORD | Size DWORD | PROCESSOR_RELATIONSHIP {
    #     Flags BYTE | EfficiencyClass BYTE | Reserved[20] |
    #     GroupCount WORD | GROUP_AFFINITY GroupMask[] }
    #   GROUP_AFFINITY: Mask ULONG_PTR | Group WORD | Reserved[3] WORD
    physical = []
    off = 0
    while off < size.value:
        rec_size = int.from_bytes(bytes(buf[off + 4:off + 8]), "little")
        smt_flag = buf[off + 8]
        group_count = int.from_bytes(bytes(buf[off + 30:off + 32]), "little")
        gm = off + 32
        mask = 0
        if group_count:
            mask = int.from_bytes(bytes(buf[gm:gm + 8]), "little")
        logical = [i for i in range(64) if (mask >> i) & 1]
        physical.append({"physical_core": len(physical),
                         "logical_cpus": logical,
                         "affinity_mask": mask,
                         "smt": bool(smt_flag & 1)})
        off += rec_size

    logical_to_physical = {}
    siblings = {}
    for core in physical:
        for lp in core["logical_cpus"]:
            logical_to_physical[lp] = core["physical_core"]
            siblings[lp] = [x for x in core["logical_cpus"] if x != lp]
    return {"available": True,
            "n_physical_cores": len(physical),
            "n_logical_cpus": len(logical_to_physical),
            "physical_cores": physical,
            "logical_to_physical": logical_to_physical,
            "smt_siblings": siblings}


def sampler_affinity_mask(exclude_logical_cpus, mapping=None) -> dict:
    """Affinity mask for the telemetry sampler: every logical CPU EXCEPT the ones
    named, and except their SMT siblings.

    The sampler is a separate thread and Windows is free to place it on the very
    logical processor the benchmark is pinned to, or on that processor's SMT
    sibling, which shares the physical core's execution ports and L1d. Either
    would make the instrument part of what it is measuring. The mask returned
    here excludes both.

    Returns the mask together with the reasoning, or available=False with the
    reason. A sampler that cannot be pinned is not run at all -- an unpinnable
    sampler sharing a physical core with the measured thread is worse than no
    trace.
    """
    mapping = mapping or logical_core_mapping()
    if not mapping.get("available"):
        return {"available": False, "reason": mapping.get("reason"),
                "mapping": mapping}
    excluded = set()
    for cpu in exclude_logical_cpus:
        excluded.add(cpu)
        excluded.update(mapping["smt_siblings"].get(cpu, []))
    allowed = [i for i in range(mapping["n_logical_cpus"]) if i not in excluded]
    if not allowed:
        return {"available": False,
                "reason": "excluding those logical CPUs and their SMT siblings "
                          "leaves the sampler nowhere to run",
                "excluded": sorted(excluded)}
    mask = 0
    for i in allowed:
        mask |= 1 << i
    return {"available": True,
            "mask": mask,
            "mask_hex": f"0x{mask:x}",
            "allowed_logical_cpus": allowed,
            "excluded_logical_cpus": sorted(excluded),
            "requested_exclusions": sorted(set(exclude_logical_cpus)),
            "n_logical_cpus": mapping["n_logical_cpus"],
            "logical_to_physical": mapping["logical_to_physical"],
            "smt_siblings": {c: mapping["smt_siblings"].get(c, [])
                             for c in sorted(set(exclude_logical_cpus))}}


class PdhCounter:
    """One PDH counter, opened once and sampled cheaply.

    Rate counters such as "% Processor Performance" are differential: a sample
    reports the average over the interval since the previous collection. That is
    exactly the property wanted here -- probing immediately before and
    immediately after a region attributes the interval to that region without
    any probe being inside it.
    """

    def __init__(self, path: str):
        if os.name != "nt":
            raise OSError("PDH is a Windows interface")
        import ctypes
        import ctypes.wintypes as wt

        class _Union(ctypes.Union):
            _fields_ = [("longValue", ctypes.c_long),
                        ("doubleValue", ctypes.c_double),
                        ("largeValue", ctypes.c_longlong),
                        ("AnsiStringValue", ctypes.c_char_p),
                        ("WideStringValue", ctypes.c_wchar_p)]

        class _Value(ctypes.Structure):
            _fields_ = [("CStatus", wt.DWORD), ("u", _Union)]

        self._ctypes = ctypes
        self._Value = _Value
        self._pdh = ctypes.WinDLL("pdh.dll")
        self.path = path
        self._query = wt.HANDLE()
        rc = self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        if rc != 0:
            raise OSError("PdhOpenQueryW failed: 0x%08x" % (rc & 0xffffffff))
        self._counter = wt.HANDLE()
        rc = self._pdh.PdhAddEnglishCounterW(self._query, path, 0,
                                             ctypes.byref(self._counter))
        if rc != 0:
            raise OSError("PdhAddEnglishCounterW(%r) failed: 0x%08x"
                          % (path, rc & 0xffffffff))
        self._pdh.PdhCollectQueryData(self._query)   # prime the differential

    def sample(self):
        """One reading, or None when the counter has no value yet."""
        self._pdh.PdhCollectQueryData(self._query)
        v = self._Value()
        rc = self._pdh.PdhGetFormattedCounterValue(
            self._counter, PDH_FMT_DOUBLE, None, self._ctypes.byref(v))
        return v.u.doubleValue if rc == 0 else None

    def close(self):
        try:
            self._pdh.PdhCloseQuery(self._query)
        except Exception:       # noqa: BLE001 - closing must not mask a result
            pass


def classify_telemetry_source(name, idle, load, recovery, per_probe_us,
                              unit="", note=None, move_pct=2.0) -> dict:
    """Decides whether a candidate source is LIVE, from its own readings.

    A source is live only when BOTH hold:
      - the readings MOVE: the load-phase median differs from the idle-phase
        median by more than move_pct of the idle median, or the source varies
        within a phase by more than move_pct;
      - it RETURNS toward idle once the load stops.

    A source whose readings never change is reported STATIC, with its constant
    value, and is not usable as telemetry. Nothing here invents a range: when a
    phase produced no readings the verdict is "unavailable" with the reason.
    """
    phases = {"idle": idle, "load": load, "recovery": recovery}
    clean = {k: [x for x in v if isinstance(x, (int, float))] for k, v in phases.items()}
    missing = [k for k, v in clean.items() if not v]
    if missing:
        return {"source": name, "unit": unit, "note": note, "live": False,
                "verdict": "unavailable", "per_probe_us": per_probe_us,
                "reason": "no readings in phase(s): " + ", ".join(missing)}

    stats = {k: {"n": len(v), "min": min(v), "max": max(v),
                 "median": statistics.median(v)} for k, v in clean.items()}
    all_vals = clean["idle"] + clean["load"] + clean["recovery"]
    if min(all_vals) == max(all_vals):
        return {"source": name, "unit": unit, "note": note, "live": False,
                "verdict": "static", "constant_value": all_vals[0],
                "per_probe_us": per_probe_us, "phases": stats,
                "reason": ("every one of %d readings was %s across idle, load and "
                           "recovery; a series that does not move while its driver "
                           "does is a static nominal read, not a live measurement"
                           % (len(all_vals), all_vals[0]))}

    idle_med = stats["idle"]["median"]
    scale = abs(idle_med) if idle_med else 1.0
    load_shift_pct = 100.0 * abs(stats["load"]["median"] - idle_med) / scale
    recovery_gap_pct = 100.0 * abs(stats["recovery"]["median"] - idle_med) / scale
    within_phase_pct = 100.0 * max((s["max"] - s["min"]) for s in stats.values()) / scale

    moved = load_shift_pct > move_pct or within_phase_pct > move_pct
    returned = recovery_gap_pct < load_shift_pct or load_shift_pct <= move_pct
    live = bool(moved and returned)
    if live:
        reason = None
    elif not moved:
        reason = ("readings changed by only %.2f%% under load and %.2f%% within a "
                  "phase, below the %.1f%% threshold"
                  % (load_shift_pct, within_phase_pct, move_pct))
    else:
        reason = "readings moved under load but did not return toward idle"
    return {
        "source": name, "unit": unit, "note": note,
        "live": live, "verdict": "live" if live else "static",
        "per_probe_us": per_probe_us, "phases": stats,
        "load_shift_pct_of_idle_median": load_shift_pct,
        "recovery_gap_pct_of_idle_median": recovery_gap_pct,
        "within_phase_range_pct_of_idle_median": within_phase_pct,
        "move_threshold_pct": move_pct,
        "reason": reason,
    }


def _spin_load(n_procs: int):
    """Holds n_procs cores busy in SEPARATE processes until they are killed.

    Separate processes rather than threads: a Python thread cannot load a second
    core, so a thread-based load would test the sources against no load at all.
    """
    code = "x=1.0\nwhile True:\n x=x*1.0000001+1.0\n"
    return [subprocess.Popen([sys.executable, "-c", code],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(n_procs)]


def measure_probe_cost_us(fn, calls: int = 100) -> float:
    """Median cost of ONE probe call, in microseconds.

    Measured before the source is used. cpu_cache_ladder reads 268435456 B per
    sample, so its per-sample durations span roughly 2 ms at L1-resident
    bandwidth to low tens of ms at DRAM-tier bandwidth; a probe costing
    milliseconds would manufacture the very variance being diagnosed.
    """
    costs = []
    for _ in range(calls):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        costs.append((t1 - t0) / 1000.0)
    return statistics.median(costs)


def probe_cpu_telemetry(load_procs: int = 4, phase_s: float = 3.0,
                        interval_s: float = 0.2, cost_calls: int = 100) -> dict:
    """Stage 0b Phase 2. Probes every candidate CPU telemetry source.

    For each: measures the per-probe cost, then samples at idle, under a
    deliberately applied load, and after that load stops, and classifies the
    source from its own readings. Nothing here is inside a timed bracket; this
    function times nothing the project reports.
    """
    results = []

    for src in TELEMETRY_SOURCES:
        try:
            ctr = PdhCounter(src["path"])
        except Exception as e:      # noqa: BLE001 - recorded, not swallowed
            results.append({"source": src["name"], "unit": src["unit"],
                            "note": src["note"], "live": False,
                            "verdict": "unavailable", "per_probe_us": None,
                            "quantity": src["quantity"],
                            "reason": "%s: %s" % (type(e).__name__, e)})
            continue
        try:
            time.sleep(0.2)
            cost = measure_probe_cost_us(ctr.sample, cost_calls)

            def phase(seconds):
                out = []
                t0 = time.time()
                while time.time() - t0 < seconds:
                    time.sleep(interval_s)
                    out.append(ctr.sample())
                return out

            idle = phase(phase_s)
            procs = _spin_load(load_procs)
            try:
                time.sleep(0.5)
                load = phase(phase_s)
            finally:
                for pr in procs:
                    pr.kill()
            time.sleep(1.0)
            recovery = phase(phase_s)
            r = classify_telemetry_source(src["name"], idle, load, recovery, cost,
                                          src["unit"], src["note"])
            r["path"] = src["path"]
            r["quantity"] = src["quantity"]
            r["raw"] = {"idle": idle, "load": load, "recovery": recovery}
            results.append(r)
        finally:
            ctr.close()

    # The two sources Stage 0 used, re-probed so the finding is current rather
    # than carried forward.
    cost_wmi = measure_probe_cost_us(_cpu_sample, 3)
    idle = [_cpu_sample() for _ in range(2)]
    procs = _spin_load(load_procs)
    try:
        time.sleep(2.0)
        load = [_cpu_sample() for _ in range(2)]
    finally:
        for pr in procs:
            pr.kill()
    time.sleep(1.0)
    recovery = [_cpu_sample() for _ in range(2)]
    for key, name, unit, quantity in (
            ("cpu_mhz", "wmi_win32_processor_currentclockspeed", "MHz", "cpu_frequency"),
            ("cpu_temp_c", "wmi_msacpi_thermalzonetemperature", "degrees C",
             "cpu_package_temperature")):
        r = classify_telemetry_source(
            name,
            [d.get(key) for d in idle],
            [d.get(key) for d in load],
            [d.get(key) for d in recovery],
            cost_wmi, unit,
            "the source HARDWARE.md 5.3 records as static in Stage 0")
        r["quantity"] = quantity
        results.append(r)

    live_freq = [r for r in results
                 if r.get("live") and r.get("quantity") == "cpu_frequency"]
    live_temp = [r for r in results
                 if r.get("live") and r.get("quantity") == "cpu_package_temperature"]
    return {
        "sources": results,
        "live_cpu_frequency_source": live_freq[0]["source"] if live_freq else None,
        "live_cpu_package_temperature_source": live_temp[0]["source"] if live_temp else None,
        "consequence_if_no_temperature_source":
            ("the interference-versus-governance question cannot be separated by "
             "package temperature on this machine, and the PERSISTENT.md flag "
             "calling for a Balanced-power-plan re-test before Stage 5 stays "
             "blocked for the same reason"),
        "sampled_inside_any_timed_bracket": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


class CpuTelemetrySampler:
    """Background sampler for the duration of a benchmark PROCESS.

    Runs in this process while the benchmark runs in another, so no probe can
    land inside the benchmark's timed bracket. The trace it produces is a
    recorded run condition, not a measurement, and the sampler's own per-probe
    cost and duty cycle are recorded alongside it so its contribution to
    background load is visible rather than assumed negligible.
    """

    def __init__(self, interval_s: float = 0.02, exclude_logical_cpus=(0, 2)):
        self.interval_s = interval_s
        # Logical CPU 2 is where bench_pin_current_thread() places the measured
        # thread (BENCH_DEFAULT_PIN_CPU); logical CPU 0 carries this system's
        # interrupt and DPC work. Both, and CPU 2's SMT sibling, are excluded.
        self.exclude_logical_cpus = tuple(exclude_logical_cpus)
        self.affinity = sampler_affinity_mask(self.exclude_logical_cpus)
        self.counters = []
        self.unavailable = []
        for src in TELEMETRY_SOURCES:
            try:
                self.counters.append((src["name"], PdhCounter(src["path"])))
            except Exception as e:      # noqa: BLE001
                self.unavailable.append({"source": src["name"],
                                         "reason": "%s: %s" % (type(e).__name__, e)})
        self.samples = []
        self._stop = None
        self._thread = None
        self._t0 = None

    def start(self):
        """Starts sampling. Raises when the sampler cannot be pinned clear of the
        measured thread's physical core: a sampler sharing that core is worse
        than no trace, so the caller falls back to running without telemetry."""
        import ctypes
        import threading

        if not self.affinity.get("available"):
            raise RuntimeError(
                "the telemetry sampler cannot be pinned clear of the measured "
                f"thread: {self.affinity.get('reason')}")

        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        applied = {"ok": False, "reason": "not attempted"}
        ready = threading.Event()

        def loop():
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            prev = k32.SetThreadAffinityMask(k32.GetCurrentThread(),
                                             ctypes.c_size_t(self.affinity["mask"]))
            if prev == 0:
                applied["ok"] = False
                applied["reason"] = ("SetThreadAffinityMask failed, "
                                     f"GetLastError={ctypes.get_last_error()}")
                ready.set()
                return
            applied["ok"] = True
            applied["reason"] = None
            applied["previous_mask_hex"] = f"0x{prev:x}"
            ready.set()
            while not self._stop.is_set():
                row = {"t_s": round(time.perf_counter() - self._t0, 6)}
                for name, ctr in self.counters:
                    row[name] = ctr.sample()
                self.samples.append(row)
                self._stop.wait(self.interval_s)
            k32.SetThreadAffinityMask(k32.GetCurrentThread(),
                                      ctypes.c_size_t(prev))

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        ready.wait(timeout=5.0)
        self._affinity_applied = applied
        if not applied["ok"]:
            self._stop.set()
            self._thread.join(timeout=5.0)
            raise RuntimeError("the telemetry sampler could not be pinned: "
                               + str(applied["reason"]))
        return self

    def stop(self) -> dict:
        elapsed = (time.perf_counter() - self._t0) if self._t0 else None
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        cost_us = None
        for _, ctr in self.counters:
            if cost_us is None:
                try:
                    cost_us = measure_probe_cost_us(ctr.sample, 20)
                except Exception:       # noqa: BLE001
                    cost_us = None
            ctr.close()
        series = {}
        for name, _ in self.counters:
            vals = [r[name] for r in self.samples if isinstance(r.get(name), (int, float))]
            series[name] = {"n": len(vals),
                            "min": min(vals) if vals else None,
                            "max": max(vals) if vals else None,
                            "median": statistics.median(vals) if vals else None}
        duty = None
        if cost_us is not None and elapsed:
            duty = (cost_us * 1e-6 * len(self.counters) * len(self.samples)) / elapsed
        return {
            "interval_s": self.interval_s,
            "affinity": self.affinity,
            "affinity_applied": getattr(self, "_affinity_applied", None),
            "n_samples": len(self.samples),
            "elapsed_s": elapsed,
            "per_probe_us": cost_us,
            "sampler_duty_cycle_of_one_core": duty,
            "sources_unavailable": self.unavailable,
            "summary": series,
            "samples": self.samples,
            "sampled_inside_any_timed_bracket": False,
            "sampler_note": ("sampled from a separate process while the benchmark "
                             "ran; no probe can be inside the benchmark's timed "
                             "bracket. The sampler is itself an added background "
                             "load and is recorded as a run condition"),
        }


def power_source() -> dict:
    """AC or battery. BENCHMARK_PROTOCOL.md 3 makes a run on battery INVALID
    outright, so this is a gate, not a note.

    GetSystemPowerStatus.ACLineStatus: 0 offline (battery), 1 online (AC),
    255 unknown. An unknown status is reported as unknown -- never assumed to
    be AC.
    """
    if os.name != "nt":
        return {"on_ac": None, "reason": "GetSystemPowerStatus is a Windows interface"}
    import ctypes
    import ctypes.wintypes as wt

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte),
                    ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", wt.DWORD),
                    ("BatteryFullLifeTime", wt.DWORD)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    st = SYSTEM_POWER_STATUS()
    if not k32.GetSystemPowerStatus(ctypes.byref(st)):
        return {"on_ac": None,
                "reason": f"GetSystemPowerStatus failed, "
                          f"GetLastError={ctypes.get_last_error()}"}
    status = {0: "on battery", 1: "on AC", 255: "unknown"}.get(st.ACLineStatus,
                                                               "unrecognised")
    return {
        "on_ac": True if st.ACLineStatus == 1 else (False if st.ACLineStatus == 0 else None),
        "ac_line_status": st.ACLineStatus,
        "ac_line_status_text": status,
        "battery_life_percent": (st.BatteryLifePercent
                                 if st.BatteryLifePercent != 255 else None),
        "battery_flag": st.BatteryFlag,
        "protocol": ("BENCHMARK_PROTOCOL.md 3: a run taken on battery is INVALID "
                     "outright"),
        "reason": None if st.ACLineStatus == 1 else
                  f"ACLineStatus reports {status}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def windows_update_state() -> dict:
    """Whether Windows Update is paused. A recorded condition: driver 591.44 is a
    frozen fingerprint value and an update installed mid-session would void every
    comparison against Stage 0 with no undo."""
    rc, out, _ = _powershell(
        "$k='HKLM:\\SOFTWARE\\Microsoft\\WindowsUpdate\\UX\\Settings';"
        "$p=Get-ItemProperty $k -EA SilentlyContinue;"
        "[pscustomobject]@{"
        "PauseUpdatesExpiryTime=$p.PauseUpdatesExpiryTime;"
        "PauseFeatureUpdatesEndTime=$p.PauseFeatureUpdatesEndTime;"
        "PauseQualityUpdatesEndTime=$p.PauseQualityUpdatesEndTime;"
        "FlightSettingsMaxPauseDays=$p.FlightSettingsMaxPauseDays;"
        "WuAuServStatus=(Get-Service wuauserv -EA SilentlyContinue).Status.ToString()"
        "} | ConvertTo-Json -Compress")
    try:
        d = json.loads(out) if rc == 0 and out else None
    except json.JSONDecodeError:
        d = None
    expiry = (d or {}).get("PauseUpdatesExpiryTime")
    return {"detail": d,
            "paused": bool(expiry) if d is not None else None,
            "pause_expiry": expiry,
            "reason": None if d is not None else
                      "the Windows Update settings key could not be read",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def network_state() -> dict:
    """Connection type and metered status. A recorded run condition for Stage 0b,
    which runs over a mobile hotspot rather than Stage 0's connection."""
    rc, out, err = _powershell(
        "$p = Get-NetConnectionProfile | Select-Object Name,InterfaceAlias,"
        "NetworkCategory,IPv4Connectivity;"
        "$a = Get-NetAdapter | Where-Object Status -eq 'Up' |"
        " Select-Object Name,InterfaceDescription,LinkSpeed;"
        "[pscustomobject]@{profiles=$p;adapters=$a} | ConvertTo-Json -Depth 4 -Compress")
    try:
        detail = json.loads(out) if rc == 0 and out else None
    except json.JSONDecodeError:
        detail = None
    rc_m, metered, _ = _powershell(
        "try{[void][Windows.Networking.Connectivity.NetworkInformation,"
        "Windows.Networking.Connectivity,ContentType=WindowsRuntime];"
        "$p=[Windows.Networking.Connectivity.NetworkInformation]::"
        "GetInternetConnectionProfile();"
        "$p.GetConnectionCost().NetworkCostType}catch{'unavailable'}")
    return {
        "detail": detail,
        "metered_cost_type": metered.strip() if rc_m == 0 else None,
        "metered_unobtainable_reason": None if rc_m == 0 else
            "the WinRT connection-cost API could not be reached from PowerShell",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------- 5.4 the stability checks ----
def compare_patterns(written: bytes, read_back: bytes) -> dict:
    """Bit-exact comparison, factored out so the corruption detector itself can
    be unit tested without a GPU."""
    if len(written) != len(read_back):
        return {"ok": False, "first_bad_offset": 0, "bad_bytes": abs(len(written) - len(read_back)),
                "reason": f"length mismatch {len(written)} vs {len(read_back)}"}
    bad = [i for i, (a, b) in enumerate(zip(written, read_back)) if a != b]
    return {"ok": not bad,
            "first_bad_offset": bad[0] if bad else None,
            "bad_bytes": len(bad),
            "reason": None if not bad else f"{len(bad)} byte(s) differ on readback"}


def vram_integrity(buffer_mib: int = 1024, repeats: int = 3) -> dict:
    """Writes known patterns to a large device buffer and verifies bit-exact
    readback, repeated so the check spans a period of thermal load.

    Uses PyTorch purely as a device-memory mover. PyTorch is this project's
    correctness oracle and is never in the inference path; a machine-state
    check is not the inference path either.
    """
    try:
        import torch
    except ImportError as e:
        return {"ok": None, "reason": f"PyTorch not importable, check not run: {e}"}
    if not torch.cuda.is_available():
        return {"ok": None, "reason": "torch.cuda.is_available() is False; check not run"}

    patterns = [0x00, 0xFF, 0xAA, 0x55]
    n = buffer_mib * 1024 * 1024
    rounds = []
    ok = True
    for r in range(repeats):
        for p in patterns:
            host = torch.full((n,), p, dtype=torch.uint8)
            dev = host.to("cuda")
            torch.cuda.synchronize()
            back = dev.cpu()
            del dev
            torch.cuda.empty_cache()
            cmp = compare_patterns(host.numpy().tobytes()[:1 << 20],
                                   back.numpy().tobytes()[:1 << 20])
            exact = bool(torch.equal(host, back))
            rounds.append({"round": r, "pattern": hex(p),
                           "bit_exact": exact, "sampled_prefix": cmp})
            ok = ok and exact
    return {"ok": ok, "buffer_mib": buffer_mib, "repeats": repeats,
            "patterns": [hex(p) for p in patterns], "rounds": rounds,
            "reason": None if ok else "readback was not bit-exact; the memory "
                                      "clock is unstable. This is a HARD STOP."}


def compute_determinism(shape=(768, 2304), k=768, trials=2) -> dict:
    """Runs the same numerical work twice and requires bit-identical results.

    The nine microbenchmarks emit timings rather than numbers, so the check
    uses the project's own QKV-projection GEMM shape through the oracle, which
    is the same arithmetic the Stage 2 kernels will have to match.
    """
    try:
        import torch
    except ImportError as e:
        return {"ok": None, "reason": f"PyTorch not importable, check not run: {e}"}
    if not torch.cuda.is_available():
        return {"ok": None, "reason": "torch.cuda.is_available() is False; check not run"}

    m, n = shape
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    outs = []
    for _ in range(trials):
        c = a @ b
        torch.cuda.synchronize()
        outs.append(c.cpu().numpy().tobytes())
    same = all(o == outs[0] for o in outs)
    return {"ok": same, "shape": {"M": m, "N": n, "K": k}, "trials": trials,
            "reason": None if same else "repeated runs of the same GEMM were not "
                                        "bit-identical on this device"}


def spread_percent(run_a: dict, run_b: dict) -> dict:
    """Run-to-run spread of the full suite: per key, |b-a| / mean * 100."""
    keys = sorted(set(run_a) & set(run_b))
    per = {}
    for kk in keys:
        x, y = float(run_a[kk]), float(run_b[kk])
        mean = (x + y) / 2.0
        per[kk] = abs(y - x) / mean * 100.0 if mean != 0 else 0.0
    missing = sorted((set(run_a) ^ set(run_b)))
    return {"per_key_pct": per,
            "max_pct": max(per.values()) if per else None,
            "mean_pct": (sum(per.values()) / len(per)) if per else None,
            "keys_missing_from_one_run": missing}


def enumerate_background_load(top: int = 20) -> dict:
    rc_g, gpu, _ = _run([SMI, "--query-compute-apps=pid,process_name,used_memory",
                         "--format=csv"], timeout=30)
    rc_c, cpu, _ = _powershell(
        "Get-Process | Sort-Object CPU -Descending | Select-Object -First "
        f"{top} Name,Id,CPU | ConvertTo-Json -Compress")
    rc_o, overlay, _ = _powershell(
        "$n=Get-Process | Where-Object { $_.Name -match "
        "'Dragon|MSI|Afterburner|RTSS|GeForce|NVIDIA|Overlay|Discord|Steam|Razer|"
        "iCUE|HWiNFO|OBS|GameBar|logi' } | Select-Object -ExpandProperty Name -Unique;"
        "if ($n) { $n -join ',' } else { 'none' }")
    try:
        cpu_list = json.loads(cpu) if rc_c == 0 and cpu else None
    except json.JSONDecodeError:
        cpu_list = None
    return {
        "gpu_processes": gpu.splitlines() if rc_g == 0 else None,
        "top_cpu_processes": cpu_list,
        "overlay_and_monitoring_software": overlay.strip() if rc_o == 0 else None,
    }


# ------------------------------------------- 5.5 environment fingerprint ----
def capture_fingerprint() -> dict:
    def first_line(cmd):
        rc, out, err = _run(cmd)
        return out.splitlines()[0].strip() if rc == 0 and out else f"unavailable: {err or rc}"

    g = _smi_query(["name", "driver_version", "vbios_version", "compute_cap",
                    "memory.total", "power.max_limit"])
    rc_pp, pp, _ = _run(["powercfg", "/getactivescheme"])
    rc_hw, hw, _ = _powershell(
        "(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers'"
        " -Name HwSchMode -ErrorAction SilentlyContinue).HwSchMode")

    build_info = REPO_ROOT / "build" / "generated" / "build_info.h"
    flags = {}
    if build_info.exists():
        for line in build_info.read_text(encoding="utf-8").splitlines():
            if line.startswith("#define BENCH_"):
                parts = line.split(None, 2)
                if len(parts) == 3:
                    flags[parts[1]] = parts[2].strip().strip('"')

    torch_info = {}
    try:
        import torch
        torch_info = {"version": torch.__version__,
                      "cuda": torch.version.cuda,
                      "cuda_available": torch.cuda.is_available(),
                      "arch_list": torch.cuda.get_arch_list()}
    except Exception as e:      # noqa: BLE001 - recorded, not swallowed
        torch_info = {"error": f"{type(e).__name__}: {e}"}

    return {
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "gpu": {k: v for k, v in g.items() if not k.endswith("_raw")},
        "nvcc": first_line(["nvcc", "--version"][:1] + ["--version"]),
        "nvidia_smi": first_line([SMI, "--version"]),
        "ncu": first_line([shutil.which("ncu") or "ncu", "--version"]),
        "windows_power_plan": pp.strip() if rc_pp == 0 else None,
        "hardware_accelerated_gpu_scheduling": hw.strip() if rc_hw == 0 else None,
        "build_flags": flags or None,
        "torch": torch_info,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _flatten(d, prefix=""):
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


def verify_fingerprint(current: dict, stored: dict, ignore=("captured_utc",)) -> dict:
    """Reports EVERY field that differs, including fields present in only one."""
    cur, old = _flatten(current), _flatten(stored)
    ignored = {k for k in set(cur) | set(old)
               if any(k == i or k.endswith("." + i) for i in ignore)}
    differences = []
    for key in sorted((set(cur) | set(old)) - ignored):
        a, b = old.get(key, "<absent>"), cur.get(key, "<absent>")
        if a != b:
            differences.append({"field": key, "stored": a, "current": b})
    return {"match": not differences, "differences": differences,
            "fields_compared": len((set(cur) | set(old)) - ignored)}


# ------------------------------------------------------------------- CLI ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("query", help="5.1 clock and power state")

    s = sub.add_parser("sustained", help="5.3 thermal and boost logging")
    s.add_argument("--duration", type=int, default=300)
    s.add_argument("--interval", type=float, default=2.0)
    s.add_argument("--load", default=None, help="command to hold the GPU busy")
    s.add_argument("--out", default=str(RESULTS_DIR / "machine_state_5_3.json"))

    st = sub.add_parser("stability", help="5.4 the four stability checks")
    st.add_argument("--vram-mib", type=int, default=1024)
    st.add_argument("--out", default=str(RESULTS_DIR / "machine_state_5_4.json"))

    f = sub.add_parser("fingerprint", help="5.5 capture")
    f.add_argument("--out", default=str(FINGERPRINT_PATH))

    v = sub.add_parser("verify", help="5.5 compare against a stored fingerprint")
    v.add_argument("--stored", default=str(FINGERPRINT_PATH))

    lk = sub.add_parser("verify-lock", help="confirm a graphics clock lock is in effect")
    lk.add_argument("--mhz", type=int, required=True)
    lk.add_argument("--load", default=None)

    tel = sub.add_parser("telemetry",
                         help="Stage 0b Phase 2: probe every candidate CPU telemetry "
                              "source and report whether it is live, its observed "
                              "range under load, and its per-probe cost")
    tel.add_argument("--load-procs", type=int, default=4)
    tel.add_argument("--phase-s", type=float, default=3.0)
    tel.add_argument("--out", default=str(RESULTS_DIR / "cpu_telemetry_probe.json"))

    sub.add_parser("network", help="record the connection type and metered status")
    sub.add_parser("cores", help="logical-to-physical core mapping and SMT siblings")
    sub.add_parser("power", help="AC or battery; a run on battery is INVALID outright")

    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.cmd == "query":
        print(json.dumps(query_clock_power_state(), indent=2))
        return 0

    if args.cmd == "sustained":
        load = args.load.split() if args.load else None
        data = log_sustained_load(args.duration, args.interval, load)
        Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"5.3 log written: {args.out}")
        print(f"  peak temperature      : {data['peak_temperature_c']} C")
        print(f"  stabilization         : {data['stabilization']}")
        print(f"  throttle reasons seen : {data['throttle_reasons_seen']}")
        print(f"  max safe continuous   : {data['max_safe_continuous_benchmark_s']}")
        return 0

    if args.cmd == "stability":
        data = {
            "vram_integrity": vram_integrity(buffer_mib=args.vram_mib),
            "compute_determinism": compute_determinism(),
            "background_load": enumerate_background_load(),
            "timing_reproducibility":
                "run bench/microbench/run_all.py twice and pass the two "
                "consolidated files to spread_percent()",
        }
        Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"5.4 results written: {args.out}")
        vi = data["vram_integrity"]
        if vi.get("ok") is False:
            print("VRAM INTEGRITY FAILED -- HARD STOP. "
                  "Drop to stock clocks before any further measurement.",
                  file=sys.stderr)
            return 3
        return 0

    if args.cmd == "fingerprint":
        fp = capture_fingerprint()
        Path(args.out).write_text(json.dumps(fp, indent=2), encoding="utf-8")
        print(f"5.5 fingerprint written: {args.out}")
        return 0

    if args.cmd == "verify":
        stored_path = Path(args.stored)
        if not stored_path.exists():
            print(f"no stored fingerprint at {stored_path}", file=sys.stderr)
            return 2
        result = verify_fingerprint(capture_fingerprint(),
                                    json.loads(stored_path.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2))
        return 0 if result["match"] else 1

    if args.cmd == "telemetry":
        data = probe_cpu_telemetry(load_procs=args.load_procs, phase_s=args.phase_s)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"CPU telemetry probe written: {args.out}")
        for r in data["sources"]:
            cost = r.get("per_probe_us")
            cost_s = (f"{cost:.1f} us/probe" if isinstance(cost, (int, float))
                      else "cost unknown")
            print(f"  {r['source']:45} {r['verdict']:12} {cost_s}")
            ph = r.get("phases")
            if ph:
                print(f"      idle {ph['idle']['min']:.6g}..{ph['idle']['max']:.6g}  "
                      f"load {ph['load']['min']:.6g}..{ph['load']['max']:.6g}  "
                      f"recovery {ph['recovery']['min']:.6g}..{ph['recovery']['max']:.6g}"
                      f"  [{r.get('unit')}]")
            if r.get("reason"):
                print(f"      {r['reason']}")
        print(f"  live CPU frequency source   : {data['live_cpu_frequency_source']}")
        print(f"  live CPU package temperature: {data['live_cpu_package_temperature_source']}")
        return 0

    if args.cmd == "network":
        print(json.dumps(network_state(), indent=2))
        return 0

    if args.cmd == "cores":
        m = logical_core_mapping()
        print(json.dumps(m, indent=2))
        print(json.dumps(sampler_affinity_mask((0, 2), m), indent=2))
        return 0

    if args.cmd == "power":
        p = power_source()
        print(json.dumps(p, indent=2))
        return 0 if p.get("on_ac") else 3

    if args.cmd == "verify-lock":
        load = args.load.split() if args.load else None
        r = verify_clock_lock(args.mhz, load_cmd=load)
        print(json.dumps(r, indent=2))
        return 0 if r["locked"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
