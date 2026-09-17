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

    if args.cmd == "verify-lock":
        load = args.load.split() if args.load else None
        r = verify_clock_lock(args.mhz, load_cmd=load)
        print(json.dumps(r, indent=2))
        return 0 if r["locked"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
