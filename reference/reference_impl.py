"""reference_impl.py -- the PyTorch correctness oracle for the C engine.

WHAT AGREEMENT BETWEEN THIS FILE AND src/model.c PROVES, AND WHAT IT DOES NOT.
Both were written in one session from one reading of the architecture. Agreement
is therefore evidence that the two implementations agree; it is NOT evidence
that either matches GPT-2. The external discriminator available is readable
English generated from published weights, and its resolution must be stated with
it: it catches a transposed weight, an untied head, a broken causal mask and a
wrong positional scheme, because any of those produces garbage. It does NOT
catch a wrong layernorm epsilon, a plain gelu substituted for gelu_new, or a
mildly wrong attention scale. What IS externally validated, independently of
this stage, is the input side: the Stage 1 tokenizer was verified against the
real `tokenizers` library across 48 records and 410 ids, and the Stage 1 loader
is byte-exact against real safetensors across 160 of 160 tensors.

SHARES THE CONFIG ARTIFACT, SHARES NO ARITHMETIC. Every architecture value comes
from models/gpt2/config.json, the same file the C engine reads, which removes an
error mode. Nothing here imports, calls or transcribes anything on the C side,
which would create one.

WEIGHT READING -- THE BRANCH TAKEN, AND WHY. `import safetensors` FAILS in the
project .venv (torch 2.14.0+cu130, numpy 2.5.3, Python 3.14.2); the library
lives only in .venv-oracle, which has no torch. Nothing is installed into either
environment -- both are frozen. This file therefore takes the second branch: it
reads the file with the standard library and numpy (8-byte little-endian header
length, JSON header, numpy.frombuffer at the absolute file offset), and asserts
its own parse against src/gpt2_tensor_inventory.json for ALL 160 tensors on
name, dtype, shape, absolute file offset and byte length before using any of
them. That cross-check is trustworthy rather than circular because the C loader
that produced the inventory was itself proven byte-exact against the real
safetensors library in Stage 1.

The data segment begins at file offset 14291, which is not a multiple of 8, 16
or 64, and the file carries no padding. Every tensor is therefore read by
seeking and reading into a fresh bytes object, never by slicing a memory map.

CONDITIONS, RECORDED BECAUSE THEY CHANGE THE NUMBERS THIS ORACLE PRODUCES:
CPU only, and torch.set_num_threads(1). A CUDA oracle would add reduction-order
divergence of its own to the very quantity this stage measures; a single-threaded
CPU reduction order is deterministic where a parallel one is not; and the GPU
carries a clock lock for the timed runs, so keeping the oracle off the device
removes a class of interference. Thread count is a recorded condition, not an
implementation detail.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO, "models", "gpt2", "model.safetensors")
DEFAULT_CONFIG = os.path.join(REPO, "models", "gpt2", "config.json")
DEFAULT_INVENTORY = os.path.join(REPO, "src", "gpt2_tensor_inventory.json")

DTYPE_NUMPY = {"F32": np.float32}


# --------------------------------------------------------------- weight file --

def read_header(path: str):
    """Returns (header dict, absolute data-segment offset). Standard library
    only: an 8-byte little-endian length prefix, then that many bytes of JSON."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


def parse_tensors(header: dict, data_offset: int):
    """The reader's own view of the file: name -> dtype, shape, absolute offset,
    byte length. Derived from the header alone, with nothing taken from the
    inventory, so the comparison against the inventory is a real comparison."""
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        out[name] = {
            "dtype": entry["dtype"],
            "shape": list(entry["shape"]),
            "file_offset": data_offset + begin,
            "nbytes": end - begin,
        }
    return out


def verify_against_inventory(parsed: dict, inventory_path: str):
    """Asserts this reader's parse equals the committed Stage 1 inventory for
    ALL tensors, on name, dtype, shape, absolute file offset and byte length.
    Returns the number of tensors checked. Raises AssertionError on the first
    disagreement, naming it."""
    with open(inventory_path, "r", encoding="utf-8") as f:
        inv = json.load(f)
    rows = {t["name"]: t for t in inv["tensors"]}
    assert len(rows) == inv["tensor_count"], (
        f"the inventory declares {inv['tensor_count']} tensors and lists {len(rows)}")
    assert set(rows) == set(parsed), (
        "the name sets differ: only in the inventory "
        f"{sorted(set(rows) - set(parsed))[:5]}, only in this parse "
        f"{sorted(set(parsed) - set(rows))[:5]}")
    for name, row in rows.items():
        mine = parsed[name]
        assert mine["dtype"] == row["dtype"], f"{name}: dtype {mine['dtype']} vs {row['dtype']}"
        assert mine["shape"] == list(row["shape"]), (
            f"{name}: shape {mine['shape']} vs {row['shape']}")
        assert mine["file_offset"] == row["file_offset"], (
            f"{name}: file_offset {mine['file_offset']} vs {row['file_offset']}")
        assert mine["nbytes"] == row["nbytes"], (
            f"{name}: nbytes {mine['nbytes']} vs {row['nbytes']}")
    return len(rows)


def read_tensor(path: str, spec: dict) -> np.ndarray:
    """Seeks to the absolute offset and reads into a FRESH bytes object. The data
    segment is unaligned and unpadded, so a view into a memory map is not safe to
    reinterpret as float32."""
    with open(path, "rb") as f:
        f.seek(spec["file_offset"])
        raw = f.read(spec["nbytes"])
    assert len(raw) == spec["nbytes"], f"short read at {spec['file_offset']}"
    arr = np.frombuffer(raw, dtype=DTYPE_NUMPY[spec["dtype"]])
    return arr.reshape(spec["shape"]).copy()


# -------------------------------------------------------------- the model ----

class ReferenceGPT2:
    """GPT-2 written out in torch, layer by layer, from the config's values and
    the shapes the weight file declares. `transformers` is not available in this
    environment and is not used."""

    def __init__(self, weights=DEFAULT_WEIGHTS, config=DEFAULT_CONFIG,
                 inventory=DEFAULT_INVENTORY, cproj_transposed=False,
                 verify=True):
        torch.set_num_threads(1)
        self.threads = torch.get_num_threads()
        self.device = torch.device("cpu")

        with open(config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.n_layer = int(cfg["n_layer"])
        self.n_head = int(cfg["n_head"])
        self.n_embd = int(cfg["n_embd"])
        self.n_ctx = int(cfg["n_ctx"])
        self.vocab_size = int(cfg["vocab_size"])
        self.eps = float(cfg["layer_norm_epsilon"])
        self.activation = str(cfg["activation_function"])
        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head
        assert self.activation == "gelu_new", (
            f"this oracle implements gelu_new; the config names {self.activation}")

        header, data_offset = read_header(weights)
        self.parsed = parse_tensors(header, data_offset)
        self.n_verified = verify_against_inventory(self.parsed, inventory) if verify else 0
        self.cproj_transposed = bool(cproj_transposed)

        def get(name):
            return torch.from_numpy(read_tensor(weights, self.parsed[name]))

        self.wte = get("wte.weight")
        self.wpe = get("wpe.weight")
        # Layer shapes are derived from the config and checked against the shapes
        # the FILE declares, rather than assumed from a convention.
        assert tuple(self.wte.shape) == (self.vocab_size, self.n_embd)
        assert tuple(self.wpe.shape) == (self.n_ctx, self.n_embd)
        self.ln_f = (get("ln_f.weight"), get("ln_f.bias"))

        E = self.n_embd
        self.blocks = []
        for i in range(self.n_layer):
            b = {
                "ln_1_w": get(f"h.{i}.ln_1.weight"),
                "ln_1_b": get(f"h.{i}.ln_1.bias"),
                "attn_w": get(f"h.{i}.attn.c_attn.weight"),
                "attn_b": get(f"h.{i}.attn.c_attn.bias"),
                "proj_w": get(f"h.{i}.attn.c_proj.weight"),
                "proj_b": get(f"h.{i}.attn.c_proj.bias"),
                "ln_2_w": get(f"h.{i}.ln_2.weight"),
                "ln_2_b": get(f"h.{i}.ln_2.bias"),
                "fc_w": get(f"h.{i}.mlp.c_fc.weight"),
                "fc_b": get(f"h.{i}.mlp.c_fc.bias"),
                "mlp_proj_w": get(f"h.{i}.mlp.c_proj.weight"),
                "mlp_proj_b": get(f"h.{i}.mlp.c_proj.bias"),
            }
            # Each rectangular weight is pinned to [input, output] by its own
            # bias length -- arithmetic on the file's shapes, assuming no
            # convention. The square attention projection is NOT pinned that way
            # and its reading is a parameter.
            assert tuple(b["attn_w"].shape) == (E, 3 * E) and b["attn_b"].shape[0] == 3 * E
            assert tuple(b["fc_w"].shape) == (E, 4 * E) and b["fc_b"].shape[0] == 4 * E
            assert tuple(b["mlp_proj_w"].shape) == (4 * E, E) and b["mlp_proj_b"].shape[0] == E
            assert tuple(b["proj_w"].shape) == (E, E)
            if self.cproj_transposed:
                b["proj_w"] = b["proj_w"].t().contiguous()
            self.blocks.append(b)

        # The head is TIED: no tensor in the file is named lm_head and exactly
        # one matrix carries the vocabulary size against the embedding width.
        self.head = self.wte

    # -- pieces, written out rather than called from a model library ----------

    def layernorm(self, x, w, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * w + b

    @staticmethod
    def gelu_new(x):
        c = 0.7978845608028654  # sqrt(2/pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))

    @torch.no_grad()
    def forward(self, ids):
        """ids: a sequence of token ids. Returns logits for EVERY position, as a
        [T, vocab_size] float32 tensor -- matching the C engine's prefill, which
        also computes the head at every position. A decode step is the same
        computation with the head applied only at the last row."""
        ids = torch.as_tensor(list(ids), dtype=torch.long)
        T = ids.shape[0]
        assert T <= self.n_ctx
        H, D = self.n_head, self.head_dim

        x = self.wte[ids] + self.wpe[:T]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))

        for b in self.blocks:
            h = self.layernorm(x, b["ln_1_w"], b["ln_1_b"])
            qkv = h @ b["attn_w"] + b["attn_b"]
            q, k, v = qkv.split(self.n_embd, dim=-1)
            q = q.view(T, H, D).permute(1, 0, 2)
            k = k.view(T, H, D).permute(1, 0, 2)
            v = v.view(T, H, D).permute(1, 0, 2)
            att = (q @ k.transpose(-1, -2)) * (1.0 / (D ** 0.5))
            att = att.masked_fill(~causal, float("-inf"))
            att = torch.softmax(att, dim=-1)
            y = (att @ v).permute(1, 0, 2).reshape(T, self.n_embd)
            x = x + (y @ b["proj_w"] + b["proj_b"])

            h2 = self.layernorm(x, b["ln_2_w"], b["ln_2_b"])
            ff = self.gelu_new(h2 @ b["fc_w"] + b["fc_b"])
            x = x + (ff @ b["mlp_proj_w"] + b["mlp_proj_b"])

        x = self.layernorm(x, self.ln_f[0], self.ln_f[1])
        return x @ self.head.t()          # the tied head

    @torch.no_grad()
    def greedy(self, ids, n_new):
        """Greedy generation with no KV cache, mirroring the engine's structure:
        the whole forward pass is re-run for every generated token. No sampling,
        no seed, no PRNG."""
        out = list(ids)
        for _ in range(n_new):
            logits = self.forward(out)
            out.append(int(torch.argmax(logits[-1]).item()))
        return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--inventory", default=DEFAULT_INVENTORY)
    p.add_argument("--ids", help="comma-separated token ids")
    p.add_argument("--generate", type=int, default=0)
    p.add_argument("--cproj", choices=("as-stored", "transposed"), default="as-stored")
    p.add_argument("--dump", help="write the logits as .npy")
    a = p.parse_args(argv)

    model = ReferenceGPT2(a.weights, a.config, a.inventory,
                          cproj_transposed=(a.cproj == "transposed"))
    print(f"verified {model.n_verified} tensors against the inventory; "
          f"torch {torch.__version__}, threads {model.threads}, device cpu")
    if not a.ids:
        return 0
    ids = [int(s) for s in a.ids.split(",") if s.strip()]
    if a.generate:
        seq = model.greedy(ids, a.generate)
        print("ids:", ",".join(str(i) for i in seq))
        return 0
    logits = model.forward(ids).numpy()
    print(f"logits {logits.shape} dtype {logits.dtype}")
    if a.dump:
        np.save(a.dump, logits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
