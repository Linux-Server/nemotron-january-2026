#!/usr/bin/env python3
"""F' (action-D review #5) — does ANY AOTI accuracy/precision/determinism knob recover byte-exactness vs eager? Default
AOTI fails the T2a byte-exact bar (cache_t 1.66e-2). The review correctly flagged that "unreachable in any language" was
asserted with ZERO knobs tried. This compiles the SAME ExportedProgram under a matrix of knobs and compares each against
the FIXED eager reference (t2a_io.pt, the default-eager output the production path produces) — eager is the target, only
the compile varies. Honest deliverable: the matrix, byte-exact recovered or not.

Run in-container: docker run ... nemotron-aoti:cu128 python3 aot_knob_matrix.py
"""
from __future__ import annotations
import os, tempfile, torch

ART = os.path.join(os.path.dirname(__file__), "artifacts")
NAMES = ["enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"]

def noexec_shim():
    import torch._inductor.cpp_builder as cb
    orig = cb.CppBuilder.get_command_line
    def patched(self):
        cmd = orig(self)
        if getattr(self, "_do_link", False) and "-shared" in cmd and "-Wl,-z,noexecstack" not in cmd:
            cmd += " -Wl,-z,noexecstack"
        return cmd
    cb.CppBuilder.get_command_line = patched

def set_inductor(cfg: dict):
    """Set only the inductor config keys that actually exist in this torch; report skips."""
    import torch._inductor.config as c
    applied, skipped = {}, []
    for k, v in cfg.items():
        obj, attr = c, k
        if "." in k:
            ns, attr = k.split(".", 1)
            obj = getattr(c, ns, None)
            if obj is None: skipped.append(k); continue
        if hasattr(obj, attr.split(".")[0]):
            try:
                # support one level of nesting (triton.x)
                if "." in attr:
                    sub, leaf = attr.split(".", 1); setattr(getattr(obj, sub), leaf, v)
                else:
                    setattr(obj, attr, v)
                applied[k] = v
            except Exception as e:
                skipped.append(f"{k}({e})")
        else:
            skipped.append(k)
    return applied, skipped

def compare(runner, io):
    ins = [io["chunk"].cuda(), io["L"].cuda(), io["clc"].cuda(), io["clt"].cuda(), io["clcl"].cuda()]
    with torch.inference_mode(): out = runner(*ins)
    outs = list(out) if isinstance(out, (list, tuple)) else [out]
    ref = [t.cuda() for t in io["out"]]
    allok = True; maxd = 0.0; per = {}
    for n, a, b in zip(NAMES, ref, outs):
        eq = (a.shape == b.shape) and torch.equal(a, b)
        d = (a.float() - b.float()).abs().max().item() if a.shape == b.shape and a.numel() else float("nan")
        allok &= eq; maxd = max(maxd, d if d == d else 0.0); per[n] = (eq, d)
    return allok, maxd, per

def build_and_check(ep, io, label, inductor_cfg, torch_globals):
    # torch globals (precision/determinism) — set BEFORE compile
    note = []
    if torch_globals.get("matmul_precision"):
        torch.set_float32_matmul_precision(torch_globals["matmul_precision"]); note.append(f"matmul={torch_globals['matmul_precision']}")
    if "allow_tf32" in torch_globals:
        torch.backends.cuda.matmul.allow_tf32 = torch_globals["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = torch_globals["allow_tf32"]; note.append(f"tf32={torch_globals['allow_tf32']}")
    if torch_globals.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True); note.append("deterministic")
    applied, skipped = set_inductor(inductor_cfg)
    try:
        with tempfile.TemporaryDirectory() as td:
            pkg = os.path.join(td, f"{label}.pt2")
            cfgs = {"aot_inductor.package": True}
            cfgs.update(inductor_cfg)
            out_path = torch._inductor.aoti_compile_and_package(ep, package_path=pkg, inductor_configs=cfgs)
            runner = torch._inductor.aoti_load_package(out_path)
            allok, maxd, per = compare(runner, io)
    except Exception as e:
        print(f"[{label}] COMPILE/RUN FAILED: {type(e).__name__}: {str(e)[:200]}"); return
    tag = "BYTE-EXACT" if allok else f"diff {maxd:.3e}"
    print(f"[{label}] {tag}  (applied={applied} skipped={skipped} globals={note})")
    print(f"    enc_out={per['enc_out'][1]:.2e} cache_ch={per['cache_ch'][1]:.2e} cache_t={per['cache_t'][1]:.2e}")

# each knob runs in its OWN process (driver: aot_knob_run.sh) to avoid global-state leakage across compiles
MATRIX = {
    "default":            ({}, {}),
    "fp32_highest":       ({}, {"matmul_precision": "highest", "allow_tf32": False}),
    "emulate_prec_casts": ({"emulate_precision_casts": True}, {"matmul_precision": "highest", "allow_tf32": False}),
    "no_fusion_autotune": ({"epilogue_fusion": False, "max_autotune": False, "coordinate_descent_tuning": False},
                           {"matmul_precision": "highest", "allow_tf32": False}),
    "deterministic_all":  ({"emulate_precision_casts": True, "epilogue_fusion": False},
                           {"matmul_precision": "highest", "allow_tf32": False, "deterministic": True}),
}

def main():
    import sys
    knob = sys.argv[1] if len(sys.argv) > 1 else "default"
    ind, glob = MATRIX[knob]
    noexec_shim()
    print("torch", torch.__version__, torch.cuda.get_device_capability(), "| knob:", knob)
    ep = torch.export.load(os.path.join(ART, "enc_steady_t2a.pt2"))
    io = torch.load(os.path.join(ART, "t2a_io.pt"), weights_only=False)
    print("eager-ref defaults: matmul_precision=", torch.get_float32_matmul_precision(),
          "tf32_matmul=", torch.backends.cuda.matmul.allow_tf32)
    build_and_check(ep, io, knob, ind, glob)

if __name__ == "__main__":
    main()
