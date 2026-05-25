#!/usr/bin/env python3
"""1.3b-enc-build (container side): AOTI-compile each finalize bucket ExportedProgram (from export_finalize_buckets.py)
constants-on-disk (tiny wrapper .so, NO embedded weights) so all buckets share ONE weight set at load via
loader.load_constants(user_managed=True) — see validate_shared_weights.py. noexecstack link shim (hardened-host dlopen).

Run in-container: python3 aot_compile_buckets.py --dir ./artifacts/finalize_buckets
"""
from __future__ import annotations
import argparse, json, os, torch


def _force_noexecstack_on_link():
    import torch._inductor.cpp_builder as cb
    orig = cb.CppBuilder.get_command_line
    fired = {"ok": False}
    def patched(self):
        cmd = orig(self)
        if getattr(self, "_do_link", False) and "-shared" in cmd and "-Wl,-z,noexecstack" not in cmd:
            cmd += " -Wl,-z,noexecstack"; fired["ok"] = True
        return cmd
    cb.CppBuilder.get_command_line = patched
    return fired


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="./artifacts/finalize_buckets"); a = ap.parse_args()
    fired = _force_noexecstack_on_link()
    manifest = json.load(open(os.path.join(a.dir, "buckets_manifest.json")))
    out_manifest = []
    for b in manifest:
        ep = torch.export.load(os.path.join(a.dir, b["ep"]))
        pkg = os.path.join(a.dir, f"enc_finalize_d{b['drop']}_T{b['T']}.pt2")
        torch._inductor.aoti_compile_and_package(
            ep, package_path=pkg,
            inductor_configs={"aot_inductor.package_constants_in_so": False,
                              "aot_inductor.package_constants_on_disk": True})
        so_mb = os.path.getsize(pkg) / 1e6
        b2 = dict(b); b2["pkg"] = os.path.basename(pkg); b2["pkg_mb"] = round(so_mb, 1)
        out_manifest.append(b2)
        print(f"  compiled drop={b['drop']} T={b['T']} -> {os.path.basename(pkg)} ({so_mb:.1f} MB)")
    assert fired["ok"], "noexecstack shim never fired on a shared-lib link"
    json.dump(out_manifest, open(os.path.join(a.dir, "buckets_manifest.json"), "w"), indent=2)
    print(f"=== compiled {len(out_manifest)} constants-on-disk finalize buckets (share finalize_shared_weights.pt) ===")


if __name__ == "__main__":
    main()
