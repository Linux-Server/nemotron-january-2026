#!/usr/bin/env python3
"""1.3b-enc-scale: strip the dead in-package weights (data/weights/*) from every finalize bucket .pt2
(constants-on-disk; weights never loaded — load_constants supplies them) into stripped_finalize_buckets/. Zeroes the
weight zip entries (keeps structure) -> each bucket 2.48GB -> ~4MB. Validated token-exact by strip_bucket_weights.py +
the corpus harness. Run from runtime/."""
import os, glob, zipfile, shutil
BD="artifacts/finalize_buckets"; OUT="artifacts/stripped_finalize_buckets"; os.makedirs(OUT, exist_ok=True)
def is_w(n): return "/data/weights/" in n
for src in sorted(glob.glob(os.path.join(BD,"enc_finalize_d*_T*.pt2"))):
    if src.endswith("_ep.pt2"): continue
    dst=os.path.join(OUT, os.path.basename(src))
    with zipfile.ZipFile(src,"r") as zin, zipfile.ZipFile(dst,"w",allowZip64=True) as zout:
        for info in zin.infolist():
            if is_w(info.filename):
                ci=zipfile.ZipInfo(info.filename, date_time=info.date_time); ci.compress_type=info.compress_type
                zout.writestr(ci, b"")
            else:
                zout.writestr(info, zin.read(info.filename))
    print(f"  stripped {os.path.basename(src)}: {os.path.getsize(src)/1e9:.2f}GB -> {os.path.getsize(dst)/1e6:.1f}MB")
