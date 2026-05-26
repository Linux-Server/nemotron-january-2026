#!/usr/bin/env bash
# Upload the Phase-2 sm_89-compile source (EPs + steady EP + bundle + shared weights) to S3 with a MANIFEST.
# Durable backup (Q1 fix) + the g6e autotune-on compile source. Run from runtime/.
set -euo pipefail
PROFILE=AWSAdministratorAccess-419599258555
BUCKET=nemotron-phase2-eps-419599258555
PREFIX=density
ART=artifacts
S3=s3://$BUCKET/$PREFIX
log(){ echo "[s3up $(date +%H:%M:%S)] $*"; }

cd "$(dirname "$0")"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
MAN=/tmp/eps_manifest.json

log "sha256 + sizes (this hashes ~80GB; takes a few min)"
{
  echo "{"
  echo "  \"generated_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"git_commit\": \"$COMMIT\","
  echo "  \"regen_command\": \"HF_HUB_OFFLINE=1 \$PARAKEET_VENV/bin/python export_finalize_buckets.py + build_drop0_buckets.py + build_range_examples.py (host/nemo); steady = enc_steady_t2a.pt2 reused\","
  echo "  \"fail_closed\": \"the regen MUST reproduce exactly the 32 (drop,T) contract keys in buckets_manifest.json\","
  echo "  \"contract\": $(cat $ART/finalize_buckets/buckets_manifest.json),"
  echo "  \"files\": ["
  first=1
  for f in $ART/enc_steady_t2a.pt2 $ART/session_bundle.ts $ART/finalize_shared_weights.pt $ART/finalize_shared_weights.ts $ART/finalize_buckets/*_ep.pt2; do
    [ -f "$f" ] || continue
    sz=$(stat -c %s "$f")
    sha=$(sha256sum "$f" | cut -d' ' -f1)
    [ $first -eq 1 ] || echo ","
    first=0
    printf '    {"path": "%s", "bytes": %s, "sha256": "%s"}' "$f" "$sz" "$sha"
  done
  echo ""
  echo "  ]"
  echo "}"
} > "$MAN"
log "manifest -> $MAN ($(wc -c < "$MAN") bytes, $(grep -c '"sha256"' "$MAN") files)"

log "upload manifest + contract first"
aws s3 cp "$MAN" "$S3/eps_manifest.json" --profile "$PROFILE"
aws s3 cp "$ART/finalize_buckets/buckets_manifest.json" "$S3/buckets_manifest.json" --profile "$PROFILE"

log "upload steady EP + bundle + shared weights"
aws s3 cp "$ART/enc_steady_t2a.pt2" "$S3/enc_steady_t2a.pt2" --profile "$PROFILE"
aws s3 cp "$ART/session_bundle.ts" "$S3/session_bundle.ts" --profile "$PROFILE"
aws s3 cp "$ART/finalize_shared_weights.pt" "$S3/finalize_shared_weights.pt" --profile "$PROFILE"
aws s3 cp "$ART/finalize_shared_weights.ts" "$S3/finalize_shared_weights.ts" --profile "$PROFILE"

log "upload the 32 finalize bucket EPs (~75GB; multipart)"
aws s3 cp "$ART/finalize_buckets/" "$S3/finalize_buckets/" --recursive --exclude "*" --include "*_ep.pt2" --profile "$PROFILE"

log "DONE. listing:"
aws s3 ls "$S3/" --recursive --human-readable --summarize --profile "$PROFILE" | tail -8
