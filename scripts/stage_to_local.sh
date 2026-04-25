#!/usr/bin/env bash
# Stage the preprocessed LibriSpeech cache from GPFS scratch to local NVMe so
# the dataloader hits a local filesystem instead of a parallel network FS.
#
# On parallel filesystems (GPFS / Lustre), batch-read latency is bursty: the
# dataloader workers stall on coordinated reads and the GPU drops to 0% util
# for 1-2 second windows. Staging the dataset to per-node NVMe (typically
# /tmp on OSC Ascend / Pitzer) eliminates that variance.
#
# Idempotent: if the destination already contains the source's bytes, skips
# the copy. Safe to invoke before every training run.
#
# Usage:
#   scripts/stage_to_local.sh                                # use defaults
#   scripts/stage_to_local.sh --src /fs/scratch/.../hf_cache --dst /tmp/my_cache
#   SRC=/fs/scratch/.../hf_cache DST=/tmp/my_cache scripts/stage_to_local.sh
#
# After it finishes, the script prints the exact --cache_dir flag to pass to
# scripts/train.py.

set -euo pipefail

SRC="${SRC:-/fs/scratch/PAS2836/lees_stuff/hf_cache}"
DST="${DST:-/tmp/leedavis_hf_cache}"

while [ $# -gt 0 ]; do
    case "$1" in
        --src) SRC="$2"; shift 2 ;;
        --dst) DST="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SRC_PRE="$SRC/preprocessed"
DST_PRE="$DST/preprocessed"

if [ ! -d "$SRC_PRE" ]; then
    echo "[stage] source not found: $SRC_PRE" >&2
    echo "[stage] expected the preprocessed/ subdir under --src" >&2
    exit 1
fi

# /tmp on cluster nodes is typically tmpfs or per-node NVMe — check it exists
# and has enough room. We require dst's parent to exist; we'll create dst.
DST_PARENT=$(dirname "$DST")
if [ ! -d "$DST_PARENT" ]; then
    echo "[stage] destination parent does not exist: $DST_PARENT" >&2
    exit 1
fi

mkdir -p "$DST_PRE"

# Each preprocessed cache is a directory like all960-<hash>/. Stage every
# subdir under preprocessed/ — usually only one, but the user may keep a few
# variants (different speed_perturbations / subsets) cached side by side.
shopt -s nullglob
SRC_ENTRIES=("$SRC_PRE"/*/)
shopt -u nullglob
if [ ${#SRC_ENTRIES[@]} -eq 0 ]; then
    echo "[stage] no preprocessed/<key>/ subdirs found under $SRC_PRE" >&2
    exit 1
fi

# Free-space check (GiB), with 2 GiB safety margin.
need_kb=$(du -sk "${SRC_ENTRIES[@]}" 2>/dev/null | awk '{s+=$1} END{print s}')
need_gb=$(awk -v k="$need_kb" 'BEGIN{printf "%.1f", k/1024/1024}')
avail_kb=$(df -k --output=avail "$DST_PARENT" | tail -1 | tr -d ' ')
avail_gb=$(awk -v k="$avail_kb" 'BEGIN{printf "%.1f", k/1024/1024}')
if (( avail_kb < need_kb + 2*1024*1024 )); then
    echo "[stage] not enough space at $DST_PARENT" >&2
    echo "[stage]   need ${need_gb} GiB + 2 GiB margin; have ${avail_gb} GiB" >&2
    exit 1
fi
echo "[stage] source: $SRC_PRE  (${need_gb} GiB)"
echo "[stage] dest:   $DST_PRE  (${avail_gb} GiB free)"

copied_anything=0
for entry in "${SRC_ENTRIES[@]}"; do
    name=$(basename "$entry")
    src_dir="${entry%/}"
    dst_dir="$DST_PRE/$name"

    # Compare byte counts to detect a complete prior stage. du -sb is fast
    # enough on local NVMe; for the GPFS source we already have it from the
    # need_kb tally if it's the only entry, but recomputing per-entry keeps
    # the loop simple.
    src_bytes=$(du -sb "$src_dir" | awk '{print $1}')
    if [ -d "$dst_dir" ]; then
        dst_bytes=$(du -sb "$dst_dir" 2>/dev/null | awk '{print $1}')
        if [ "$src_bytes" = "$dst_bytes" ]; then
            echo "[stage] skip $name (already staged, $src_bytes bytes)"
            continue
        fi
        echo "[stage] dst $name exists but size differs (src=$src_bytes dst=$dst_bytes); restaging"
        rm -rf "$dst_dir"
    fi

    echo "[stage] copying $name ..."
    t0=$(date +%s)
    cp -r "$src_dir" "$dst_dir"
    t1=$(date +%s)
    secs=$((t1 - t0))
    mbs=$(awk -v b="$src_bytes" -v s="$secs" 'BEGIN{if(s<=0){s=1}; printf "%.0f", b/1024/1024/s}')
    echo "[stage] $name done in ${secs}s (${mbs} MB/s)"
    copied_anything=1
done

if [ "$copied_anything" -eq 0 ]; then
    echo "[stage] nothing to do — destination is up to date"
fi

echo
echo "[stage] ready. Run training with:"
echo "    --cache_dir $DST"
echo
echo "[stage] e.g.:"
echo "    .a100/bin/python scripts/train.py --config configs/conformer_c2x_whisper.yaml --cache_dir $DST"
