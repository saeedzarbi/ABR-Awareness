#!/usr/bin/env bash
# =============================================================================
# v19 video data pipeline: download -> encode -> VMAF -> SI/TI -> (optional) multires
#
# Produces artifacts under data/:
#   raw_videos/{slug}.mp4
#   encoded_videos/{slug}/{bitrate}kbps.mp4
#   vmaf_scores/vmaf_summary.csv
#   content_features/{slug}_siti.json + siti_summary.csv
#   vmaf_scores/vmaf_perchunk_multires.csv   (if BUILD_MULTIRES=1)
#
# Usage (Linux server with ffmpeg + libvmaf + xz):
#   cd new
#   bash data/run_v19_video_pipeline.sh
#
# Skip re-download if references already exist:
#   SKIP_DOWNLOAD=1 bash data/run_v19_video_pipeline.sh
#
# Fast VMAF (720p subsampling on low rungs; good for first pass):
#   FAST_VMAF=1 bash data/run_v19_video_pipeline.sh
#
# Env: PYTHON, RAW_DIR, SKIP_DOWNLOAD, FAST_VMAF, BUILD_MULTIRES, PARALLEL_VMAF
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PY="${PYTHON:-python}"
RAW_DIR="${RAW_DIR:-data/raw_videos}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
FAST_VMAF="${FAST_VMAF:-0}"
BUILD_MULTIRES="${BUILD_MULTIRES:-0}"
PARALLEL_VMAF="${PARALLEL_VMAF:-0}"

VIDEOS="$("${PY}" -c "from configs.videos import EVAL_VIDEOS_CSV; print(EVAL_VIDEOS_CSV)")"
N_VIDEOS="$("${PY}" -c "from configs.videos import ALL_VIDEOS; print(len(ALL_VIDEOS))")"

echo "=============================================================="
echo "v19 video pipeline (${N_VIDEOS} titles)"
echo "Root=${ROOT_DIR}  raw=${RAW_DIR}"
echo "Videos: ${VIDEOS}"
echo "=============================================================="

if [ "${SKIP_DOWNLOAD}" != "1" ]; then
    echo ""
    echo "[1/5] Download + prepare raw references"
    RAW_DIR="${RAW_DIR}" bash data/download_raw_videos.sh
else
    echo ""
    echo "[1/5] SKIP_DOWNLOAD=1 — using existing ${RAW_DIR}/"
fi

echo ""
echo "[2/5] Encode 6-rung ladder (per video)"
for v in $(echo "${VIDEOS}" | tr ',' ' '); do
    echo "  -> encode ${v}"
    "${PY}" data/video_encoder.py --video "${v}" \
        --input-dir "${RAW_DIR}" --output-dir data/encoded_videos
done

echo ""
echo "[3/5] Session-mean VMAF -> data/vmaf_scores/vmaf_summary.csv"
VMAF_ARGS=(data/vmaf_calculator.py --videos "${VIDEOS}")
if [ "${FAST_VMAF}" = "1" ]; then
    VMAF_ARGS+=(--fast)
fi
if [ "${PARALLEL_VMAF}" = "1" ]; then
    VMAF_ARGS+=(--parallel)
fi
"${PY}" "${VMAF_ARGS[@]}"

echo ""
echo "[4/5] SI/TI features -> data/content_features/"
"${PY}" data/si_ti_extractor.py --videos "${VIDEOS}" --video-dir "${RAW_DIR}"

if [ "${BUILD_MULTIRES}" = "1" ]; then
    echo ""
    echo "[5/5] Per-chunk multi-resolution VMAF ladder"
    if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q libvmaf; then
        echo "[FATAL] ffmpeg lacks libvmaf; cannot build multires ladder."; exit 2
    fi
    "${PY}" data/build_multires_vmaf.py --raw-dir "${RAW_DIR}" --videos "${VIDEOS}"
else
    echo ""
    echo "[5/5] Skipping multires ladder (set BUILD_MULTIRES=1 to enable)"
fi

echo ""
echo "=============================================================="
echo "v19 pipeline complete."
echo "  VMAF summary : data/vmaf_scores/vmaf_summary.csv"
echo "  SI/TI summary: data/content_features/siti_summary.csv"
echo ""
echo "Next: CPS eval on model-agnostic baselines (no PPO retrain required):"
echo "  cd new && bash src/training/run_certified_v18.sh"
echo "=============================================================="
