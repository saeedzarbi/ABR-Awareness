#!/usr/bin/env bash
# =============================================================================
# Download + prepare the 4 source videos for the V16 multi-resolution VMAF ladder.
# Produces (1080p, audio stripped, clean H.264) references under raw_videos/:
#   bigbuckbunny.mp4  crowd_run.mp4  tearsofsteel_short.mp4  sintel.mp4
#
# Total download ~2.7 GB. Requires: ffmpeg, and wget OR curl, and python (for
# unzip via `python -m zipfile`). Re-running skips files that already exist.
#
# Usage (on the server):
#   cd new
#   bash data/download_raw_videos.sh
#   # then:  bash src/training/run_multires_v16.sh
#
# Env vars: RAW_DIR (default raw_videos), TMP (default .raw_downloads),
#           PYTHON (default python), KEEP_TMP=1 to keep downloads.
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> new/
cd "${ROOT_DIR}"

RAW_DIR="${RAW_DIR:-raw_videos}"
TMP="${TMP:-.raw_downloads}"
PY="${PYTHON:-python}"
CRF="${CRF:-12}"                 # near-visually-lossless reference quality
mkdir -p "${RAW_DIR}" "${TMP}"

command -v ffmpeg >/dev/null 2>&1 || { echo "[FATAL] ffmpeg not found (needed to transcode references)."; exit 2; }

fetch() {  # fetch <url> <out>  (resumable)
    local url="$1" out="$2"
    if [ -f "${out}" ]; then echo "  cached: ${out}"; return 0; fi
    echo "  downloading: ${url}"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "${out}.part" "${url}" && mv "${out}.part" "${out}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "${out}.part" "${url}" && mv "${out}.part" "${out}"
    else
        echo "[FATAL] neither wget nor curl is available."; exit 2
    fi
}

unzip_one() {  # unzip_one <zip> <destdir>
    local zip="$1" dest="$2"
    mkdir -p "${dest}"
    "${PY}" -m zipfile -e "${zip}" "${dest}"
}

find_first() {  # find_first <dir> <ext>  -> prints first matching file
    find "$1" -type f -iname "*.$2" 2>/dev/null | head -n 1
}

echo "=============================================================="
echo "Preparing source videos -> ${RAW_DIR}/"
echo "=============================================================="

# ---------------------------------------------------------------- crowd_run
# 1080p50, 500 frames (~10s). High motion -> best chance of resolution crossovers.
if [ ! -f "${RAW_DIR}/crowd_run.mp4" ]; then
    echo "[crowd_run] (1.4 GB y4m)"
    fetch "https://media.xiph.org/video/derf/y4m/crowd_run_1080p50.y4m" "${TMP}/crowd_run.y4m"
    ffmpeg -y -hide_banner -loglevel error -i "${TMP}/crowd_run.y4m" \
        -c:v libx264 -crf "${CRF}" -preset medium -pix_fmt yuv420p -an "${RAW_DIR}/crowd_run.mp4"
    echo "  -> ${RAW_DIR}/crowd_run.mp4"
else echo "[crowd_run] exists, skip"; fi

# ---------------------------------------------------------------- sintel
# Official 1080p trailer (~52s). Matches the ~14-chunk sintel clip.
if [ ! -f "${RAW_DIR}/sintel.mp4" ]; then
    echo "[sintel] (15 MB 1080p trailer)"
    fetch "https://download.blender.org/apricot/trailer/sintel_trailer-1080p.mp4" "${TMP}/sintel_src.mp4"
    ffmpeg -y -hide_banner -loglevel error -i "${TMP}/sintel_src.mp4" \
        -c:v libx264 -crf "${CRF}" -preset medium -pix_fmt yuv420p -an "${RAW_DIR}/sintel.mp4"
    echo "  -> ${RAW_DIR}/sintel.mp4"
else echo "[sintel] exists, skip"; fi

# ---------------------------------------------------------------- bigbuckbunny
# 1080p H.264 (zip 725 MB). Trim a 192s segment (48 chunks @ 4s) from t=60s.
if [ ! -f "${RAW_DIR}/bigbuckbunny.mp4" ]; then
    echo "[bigbuckbunny] (725 MB zip)"
    fetch "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov.zip" "${TMP}/bbb.zip"
    unzip_one "${TMP}/bbb.zip" "${TMP}/bbb"
    SRC="$(find_first "${TMP}/bbb" mov)"
    [ -n "${SRC}" ] || { echo "[FATAL] no .mov inside bbb.zip"; exit 1; }
    ffmpeg -y -hide_banner -loglevel error -ss 60 -i "${SRC}" -t 192 \
        -c:v libx264 -crf "${CRF}" -preset medium -pix_fmt yuv420p -an "${RAW_DIR}/bigbuckbunny.mp4"
    echo "  -> ${RAW_DIR}/bigbuckbunny.mp4"
else echo "[bigbuckbunny] exists, skip"; fi

# ---------------------------------------------------------------- tearsofsteel_short
# 1080p (zip 583 MB). Trim a 60s segment (15 chunks) from t=120s.
if [ ! -f "${RAW_DIR}/tearsofsteel_short.mp4" ]; then
    echo "[tearsofsteel_short] (583 MB zip)"
    fetch "https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip" "${TMP}/tos.zip"
    unzip_one "${TMP}/tos.zip" "${TMP}/tos"
    SRC="$(find_first "${TMP}/tos" mov)"
    [ -n "${SRC}" ] || { echo "[FATAL] no .mov inside tos.zip"; exit 1; }
    ffmpeg -y -hide_banner -loglevel error -ss 120 -i "${SRC}" -t 60 \
        -c:v libx264 -crf "${CRF}" -preset medium -pix_fmt yuv420p -an "${RAW_DIR}/tearsofsteel_short.mp4"
    echo "  -> ${RAW_DIR}/tearsofsteel_short.mp4"
else echo "[tearsofsteel_short] exists, skip"; fi

echo ""
echo "=============================================================="
echo "Done. References in ${RAW_DIR}/:"
ls -lh "${RAW_DIR}"/*.mp4 2>/dev/null || true
if [ "${KEEP_TMP:-0}" != "1" ]; then
    echo "Removing temporary downloads in ${TMP} (set KEEP_TMP=1 to keep)."
    rm -rf "${TMP}"
fi
echo "Next:  bash src/training/run_multires_v16.sh"
echo "=============================================================="
