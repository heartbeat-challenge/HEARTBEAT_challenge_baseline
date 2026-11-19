#!/usr/bin/env bash
# Download the shared Realtime-Venus Audio checkpoint for Track I.
set -euo pipefail

local_dir=${1:-models}
revision=a1d95d3ac61416c8db34736b152241396850a6ed
hf download inclusionAI/Realtime-Venus download_models.py \
  --revision "$revision" --local-dir "$local_dir"
python "$local_dir/download_models.py" --model audio --local-dir "$local_dir" \
  --revision "$revision"
test -s "$local_dir/Realtime-Venus-Audio/config.json"
test -s "$local_dir/Realtime-Venus-Audio/assets/HT_ref_audio.wav"
