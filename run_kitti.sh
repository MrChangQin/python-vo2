#!/usr/bin/env bash
set -euo pipefail

sequence="00"
max_frames="null"
configs=()
total_runs=0
run_index=0
no_gui=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--sequence)
      sequence="$2"
      shift 2
      ;;
    -c|--configs)
      IFS=',' read -r -a configs <<< "$2"
      shift 2
      ;;
    -m|--max-frames)
      max_frames="$2"
      shift 2
      ;;
    --no-gui)
      no_gui=true
      shift
      ;;
    -h|--help)
      echo "Usage: ./run_kitti.sh [-s|--sequence 00] [-c|--configs file1,file2] [-m|--max-frames N|null] [--no-gui]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "[INFO] No configs provided, using default list."
  configs=(
    'config/kitti_orb_brutematch.yaml'
    # 'config/kitti_sift_flannmatch.yaml'
    'config/kitti_superpoint_flannmatch.yaml'
    'config/kitti_superpoint_supergluematch.yaml'
    'config/kitti_xfeat_xfeatmatch.yaml'
    'config/kitti_swiftfeat_swiftfeatmatch.yaml'
    'config/kitti_xfeat_lightgluematch.yaml'
  )
fi

total_runs="${#configs[@]}"
echo "[INFO] Sequence: ${sequence}, max_frames: ${max_frames}, configs: ${total_runs}"

for cfg in "${configs[@]}"; do
  run_index=$((run_index + 1))
  echo "[INFO] (${run_index}/${total_runs}) Preparing ${cfg}"
  if [[ ! -f "$cfg" ]]; then
    echo "Config not found: $cfg" >&2
    continue
  fi

  tmp_name="$(basename "${cfg%.yaml}")_seq_${sequence}.yaml"
  tmp_path="/tmp/${tmp_name}"
  if [[ -n "$max_frames" ]]; then
    sed -E "s/^([[:space:]]*sequence:[[:space:]]*).+$/\\1'${sequence}'/" "$cfg" \
      | sed -E "s/^([[:space:]]*max_frames:[[:space:]]*).+$/\\1${max_frames}/" \
      > "$tmp_path"
  else
    sed -E "s/^([[:space:]]*sequence:[[:space:]]*).+$/\\1'${sequence}'/" "$cfg" > "$tmp_path"
  fi

  echo "[INFO] (${run_index}/${total_runs}) Running ${cfg} (sequence ${sequence})"
  if [[ "$no_gui" == "true" ]]; then
    python main.py --config "$tmp_path" --no-gui
  else
    python main.py --config "$tmp_path"
  fi
  echo "[INFO] (${run_index}/${total_runs}) Done ${cfg}"
done
