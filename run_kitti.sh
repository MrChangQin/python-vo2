#!/usr/bin/env bash
set -euo pipefail

sequence="00"
max_frames="null"
configs=()

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
    -h|--help)
      echo "Usage: ./run_kitti.sh [-s|--sequence 00] [-c|--configs file1,file2] [-m|--max-frames N|null]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ${#configs[@]} -eq 0 ]]; then
  configs=(
    'config/kitti_orb_brutematch.yaml',
    # 'config\kitti_sift_flannmatch.yaml',
    'config/kitti_superpoint_flannmatch.yaml',
    'config/kitti_superpoint_supergluematch.yaml',
    'config/kitti_xfeat_xfeatmatch.yaml',
    'config/kitti_swiftfeat_swiftfeatmatch.yaml',
    'config/kitti_xfeat_lightgluematch.yaml'
  )
fi

for cfg in "${configs[@]}"; do
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

  echo "Running $cfg (sequence $sequence)"
  python main.py --config "$tmp_path"
done
