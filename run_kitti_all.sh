#!/usr/bin/env bash
set -euo pipefail

configs=""
max_frames="null"
total_sequences=11
sequence_index=0
no_gui=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--configs)
      configs="$2"
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
      echo "Usage: ./run_kitti_all.sh [-c|--configs file1,file2] [-m|--max-frames N|null] [--no-gui]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "[INFO] Starting KITTI sequences 00-10"
for seq in 00 01 02 03 04 05 06 07 08 09 10; do
  sequence_index=$((sequence_index + 1))
  echo "[INFO] (${sequence_index}/${total_sequences}) Sequence ${seq} start"
  args=(--sequence "$seq")
  if [[ -n "$configs" ]]; then
    args+=(--configs "$configs")
  fi
  if [[ -n "$max_frames" ]]; then
    args+=(--max-frames "$max_frames")
  fi
  if [[ "$no_gui" == "true" ]]; then
    args+=(--no-gui)
  fi
  ./run_kitti.sh "${args[@]}"
  echo "[INFO] (${sequence_index}/${total_sequences}) Sequence ${seq} done"
done
