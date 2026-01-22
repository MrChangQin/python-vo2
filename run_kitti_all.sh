#!/usr/bin/env bash
set -euo pipefail

configs=""
max_frames="null"

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
    -h|--help)
      echo "Usage: ./run_kitti_all.sh [-c|--configs file1,file2] [-m|--max-frames N|null]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
  if [[ -n "$configs" && -n "$max_frames" ]]; then
    ./run_kitti.sh --sequence "$seq" --configs "$configs" --max-frames "$max_frames"
  elif [[ -n "$configs" ]]; then
    ./run_kitti.sh --sequence "$seq" --configs "$configs"
  elif [[ -n "$max_frames" ]]; then
    ./run_kitti.sh --sequence "$seq" --max-frames "$max_frames"
  else
    ./run_kitti.sh --sequence "$seq"
  fi
done
