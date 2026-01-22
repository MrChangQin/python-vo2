#!/usr/bin/env bash
set -euo pipefail

configs=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--configs)
      configs="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: ./run_kitti_all.sh [-c|--configs file1,file2]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
  if [[ -n "$configs" ]]; then
    ./run_kitti.sh --sequence "$seq" --configs "$configs"
  else
    ./run_kitti.sh --sequence "$seq"
  fi
done
