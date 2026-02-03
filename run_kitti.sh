#!/usr/bin/env bash
set -euo pipefail

sequence="00"
max_frames="null"
configs=()
total_runs=0
run_index=0
no_gui=false
jobs=1
repo_root="$(cd "$(dirname "$0")" && pwd)"

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
    -j|--jobs)
      jobs="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: ./run_kitti.sh [-s|--sequence 00] [-c|--configs file1,file2] [-m|--max-frames N|null] [--no-gui] [-j|--jobs N]"
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
echo "[INFO] Sequence: ${sequence}, max_frames: ${max_frames}, configs: ${total_runs}, jobs: ${jobs}"

failures=0

run_one() {
  local cfg="$1"
  local run_idx="$2"

  echo "[INFO] (${run_idx}/${total_runs}) Preparing ${cfg}"
  if [[ ! -f "$cfg" ]]; then
    echo "Config not found: $cfg" >&2
    return 1
  fi

  local tmp_name tmp_path run_name out_dir txt_path
  tmp_name="$(basename "${cfg%.yaml}")_seq_${sequence}.yaml"
  tmp_path="/tmp/${tmp_name}"
  if [[ -n "$max_frames" ]]; then
    sed -E "s/^([[:space:]]*sequence:[[:space:]]*).+$/\\1'${sequence}'/" "$cfg" \
      | sed -E "s/^([[:space:]]*max_frames:[[:space:]]*).+$/\\1${max_frames}/" \
      > "$tmp_path"
  else
    sed -E "s/^([[:space:]]*sequence:[[:space:]]*).+$/\\1'${sequence}'/" "$cfg" > "$tmp_path"
  fi

  run_name="$(basename "${tmp_path%.yaml}")"
  out_dir="output/kitti_sequence_${sequence}"
  txt_path="${out_dir}/${run_name}.txt"

  local root_path start_idx max_frames_cfg expected_frames seq_dir total_frames limit last_idx
  root_path="$(sed -n "s/^[[:space:]]*root_path:[[:space:]]*//p" "$tmp_path" | head -n1)"
  start_idx="$(sed -n "s/^[[:space:]]*start:[[:space:]]*//p" "$tmp_path" | head -n1)"
  max_frames_cfg="$(sed -n "s/^[[:space:]]*max_frames:[[:space:]]*//p" "$tmp_path" | head -n1)"
  root_path="${root_path%\"}"
  root_path="${root_path#\"}"
  root_path="${root_path%\'}"
  root_path="${root_path#\'}"
  start_idx="${start_idx:-0}"
  max_frames_cfg="${max_frames_cfg:-null}"
  if [[ "$max_frames_cfg" == "null" ]]; then
    max_frames_cfg=""
  fi
  if [[ -n "$root_path" && "$root_path" != /* ]]; then
    root_path="${repo_root}/${root_path}"
  fi

  expected_frames=""
  seq_dir="${root_path}/sequences/${sequence}/image_0"
  if [[ -d "$seq_dir" ]]; then
    total_frames=$(find "$seq_dir" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
    if [[ -n "$max_frames_cfg" && "$max_frames_cfg" -gt 0 ]]; then
      limit=$((start_idx + max_frames_cfg))
      if [[ "$limit" -gt "$total_frames" ]]; then
        limit="$total_frames"
      fi
      expected_frames=$((limit - start_idx))
    else
      expected_frames=$((total_frames - start_idx))
    fi
  fi

  if [[ -n "$expected_frames" && -f "$txt_path" && "$expected_frames" -gt 0 ]]; then
    last_idx="$(tail -n 1 "$txt_path" | awk '{print $1}')"
    if [[ "$last_idx" == "$((expected_frames - 1))" ]]; then
      echo "[INFO] (${run_idx}/${total_runs}) Skip ${cfg} (existing log: ${txt_path})"
      return 0
    fi
  fi

  echo "[INFO] (${run_idx}/${total_runs}) Running ${cfg} (sequence ${sequence})"
  if [[ "$no_gui" == "true" ]]; then
    python main.py --config "$tmp_path" --no-gui
  else
    python main.py --config "$tmp_path"
  fi
  echo "[INFO] (${run_idx}/${total_runs}) Done ${cfg}"
}

for cfg in "${configs[@]}"; do
  run_index=$((run_index + 1))
  run_one "$cfg" "$run_index" &
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$jobs" ]]; do
    wait -n || failures=$((failures + 1))
  done
done

while [[ "$(jobs -rp | wc -l | tr -d ' ')" -gt 0 ]]; do
  wait -n || failures=$((failures + 1))
done

if [[ "$failures" -gt 0 ]]; then
  echo "[ERROR] ${failures} job(s) failed." >&2
  exit 1
fi
