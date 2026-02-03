# based on: https://github.com/syinari0123/SuperPoint-VO

import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def make_label(config_name):
    name = config_name.lower()
    if name.startswith("kitti_"):
        name = name[len("kitti_"):]

    detector_map = {
        "orb": "ORB",
        "sift": "SIFT",
        "superpoint": "SuperPoint",
        "xfeat": "SwiftFeat",
        "swiftfeat": "SwiftFeat",
    }
    matcher_map = {
        "brutematch": "BF",
        "flannmatch": "FLANN",
        "supergluematch": "SuperGlue",
        "lightgluematch": "SwiftGlue",
        # "xfeatmatch_star": "XFM*",
        "xfeatmatch": " ",
        # "swiftfeatmatch_star": "SFM*",
        "swiftfeatmatch": " ",
    }

    detector = None
    matcher = None
    for key, label in detector_map.items():
        if key in name:
            detector = label
            break
    for key, label in matcher_map.items():
        if key in name:
            matcher = label
            break

    if detector and matcher:
        return f"{detector}+{matcher}"
    return config_name

def _get_fixed_color(config_name):
    name = config_name.lower()
    color_map = {
        "kitti_xfeat_lightgluematch": "red",
        "kitti_superpoint_supergluematch": "#4DA3FF",
        "kitti_superpoint_flannmatch": "#E67E22",
    }
    for key, color in color_map.items():
        if key in name:
            return color
    return None


def read_log(log_file):
    img_ids = []
    est_points = []
    gt_points = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            tmp_data = line.split()
            img_ids.append(int(tmp_data[0]))
            est_points.append([float(x) for x in tmp_data[1:4]])
            gt_points.append([float(x) for x in tmp_data[4:7]])
    return np.array(img_ids), np.array(est_points), np.array(gt_points)

def _parse_max_frames(value):
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in ("null", "none", ""):
        return None
    return int(lowered)

def _parse_exclude(value):
    if value is None:
        return []
    items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item]

def _should_skip_log(path, exclude_tokens):
    name = path.stem
    return any(token in name for token in exclude_tokens)


def compute_errors(est_xyz, gt_xyz):
    if est_xyz.ndim != 2 or gt_xyz.ndim != 2 or est_xyz.shape[0] == 0 or gt_xyz.shape[0] == 0:
        return np.array([]), np.array([])
    errors = np.linalg.norm((est_xyz - gt_xyz), axis=1)
    avg_error = np.array([np.mean(errors[:i + 1]) for i in range(len(errors))])

    relative_errors = np.zeros((est_xyz.shape[0] - 1))
    for i in range(1, est_xyz.shape[0]):
        relative_est_xyz = est_xyz[i, :] - est_xyz[i - 1, :]
        relative_gt_xyz = gt_xyz[i, :] - gt_xyz[i - 1, :]
        relative_errors[i - 1] = np.linalg.norm((relative_est_xyz - relative_gt_xyz))

    return avg_error, relative_errors


def plot_sequence(sequence_dir, seq_name, results):
    figure = plt.figure()
    for config_name, data in results.items():
        label = make_label(config_name)
        ids = data["ids"]
        avg_error = data["avg_error"]
        relative_errors = data["relative_errors"]
        color = _get_fixed_color(config_name)

        plt.subplot(2, 1, 1)
        plt.plot(ids, avg_error, label=label, color=color)

        plt.subplot(2, 1, 2)
        plt.plot(ids[1:], relative_errors, label=label, color=color)

    plt.subplot(2, 1, 1)
    plt.xlabel("FrameIndex")
    plt.ylabel("Avg Distance Error [m]")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel("FrameIndex")
    plt.ylabel("Relative Distance Error [m]")
    plt.legend()

    plot_path = sequence_dir / f"{seq_name}_eval.png"
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VO results.")
    parser.add_argument("--max-frames", default=1500,
                        help="Evaluate only the first N frames. Use 'null' for all.")
    parser.add_argument("--exclude", default="kitti_xfeat_xfeatmatch" \
                                             ",kitti_swiftfeat_swiftfeatmatch" ,
                                            #  ",kitti_superpoint_flannmatch",
                                             
                        help="Comma-separated config name substrings to skip.")
    args = parser.parse_args()
    max_frames = _parse_max_frames(args.max_frames)
    exclude_tokens = _parse_exclude(args.exclude)

    sequence_files = glob.glob("output/*/*.txt")
    if not sequence_files:
        print("No log files found under output/*/.")
        return

    sequences = {}
    for file_path in sequence_files:
        path = Path(file_path)
        if path.name == "relative_errors.txt":
            continue
        if _should_skip_log(path, exclude_tokens):
            continue
        sequences.setdefault(path.parent, []).append(path)

    for seq_dir, logs in sorted(sequences.items()):
        seq_name = seq_dir.name
        results = {}

        summary_path = seq_dir / "relative_errors.txt"
        with open(summary_path, mode="w") as log_fopen:
            print("config,mean_abs_error,mean_relative_error", file=log_fopen)
            for log_path in sorted(logs):
                config_name = log_path.stem
                ids, est_xyz, gt_xyz = read_log(str(log_path))
                if max_frames is not None:
                    ids = ids[:max_frames]
                    est_xyz = est_xyz[:max_frames]
                    gt_xyz = gt_xyz[:max_frames]
                avg_error, relative_errors = compute_errors(est_xyz, gt_xyz)

                results[config_name] = {
                    "ids": ids,
                    "avg_error": avg_error,
                    "relative_errors": relative_errors
                }

                mean_abs_error = float(avg_error[-1]) if len(avg_error) else 0.0
                mean_relative_error = float(relative_errors.mean()) if len(relative_errors) else 0.0
                print(f"{config_name},{mean_abs_error:.6f},{mean_relative_error:.6f}", file=log_fopen)

        plot_sequence(seq_dir, seq_name, results)


if __name__ == "__main__":
    main()
