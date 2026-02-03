# based on: https://github.com/syinari0123/SuperPoint-VO

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _parse_exclude(value):
    if value is None:
        return []
    items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item]

def _should_skip_log(path, exclude_tokens):
    name = path.stem
    return any(token in name for token in exclude_tokens)

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
        "xfeatmatch": "XFeatMatch",
        # "swiftfeatmatch_star": "SFM*",
        "swiftfeatmatch": "SwiftFeatMatch",
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
    est_points = []
    gt_points = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            tmp_data = line.split()
            if len(tmp_data) < 7:
                continue
            est_points.append([float(x) for x in tmp_data[1:4]])
            gt_points.append([float(x) for x in tmp_data[4:7]])
    if not est_points or not gt_points:
        return None, None
    return np.array(est_points), np.array(gt_points)


def plot_sequence(sequence_dir, seq_name, logs, mode, segment_start, segment_end):
    figure = plt.figure()

    gt_plotted = False
    for log_path in sorted(logs):
        config_name = log_path.stem
        label = make_label(config_name)
        color = _get_fixed_color(config_name)
        est_xyz, gt_xyz = read_log(str(log_path))
        if est_xyz is None or gt_xyz is None:
            continue

        if mode == "segment":
            seg_start = max(0, segment_start)
            seg_end = min(len(est_xyz), segment_end) if segment_end > 0 else len(est_xyz)
            if seg_end <= seg_start:
                continue

            if not gt_plotted:
                plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], color="black", linewidth=1.0, alpha=0.25)
                plt.plot(gt_xyz[seg_start:seg_end, 0],
                         gt_xyz[seg_start:seg_end, 2],
                         color="black", linewidth=2.0, label="ground_truth")
                gt_plotted = True

            plt.plot(est_xyz[seg_start:seg_end, 0],
                     est_xyz[seg_start:seg_end, 2],
                     linewidth=1.8, label=label, color=color)
        else:
            if not gt_plotted:
                plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], color="black", linewidth=2, label="ground_truth")
                gt_plotted = True
            plt.plot(est_xyz[:, 0], est_xyz[:, 2], linewidth=1.5, label=label, color=color)

    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.legend()

    plot_path = sequence_dir / f"{seq_name}_traj_compare.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory comparisons.")
    parser.add_argument("--mode", choices=["overlay", "segment"], default="overlay",
                        help="plot mode: overlay or segment")
    parser.add_argument("--segment-start", type=int, default=0,
                        help="segment start frame index (segment mode only)")
    parser.add_argument("--segment-end", type=int, default=1000,
                        help="segment end frame index (segment mode only; <=0 means full length)")
    parser.add_argument("--exclude", default="kitti_xfeat_xfeatmatch" \
                                             ",kitti_swiftfeat_swiftfeatmatch",
                                            #  ",kitti_superpoint_flannmatch",
                        help="Comma-separated config name substrings to skip.")
    args = parser.parse_args()
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
        plot_sequence(seq_dir, seq_name, logs, args.mode, args.segment_start, args.segment_end)


if __name__ == "__main__":
    main()
