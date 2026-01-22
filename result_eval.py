# based on: https://github.com/syinari0123/SuperPoint-VO

import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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


def compute_errors(est_xyz, gt_xyz):
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
        ids = data["ids"]
        avg_error = data["avg_error"]
        relative_errors = data["relative_errors"]

        plt.subplot(2, 1, 1)
        plt.plot(ids, avg_error, label=config_name)

        plt.subplot(2, 1, 2)
        plt.plot(ids[1:], relative_errors, label=config_name)

    plt.subplot(2, 1, 1)
    plt.xlabel("FrameIndex")
    plt.ylabel("Avg Distance Error [m]")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel("FrameIndex")
    plt.ylabel("Relative Distance Error [m]")
    plt.legend()

    plot_path = sequence_dir / f"{seq_name}_eval.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    sequence_files = glob.glob("output/*/*.txt")
    if not sequence_files:
        print("No log files found under output/*/.")
        return

    sequences = {}
    for file_path in sequence_files:
        path = Path(file_path)
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
