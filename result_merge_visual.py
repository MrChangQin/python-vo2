# Combine eval and trajectory images side-by-side for each sequence.

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def load_image(path):
    return plt.imread(str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Combine eval and trajectory plots into a single image per sequence."
    )
    parser.add_argument("--eval-suffix", default="_eval.png",
                        help="Eval image filename suffix.")
    parser.add_argument("--traj-suffix", default="_traj_compare.png",
                        help="Trajectory image filename suffix.")
    parser.add_argument("--out-suffix", default="_eval_traj.png",
                        help="Output image filename suffix.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output image DPI.")
    args = parser.parse_args()

    output_root = Path("output")
    if not output_root.exists():
        print("No output directory found.")
        return

    seq_dirs = sorted([p for p in output_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        print("No sequence directories found under output/.")
        return

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        eval_path = seq_dir / f"{seq_name}{args.eval_suffix}"
        traj_path = seq_dir / f"{seq_name}{args.traj_suffix}"
        if not eval_path.exists() or not traj_path.exists():
            print(f"Skip {seq_name}: missing eval or traj image.")
            continue

        eval_img = load_image(eval_path)
        traj_img = load_image(traj_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.98, bottom=0.02)
        axes[0].imshow(eval_img)
        axes[0].axis("off")
        axes[0].set_title("Average & Relative Position Error")

        axes[1].imshow(traj_img)
        axes[1].axis("off")
        axes[1].set_title("Trajectory Comparison in X–Z Plane")

        out_path = seq_dir / f"{seq_name}{args.out_suffix}"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
