import os

import numpy as np
import cv2
import argparse
import yaml
import logging
from tqdm import tqdm

from utils.tools import plot_keypoints

from dataloader import create_dataloader
from detectors import create_detector
from matchers import create_matcher
from vo.VisualOdometry import VisualOdometry, AbosluteScaleComputer


def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"])


class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        pass

    def update(self, est_xyz, gt_xyz):
        est_flat = np.asarray(est_xyz).reshape(-1)
        gt_flat = np.asarray(gt_xyz).reshape(-1)
        x, z = float(est_flat[0]), float(est_flat[2])
        gt_x, gt_z = float(gt_flat[0]), float(gt_flat[2])

        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)

        error = np.linalg.norm(est - gt)

        self.errors.append(error)

        avg_error = np.mean(np.array(self.errors))

        # === drawer ==================================
        # each point
        draw_x, draw_y = int(x) + 290, int(z) + 90
        true_x, true_y = int(gt_x) + 290, int(gt_z) + 90

        # draw trajectory
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(self.traj, (10, 20), (600, 80), (0, 0, 0), -1)

        # draw text
        text = "[AvgError] %2.4fm" % (avg_error)
        cv2.putText(self.traj, text, (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        return self.traj


def get_results_dir(config):
    results_dir = "output"
    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name")
    sequence = dataset_config.get("sequence")
    if sequence is not None:
        if dataset_name == "KITTILoader":
            results_dir = os.path.join(results_dir, f"kitti_sequence_{sequence}")
        else:
            results_dir = os.path.join(results_dir, f"sequence_{sequence}")
    return results_dir


def run(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    show_gui = True
    if args.no_gui:
        show_gui = False
    elif os.name != "nt" and not os.environ.get("DISPLAY"):
        logging.warning("DISPLAY is not set; disabling GUI.")
        show_gui = False

    # create dataloader
    loader = create_dataloader(config["dataset"])
    # create detector
    detector = create_detector(config["detector"])
    # create matcher
    matcher = create_matcher(config["matcher"])

    absscale = AbosluteScaleComputer()
    traj_plotter = TrajPlotter()

    # log
    fname = args.config.split('/')[-1].split('.')[0]
    results_dir = get_results_dir(config)
    os.makedirs(results_dir, exist_ok=True)
    log_fopen = open(os.path.join(results_dir, fname + ".txt"), mode='a')

    vo = VisualOdometry(detector, matcher, loader.cam)
    for i, img in enumerate(tqdm(loader, total=len(loader), desc="Frames", unit="frame")):
        gt_pose = loader.get_cur_pose()
        R, t = vo.update(img, absscale.update(gt_pose))

        # === log writer ==============================
        print(i, t[0, 0], t[1, 0], t[2, 0], gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3], file=log_fopen)

        # === drawer ==================================
        img1 = keypoints_plot(img, vo)
        img2 = traj_plotter.update(t, gt_pose[:, 3])

        if show_gui:
            cv2.imshow("keypoints", img1)
            cv2.imshow("trajectory", img2)
            if cv2.waitKey(10) == 27:
                break

    cv2.imwrite(os.path.join(results_dir, fname + ".png"), img2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')
    parser.add_argument('--config', type=str, default='config/kitti_orb_brutematch.yaml',
                        help='config file')
    parser.add_argument('--logging', type=str, default='INFO',
                        help='logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL')
    parser.add_argument('--no-gui', action='store_true',
                        help='disable OpenCV GUI (useful for headless environments)')

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])

    run(args)

