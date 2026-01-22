import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _resolve_xfeat_root():
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "third_party/xfeat",
        repo_root / "detectors/xfeat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_weights_path(weights_value, xfeat_root):
    weights_path = Path(weights_value)
    if weights_path.is_absolute():
        return weights_path
    repo_root = Path(__file__).resolve().parent.parent
    repo_candidate = repo_root / weights_path
    if repo_candidate.exists():
        return repo_candidate
    return xfeat_root / "weights/xfeat.pt"


class XFeatDetector(object):
    default_config = {
        "top_k": 4096,
        "detection_threshold": 0.05,
        "weights": "detectors/xfeat/weights/xfeat.pt",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("XFeat detector config: ")
        logging.info(self.config)

        xfeat_root = _resolve_xfeat_root()
        if str(xfeat_root) not in sys.path:
            sys.path.insert(0, str(xfeat_root))

        from modules.xfeat import XFeat

        logging.info("creating XFeat detector...")
        weights_path = _resolve_weights_path(self.config["weights"], xfeat_root)
        self.xfeat = XFeat(weights=str(weights_path),
                           top_k=self.config["top_k"],
                           detection_threshold=self.config["detection_threshold"])

        if not self.config["cuda"] and torch.cuda.is_available():
            self.xfeat.dev = torch.device("cpu")
            self.xfeat.net.to(self.xfeat.dev)

    def __call__(self, image):
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logging.debug("detecting keypoints with XFeat...")
        output = self.xfeat.detectAndCompute(image,
                                             top_k=self.config["top_k"],
                                             detection_threshold=self.config["detection_threshold"])[0]

        ret_dict = {
            "image": image,
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "keypoints": output["keypoints"].cpu().detach().numpy(),
            "scores": output["scores"].cpu().detach().numpy(),
            "descriptors": output["descriptors"].cpu().detach().numpy()
        }

        return ret_dict
