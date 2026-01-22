import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _resolve_swiftfeat_root():
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "third_party/swiftfeat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_weights_path(weights_value, swiftfeat_root):
    weights_path = Path(weights_value)
    if weights_path.is_absolute():
        return weights_path
    repo_root = Path(__file__).resolve().parent.parent
    repo_candidate = repo_root / weights_path
    if repo_candidate.exists():
        return repo_candidate
    return swiftfeat_root / "weights/xfeat_default_169500_swiftfeatv5.pt"


class SwiftFeatDetector(object):
    default_config = {
        "top_k": 4096,
        "detection_threshold": 0.05,
        "weights": "third_party/swiftfeat/weights/xfeat_default_169500_swiftfeatv5.pt",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("SwiftFeat detector config: ")
        logging.info(self.config)

        swiftfeat_root = _resolve_swiftfeat_root()
        if str(swiftfeat_root) not in sys.path:
            sys.path.insert(0, str(swiftfeat_root))

        from modules.xfeat import XFeat

        logging.info("creating SwiftFeat detector...")
        weights_path = _resolve_weights_path(self.config["weights"], swiftfeat_root)
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

        logging.debug("detecting keypoints with SwiftFeat...")
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
