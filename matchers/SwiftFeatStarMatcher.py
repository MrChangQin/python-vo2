import logging
import sys
from pathlib import Path

import numpy as np


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


class SwiftFeatStarMatcher(object):
    default_config = {
        "top_k": 4096,
        "weights": "third_party/swiftfeat/weights/xfeat_default_169500_swiftfeatv5.pt",
        "detection_threshold": 0.05
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("SwiftFeat star matcher config: ")
        logging.info(self.config)

        swiftfeat_root = _resolve_swiftfeat_root()
        if str(swiftfeat_root) not in sys.path:
            sys.path.insert(0, str(swiftfeat_root))

        from modules.xfeat import XFeat

        logging.info("creating SwiftFeat star matcher...")
        weights_path = _resolve_weights_path(self.config["weights"], swiftfeat_root)
        self.xfeat = XFeat(weights=str(weights_path),
                           top_k=self.config["top_k"],
                           detection_threshold=self.config["detection_threshold"])

    def __call__(self, kptdescs):
        img0 = kptdescs["ref"]["image"]
        img1 = kptdescs["cur"]["image"]

        logging.debug("matching keypoints with SwiftFeat match_xfeat_star...")
        mkpts0, mkpts1 = self.xfeat.match_xfeat_star(img0, img1,
                                                     top_k=self.config["top_k"])

        match_score = np.ones((mkpts0.shape[0],), dtype=np.float32)
        ret_dict = {
            "ref_keypoints": mkpts0,
            "cur_keypoints": mkpts1,
            "match_score": match_score
        }
        return ret_dict
