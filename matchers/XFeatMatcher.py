import logging
import sys
from pathlib import Path

import numpy as np


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


class XFeatMatcher(object):
    default_config = {
        "top_k": 4096,
        "min_cossim": -1,
        "weights": "detectors/xfeat/weights/xfeat.pt",
        "detection_threshold": 0.05
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("XFeat matcher config: ")
        logging.info(self.config)

        xfeat_root = _resolve_xfeat_root()
        if str(xfeat_root) not in sys.path:
            sys.path.insert(0, str(xfeat_root))

        from modules.xfeat import XFeat

        logging.info("creating XFeat matcher...")
        weights_path = _resolve_weights_path(self.config["weights"], xfeat_root)
        self.xfeat = XFeat(weights=str(weights_path),
                           top_k=self.config["top_k"],
                           detection_threshold=self.config["detection_threshold"])

    def __call__(self, kptdescs):
        img0 = kptdescs["ref"]["image"]
        img1 = kptdescs["cur"]["image"]

        logging.debug("matching keypoints with XFeat match_xfeat...")
        mkpts0, mkpts1 = self.xfeat.match_xfeat(img0, img1,
                                                top_k=self.config["top_k"],
                                                min_cossim=self.config["min_cossim"])

        match_score = np.ones((mkpts0.shape[0],), dtype=np.float32)
        ret_dict = {
            "ref_keypoints": mkpts0,
            "cur_keypoints": mkpts1,
            "match_score": match_score
        }
        return ret_dict
