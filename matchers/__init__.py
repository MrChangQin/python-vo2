from .FrameByFrameMatcher import FrameByFrameMatcher
from .SuperGlueMatcher import SuperGlueMatcher
from .XFeatMatcher import XFeatMatcher
from .XFeatStarMatcher import XFeatStarMatcher
from .XFeatLightGlueMatcher import XFeatLightGlueMatcher
from .SwiftFeatLightGlueMatcher import SwiftFeatLightGlueMatcher
from .SwiftFeatMatcher import SwiftFeatMatcher
from .SwiftFeatStarMatcher import SwiftFeatStarMatcher


def create_matcher(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        matcher = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return matcher
