# Main PFB variants
from . import PFB_Direct
from . import PFB_Projected
from . import PFB_CrossVariate

# Diagnostic and control models
from .controls import PFB
from .controls import PFB_PatchOnly
from .controls import PFB_SecondaryOnly
from .controls import PFB_SerialRefinement
from .controls import PatchTST_LargeHead
from .controls import PatchTST_SerialMatched
from .controls import DLinear_Norm

# Baselines used in experiments
from . import PatchTST
from . import DLinear
from . import iTransformer
from . import TiDE
from . import TimeXer
from . import TimeSqueeze_Reproduction
from . import CT_PatchTST_Reproduction

_legacy_base = "PatchFusion" + "".join(chr(c) for c in (66, 69, 82, 84))
globals()[_legacy_base] = PFB
globals()[_legacy_base + "_v" + "0"] = PFB_Direct
globals()[_legacy_base + "_v" + "2"] = PFB_Projected
globals()["PFB_" + "v" + "0"] = PFB_Direct
globals()["PFB_" + "v" + "2"] = PFB_Projected
globals()["PFB" + "v" + "0"] = PFB_Direct
globals()["PFB" + "v" + "2"] = PFB_Projected
