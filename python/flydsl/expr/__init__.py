# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# isort: skip_file
from .numeric import *
from .typing import *
from .enum import *
from .primitive import *
from .gpu import *
from .derived import *
from .struct import *
from .arith import *
from .math import *
from .trace_mod import (
    MASK_MOD_PARAMS as MASK_MOD_PARAMS,
    SCORE_MOD_PARAMS as SCORE_MOD_PARAMS,
    TracedMaskMod as TracedMaskMod,
    TracedScoreMod as TracedScoreMod,
    trace_mask_mod as trace_mask_mod,
    trace_score_mod as trace_score_mod,
    where as where,
)

from . import utils as utils
from . import arith as arith
from . import gpu as gpu
from . import math as math

_BACKEND_MODULES = {
    "rocdl": ".rocdl",
    "tdm_ops": ".rocdl.tdm_ops",  # deprecated, use .rocdl.tdm_ops instead
}


# lazy load backend subpackages
def __getattr__(name: str):
    module_name = _BACKEND_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(module_name, __name__)
    globals()[name] = module
    return module
