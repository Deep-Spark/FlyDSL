"""FlyIXDL dialect extension for Iluvatar ivcore11 GPU programming.

This module provides CuTe-style atom types for Iluvatar hardware:
- MMA atom: MMAD (16x16x16 TCU instruction)
- Copy atoms: SMECopy16x (G2S), SLBLoad (S2R), DescStore (R2G)
"""

from .universal import MMAD, SMECopy16x, SLBLoad, DescStore, make_smem_tile  # noqa: F401
