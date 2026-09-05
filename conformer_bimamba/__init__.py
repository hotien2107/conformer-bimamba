"""Hybrid Conformer-BiMamba speech separation for Vietnamese.

Replaces Multi-Head Self-Attention in the Conformer block with a
Bidirectional Mamba (Bi-Mamba) layer, and conditions the state-space
transition (B, C matrices) on the F0 contour of the mixture.

"""

__version__ = "0.1.0"
