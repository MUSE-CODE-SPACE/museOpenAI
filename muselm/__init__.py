"""MuseLM: a fully open, from-scratch language model stack.

Everything needed to build a modern LLM — byte-level BPE tokenizer,
Mixture-of-Experts transformer, training loop, and inference engine —
implemented in plain PyTorch with no hidden dependencies.
"""

from muselm.config import MuseLMConfig, TrainConfig
from muselm.model import MuseLM
from muselm.tokenizer import ByteBPETokenizer

__version__ = "0.1.0"

__all__ = [
    "MuseLM",
    "MuseLMConfig",
    "TrainConfig",
    "ByteBPETokenizer",
    "__version__",
]
