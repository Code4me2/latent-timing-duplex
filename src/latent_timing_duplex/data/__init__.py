"""Data pipelines for speaker-separated dual-channel dialogue.

This package contains **code and license notes only**. No CANDOR audio, no
DuplexChat episodes, and no reconstructed shards are shipped.
"""

from latent_timing_duplex.data.candor import CANDOR_ACCESS_URL, CandorPipeline
from latent_timing_duplex.data.duplexchat import DUPLEXCHAT_MANIFEST_ID, DuplexChatPipeline
from latent_timing_duplex.data.synthetic import generate_synthetic_session

__all__ = [
    "CANDOR_ACCESS_URL",
    "DUPLEXCHAT_MANIFEST_ID",
    "CandorPipeline",
    "DuplexChatPipeline",
    "generate_synthetic_session",
]
