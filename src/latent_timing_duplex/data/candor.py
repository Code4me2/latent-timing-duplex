"""CANDOR corpus access notes and a load stub.

CANDOR (Conversation: A Naturalistic Dataset of Online Recordings) is a
multimodal corpus of unscripted dyadic video-chat conversations collected by
BetterUp Labs (Reece, Cooney, et al.; Science Advances / arXiv:2203.00674).
The public paper describes ~1,656 conversations and 850+ hours of audio/video.

This repository does **not** include CANDOR files. Access is by request through
the BetterUp data portal. After approval, a later fill-in of this module will
select a 200–500 h speaker-separated dual-channel working subset.
"""

from __future__ import annotations

from pathlib import Path

from latent_timing_duplex.exceptions import Phase0NotImplemented
from latent_timing_duplex.types import DualChannelSession

CANDOR_PAPER = "https://arxiv.org/abs/2203.00674"
CANDOR_ACCESS_URL = "https://betterup-data-requests.herokuapp.com/"
CANDOR_INFO_URL = "https://www.betterup.com/research/candor-research"

LICENSE_NOTE = """
CANDOR is human-subjects conversational data released to researchers through
BetterUp Labs. It is not redistributed by this project.

- Request access: https://betterup-data-requests.herokuapp.com/
- Overview: https://www.betterup.com/research/candor-research
- Paper: https://arxiv.org/abs/2203.00674

Do not commit raw audio, video, transcripts, or derived waveforms. Point
``local_root`` at a directory you obtained under the CANDOR data-use terms.
Speaker-separated dual-channel audio is not a published dump in this repo;
building that subset is a Phase 0 work item after access is granted.
""".strip()


class CandorPipeline:
    """Reserved loader for a local CANDOR checkout.

    Parameters
    ----------
    local_root:
        Directory you created after a successful BetterUp request. This stub
        never downloads and never invents a default cache path.
    """

    def __init__(self, local_root: Path | None = None) -> None:
        self.local_root = Path(local_root) if local_root is not None else None

    def license_note(self) -> str:
        return LICENSE_NOTE

    def expected_layout(self) -> str:
        return (
            "After BetterUp approval, expect a local tree with per-conversation "
            "audio/video and transcripts. This skeleton does not prescribe a "
            "layout or a download command. Fill in ``list_sessions`` / "
            "``load_session`` once the DUA-compliant files are on disk."
        )

    def list_sessions(self) -> list[str]:
        raise Phase0NotImplemented(
            "CANDOR listing is reserved until access is granted via "
            f"{CANDOR_ACCESS_URL}. See PHASE0.md (blocker: BetterUp request)."
        )

    def load_session(self, session_id: str) -> DualChannelSession:
        raise Phase0NotImplemented(
            f"CANDOR session {session_id!r} cannot be loaded in the Phase 0 "
            "skeleton. Request the corpus, then implement speaker-separated "
            "dual-channel conversion here. No files are bundled."
        )
