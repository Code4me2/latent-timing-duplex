"""DuplexChat reconstruct-from-podcasts notes and a load stub.

DuplexChat (arXiv:2607.04941) is a large two-speaker full-duplex spoken
dialogue corpus built from public podcast feeds. Hugging Face
``sarulab-speech/DuplexChat`` ships **manifests only** (pointers + timings);
it does not redistribute audio. Reconstruction is local: download each
episode, slice the dialogue span, and re-run the official separation pipeline.

Official reconstruct code: https://github.com/sarulab-speech/DuplexChat
(``scripts/reconstruct_dataset.py``). This repo does not wrap or invoke that
script, and it does not download podcasts.
"""

from __future__ import annotations

from pathlib import Path

from latent_timing_duplex.exceptions import Phase0NotImplemented
from latent_timing_duplex.types import DualChannelSession

DUPLEXCHAT_PAPER = "https://arxiv.org/abs/2607.04941"
DUPLEXCHAT_MANIFEST_ID = "sarulab-speech/DuplexChat"
DUPLEXCHAT_RECONSTRUCT_REPO = "https://github.com/sarulab-speech/DuplexChat"

LICENSE_NOTE = """
The DuplexChat Hugging Face dataset (sarulab-speech/DuplexChat) is
reconstruction metadata. Manifests are released so each user can rebuild
clips locally; the audio itself is not in that repo.

- Manifest: https://huggingface.co/datasets/sarulab-speech/DuplexChat
- Reconstruct: https://github.com/sarulab-speech/DuplexChat
- Paper: https://arxiv.org/abs/2607.04941

Audio and RSS content remain with the original podcast rightsholders. Do not
commit reconstructed shards, episode MP3s, or WebDataset tarballs. A later
fill-in of this module will read a *local* reconstruct output directory that
you created yourself (small English subset first, on the order of tens of
hours, then grow toward the 200–500 h working set).
""".strip()


class DuplexChatPipeline:
    """Reserved reader for a locally reconstructed DuplexChat subset.

    Parameters
    ----------
    local_root:
        Directory produced by the official reconstruct script (or an equivalent
        local pipeline). This stub does not call Hugging Face or podcast URLs.
    """

    def __init__(self, local_root: Path | None = None) -> None:
        self.local_root = Path(local_root) if local_root is not None else None

    def license_note(self) -> str:
        return LICENSE_NOTE

    def reconstruct_pointer(self) -> str:
        return (
            f"Use the upstream reconstruct script in {DUPLEXCHAT_RECONSTRUCT_REPO} "
            f"with manifests from {DUPLEXCHAT_MANIFEST_ID}. This package does not "
            "download episodes or run DialogueSidon."
        )

    def list_sessions(self) -> list[str]:
        raise Phase0NotImplemented(
            "DuplexChat listing is reserved until you reconstruct a local subset "
            f"from {DUPLEXCHAT_MANIFEST_ID}. See PHASE0.md (blocker: "
            "reconstruct-from-podcasts)."
        )

    def load_session(self, session_id: str) -> DualChannelSession:
        raise Phase0NotImplemented(
            f"DuplexChat session {session_id!r} cannot be loaded in the Phase 0 "
            "skeleton. Reconstruct locally, then point this pipeline at that "
            "directory. No audio is bundled."
        )
