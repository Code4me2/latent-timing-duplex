"""Stubs import and refuse to pretend inference or downloads exist."""

from __future__ import annotations

import pytest

from latent_timing_duplex.baselines.vap import VAPBaseline
from latent_timing_duplex.data.candor import CANDOR_ACCESS_URL, CandorPipeline
from latent_timing_duplex.data.duplexchat import DUPLEXCHAT_MANIFEST_ID, DuplexChatPipeline
from latent_timing_duplex.exceptions import Phase0NotImplemented, WeightsNotBundled
from latent_timing_duplex.extract.nll import FrozenNLLExtractor
from latent_timing_duplex.models.bayling_duplex import BayLingDuplexWrapper
from latent_timing_duplex.models.moshi import MoshiWrapper
from latent_timing_duplex.types import DualChannelSession


def test_moshi_requires_local_dir() -> None:
    wrapper = MoshiWrapper("moshiko")
    assert wrapper.model_id == "kyutai/moshiko-pytorch-bf16"
    with pytest.raises(WeightsNotBundled, match="kyutai/moshiko-pytorch-bf16"):
        wrapper.load()


def test_moshi_missing_local_dir() -> None:
    with pytest.raises(FileNotFoundError, match="not a directory"):
        MoshiWrapper().load(local_dir="/tmp/not-a-real-moshi-dir")


def test_bayling_documents_shards_and_display_bug() -> None:
    ids = BayLingDuplexWrapper().documented_ids()
    assert ids["weights"] == "BayLing-Models/BayLing-Duplex"
    assert ids["tokenizer"] == "zai-org/glm-4-voice-tokenizer"
    assert ids["decoder"] == "zai-org/glm-4-voice-decoder"
    assert ids["n_safetensor_shards"] == 4
    assert ids["hf_param_display_bug"] == "516k"
    with pytest.raises(WeightsNotBundled, match="516k"):
        BayLingDuplexWrapper().load()


def test_vap_requires_local_checkpoint() -> None:
    with pytest.raises(WeightsNotBundled, match="ErikEkstedt/VAP"):
        VAPBaseline().load()


def test_candor_notes_and_blocker() -> None:
    pipe = CandorPipeline()
    note = pipe.license_note()
    assert "BetterUp" in note
    assert CANDOR_ACCESS_URL in note
    with pytest.raises(Phase0NotImplemented, match="BetterUp"):
        pipe.list_sessions()


def test_duplexchat_notes_and_blocker() -> None:
    pipe = DuplexChatPipeline()
    assert DUPLEXCHAT_MANIFEST_ID in pipe.license_note()
    assert "reconstruct" in pipe.reconstruct_pointer().lower()
    with pytest.raises(Phase0NotImplemented, match="reconstruct"):
        pipe.list_sessions()


def test_nll_extractor_needs_loaded_model() -> None:
    session = DualChannelSession(session_id="x", duration_s=1.0)
    extractor = FrozenNLLExtractor(MoshiWrapper())
    with pytest.raises(WeightsNotBundled, match="not loaded"):
        extractor.extract(session)
    wrapped = extractor.wrap_values(session, [1.0, 2.0, 3.0])
    assert [c.value for c in wrapped] == [1.0, 2.0, 3.0]
