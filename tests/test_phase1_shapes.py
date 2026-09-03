"""Head shapes, param budget, dataset packing, frozen extractors."""

from __future__ import annotations

import numpy as np
import pytest

from latent_timing_duplex.exceptions import Phase2OutOfScope, WeightsNotBundled
from latent_timing_duplex.phase1.dataset import ChunkedStereoDataset, make_synthetic_stereo_session
from latent_timing_duplex.phase1.heads import (
    DEFAULT_EMBED_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_WIDTH,
    MAX_PARAMS,
    MLPPredictor,
    TinyTransformerPredictor,
    count_mlp_parameters,
    count_parameters,
)
from latent_timing_duplex.phase1.hidden import (
    FakeHiddenStateExtractor,
    MoshiHiddenStateExtractor,
    refuse_unfreeze,
)
from latent_timing_duplex.phase1.precompute import write_hidden_npz, write_target_npz
from latent_timing_duplex.phase1.targets import FakeTargetEmbedder, FrozenUserChunkEmbedder
from latent_timing_duplex.phase1.windows import crop_session, window_bounds


def test_default_mlp_under_budget() -> None:
    n = count_mlp_parameters(DEFAULT_HIDDEN_DIM, DEFAULT_EMBED_DIM, DEFAULT_WIDTH, 2)
    assert n == 2_491_648
    assert n < MAX_PARAMS
    # Do not construct the 4096-wide head in unit tests (init is cheap, but
    # keep CI tiny). The formula is what the Spark config relies on.


def test_tiny_mlp_forward_shape_and_param_count() -> None:
    head = MLPPredictor(hidden_dim=16, embed_dim=8, width=12, n_layers=2, seed=0)
    out = head.forward(np.zeros((5, 16)))
    assert out.shape == (5, 8)
    assert count_parameters(head) == count_mlp_parameters(16, 8, 12, 2)
    # 16→12 + 12→12 + 12→8
    assert count_parameters(head) == (16 * 12 + 12) + (12 * 12 + 12) + (12 * 8 + 8)


def test_mlp_rejects_over_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        MLPPredictor(hidden_dim=8192, embed_dim=4096, width=4096, n_layers=2)


def test_tiny_transformer_forward_shapes() -> None:
    head = TinyTransformerPredictor(
        hidden_dim=16, embed_dim=8, d_model=16, n_layers=2, n_heads=4, seed=0
    )
    out = head.forward(np.ones((3, 16)))
    assert out.shape == (3, 8)
    out_seq = head.forward(np.ones((3, 5, 16)))
    assert out_seq.shape == (3, 8)
    assert count_parameters(head) > 0
    assert count_parameters(head) < MAX_PARAMS


def test_chunked_stereo_left_is_user() -> None:
    session = make_synthetic_stereo_session(duration_s=0.80, sample_rate=16000, seed=3)
    data = ChunkedStereoDataset(session, chunk_duration_s=0.08)
    assert len(data) == 10
    chunk = data[0]
    assert chunk.left.shape == (int(0.08 * 16000),)
    assert chunk.user is chunk.left
    stereo = data.stereo_matrix()
    assert stereo.shape[0] == 2
    np.testing.assert_array_equal(stereo[0, 0], chunk.left)
    # LEFT and RIGHT are distinct (channel convention check).
    assert not np.allclose(data[0].left, data[0].right)


def test_dataset_horizon_pairs() -> None:
    session = make_synthetic_stereo_session(duration_s=1.60, sample_rate=8000, seed=0)
    data = ChunkedStereoDataset(session)
    pairs = data.pairs_for_horizon(0.08)
    assert pairs[0].target_index == pairs[0].source_index + 1
    assert pairs[0].source.t_end == pytest.approx(0.08)
    assert pairs[0].target.t_end == pytest.approx(0.16)
    pairs_1s = data.pairs_for_horizon(1.00)
    assert pairs_1s[0].target_index == 13
    assert pairs_1s[0].target.t_end == pytest.approx(1.12)  # 14 * 0.08


def test_fake_hidden_and_target_shapes() -> None:
    session = make_synthetic_stereo_session(duration_s=0.40, sample_rate=8000, seed=1)
    hidden = FakeHiddenStateExtractor(hidden_dim=16).extract(session)
    assert hidden.shape == (5, 16)
    data = ChunkedStereoDataset(session)
    z = FakeTargetEmbedder(embed_dim=8).embed_dataset(data)
    assert z.shape == (5, 8)


def test_moshi_extractor_needs_weights(tmp_path) -> None:
    ext = MoshiHiddenStateExtractor(hidden_dim=4)
    session = make_synthetic_stereo_session(duration_s=0.24, sample_rate=8000)
    with pytest.raises(WeightsNotBundled, match="locally loaded"):
        ext.extract(session)
    path = write_hidden_npz(tmp_path / "h.npz", np.ones((3, 4)), session_id="s")
    loaded = ext.load_precomputed(path)
    assert loaded.shape == (3, 4)
    with pytest.raises(Phase2OutOfScope, match="Phase 2"):
        refuse_unfreeze()


def test_frozen_target_needs_weights(tmp_path) -> None:
    enc = FrozenUserChunkEmbedder(embed_dim=4)
    session = make_synthetic_stereo_session(duration_s=0.24, sample_rate=8000)
    data = ChunkedStereoDataset(session)
    with pytest.raises(WeightsNotBundled):
        enc.embed_chunks([data[0]])
    path = write_target_npz(tmp_path / "z.npz", np.ones((3, 4)), session_id="s")
    assert enc.load_precomputed(path).shape == (3, 4)


def test_equal_length_windows() -> None:
    start, end = window_bounds(10.0, window_s=4.0, mode="first")
    assert (start, end) == (0.0, 4.0)
    start, end = window_bounds(10.0, window_s=4.0, mode="mid")
    assert start == pytest.approx(3.0)
    assert end == pytest.approx(7.0)
    with pytest.raises(ValueError, match="shorter"):
        window_bounds(3.0, window_s=4.0)
    session = make_synthetic_stereo_session(duration_s=1.0, sample_rate=8000, seed=0)
    cropped = crop_session(session, window_s=0.40, mode="mid")
    assert cropped.duration_s == pytest.approx(0.40)
    assert ":mid0" in cropped.session_id or "mid" in cropped.session_id
