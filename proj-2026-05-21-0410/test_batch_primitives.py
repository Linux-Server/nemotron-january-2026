"""Structural unit test for batch_primitives (no NeMo needed). Inference-correctness is Probe B.

Verifies: cache stack→scatter round-trips on the CORRECT axes (dim1 channel/time, dim0 len), the
hypothesis flattening + alias guard, ragged-batch rejection, and the grouping key.
"""
import sys
import torch

sys.path.insert(0, "src")
from nemotron_speech.batch_primitives import (  # noqa: E402
    batch_group_key, ready_predicate, stack_processed, stack_caches, scatter_cache_row,
    stack_hypotheses, stack_pred_out,
)

LAYERS, CT, D, TT = 18, 9, 512, 5  # representative conformer cache dims


def b1_cache(seed):
    g = torch.Generator().manual_seed(seed)
    clc = torch.randn(LAYERS, 1, CT, D, generator=g)
    clt = torch.randn(LAYERS, 1, D, TT, generator=g)
    clcl = torch.tensor([CT], dtype=torch.long)
    return clc, clt, clcl


def test_cache_stack_scatter_roundtrip():
    caches = [b1_cache(i) for i in range(4)]
    clc, clt, clcl = stack_caches(caches)
    assert clc.shape == (LAYERS, 4, CT, D), clc.shape          # batch on dim 1
    assert clt.shape == (LAYERS, 4, D, TT), clt.shape          # batch on dim 1
    assert clcl.shape == (4,), clcl.shape                       # batch on dim 0
    for i in range(4):
        rclc, rclt, rclcl = scatter_cache_row(clc, clt, clcl, i)
        assert torch.equal(rclc, caches[i][0]), f"channel row {i} mismatch"
        assert torch.equal(rclt, caches[i][1]), f"time row {i} mismatch"
        assert torch.equal(rclcl, caches[i][2]), f"len row {i} mismatch"
    print("  cache stack→scatter round-trip (dim1/dim0): OK")


def test_scattered_cache_rows_have_independent_storage():
    caches = [b1_cache(i) for i in range(3)]
    clc, clt, clcl = stack_caches(caches)
    source_clc = clc.clone()
    source_clt = clt.clone()
    source_clcl = clcl.clone()

    row0 = scatter_cache_row(clc, clt, clcl, 0)
    row1 = scatter_cache_row(clc, clt, clcl, 1)
    row0_before = tuple(t.clone() for t in row0)
    row1_before = tuple(t.clone() for t in row1)

    row0[0].add_(1000)
    row0[1].mul_(0)
    row0[2].fill_(123)

    assert torch.equal(clc, source_clc), "mutating scattered channel row changed source batch"
    assert torch.equal(clt, source_clt), "mutating scattered time row changed source batch"
    assert torch.equal(clcl, source_clcl), "mutating scattered len row changed source batch"
    assert not torch.equal(row0[0], row0_before[0])
    assert not torch.equal(row0[1], row0_before[1])
    assert not torch.equal(row0[2], row0_before[2])
    assert torch.equal(row1[0], row1_before[0]), "mutating row 0 changed scattered row 1"
    assert torch.equal(row1[1], row1_before[1]), "mutating row 0 changed scattered row 1"
    assert torch.equal(row1[2], row1_before[2]), "mutating row 0 changed scattered row 1"
    print("  scattered cache clone-independent storage: OK")


def test_processed_stack_and_ragged_rejection():
    mels = [torch.randn(1, 128, 16) for _ in range(3)]
    proc, lens = stack_processed(mels)
    assert proc.shape == (3, 128, 16) and lens.tolist() == [16, 16, 16]
    try:
        stack_processed([torch.randn(1, 128, 16), torch.randn(1, 128, 20)])
        assert False, "ragged batch should have been rejected"
    except AssertionError as e:
        assert "ragged" in str(e)
    try:
        stack_processed([torch.randn(2, 128, 16), torch.randn(1, 128, 16)])
        assert False, "shape[0] != 1 should have been rejected"
    except AssertionError as e:
        assert "[1,F,T]" in str(e)
    try:
        stack_processed([torch.randn(1, 128, 16), torch.randn(1, 128, 16, dtype=torch.float64)])
        assert False, "dtype mismatch should have been rejected"
    except AssertionError as e:
        assert "dtype mismatch" in str(e)
    try:
        stack_processed([torch.randn(1, 128, 16), torch.empty(1, 128, 16, device="meta")])
        assert False, "device mismatch should have been rejected"
    except AssertionError as e:
        assert "device mismatch" in str(e)
    print("  processed stack + ragged rejection: OK")


def test_hypotheses_flatten_and_alias_guard():
    h0, h1 = object(), object()
    assert stack_hypotheses([[h0], [h1]]) == [h0, h1]
    assert stack_hypotheses([None, None]) == [None, None]
    try:
        stack_hypotheses([[h0], None])
        assert False, "mixed fresh/established hypotheses should have been rejected"
    except AssertionError as e:
        assert "uniformly None-or-not" in str(e)
    shared = object()
    try:
        stack_hypotheses([[shared], [shared]])
        assert False, "aliasing should have been rejected"
    except AssertionError as e:
        assert "aliasing" in str(e)
    # pred_out: None only if all rows lack it
    assert stack_pred_out([[1], [2]]) == [1, 2]
    assert stack_pred_out([None, None]) is None
    try:
        stack_pred_out([[1], None])
        assert False, "mixed fresh/established pred_out should have been rejected"
    except AssertionError as e:
        assert "uniformly None-or-not" in str(e)
    try:
        stack_pred_out([[1], [2]], rnnt=False)
        assert False, "non-RNNT pred_out stack should have been rejected"
    except AssertionError as e:
        assert "RNNT" in str(e)
    print("  hypothesis flatten + alias guard + pred_out: OK")


def test_group_key():
    a = batch_group_key("en-US", False, 2, 25, "greedy")
    b = batch_group_key("en-US", False, 2, 25, "greedy")
    c = batch_group_key("es-ES", False, 2, 25, "greedy")
    d = batch_group_key("en-US", False, 0, 25, "greedy")  # first-chunk drop differs
    assert a == b and a != c and a != d
    print("  grouping key (lang/drop/chunk distinctions): OK")


def test_ready_predicate():
    kwargs = dict(
        synthetic_prefix_samples=0,
        total_audio_samples=2720,
        emitted_frames=0,
        shift_frames=16,
        hop_samples=160,
        pending_audio_len=2720,
        preprocess_new_audio_samples=2720,
    )
    assert ready_predicate(**kwargs)
    assert not ready_predicate(**{**kwargs, "total_audio_samples": 2719})
    assert not ready_predicate(**{**kwargs, "pending_audio_len": 2719})
    assert ready_predicate(**{**kwargs, "synthetic_prefix_samples": 160, "total_audio_samples": 2560})
    print("  exact two-guard ready predicate: OK")


if __name__ == "__main__":
    test_cache_stack_scatter_roundtrip()
    test_scattered_cache_rows_have_independent_storage()
    test_processed_stack_and_ragged_rejection()
    test_hypotheses_flatten_and_alias_guard()
    test_group_key()
    test_ready_predicate()
    print("ALL batch_primitives tests PASSED")
