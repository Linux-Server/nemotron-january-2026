"""Structural unit test for batch_primitives (no NeMo needed). Inference-correctness is Probe B.

Verifies: cache stack→scatter round-trips on the CORRECT axes (dim1 channel/time, dim0 len), the
hypothesis flattening + alias guard, ragged-batch rejection, and the grouping key.
"""
import sys
import torch

sys.path.insert(0, "src")
from nemotron_speech.batch_primitives import (  # noqa: E402
    batch_group_key, stack_processed, stack_caches, scatter_cache_row,
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


def test_processed_stack_and_ragged_rejection():
    mels = [torch.randn(1, 128, 16) for _ in range(3)]
    proc, lens = stack_processed(mels)
    assert proc.shape == (3, 128, 16) and lens.tolist() == [16, 16, 16]
    try:
        stack_processed([torch.randn(1, 128, 16), torch.randn(1, 128, 20)])
        assert False, "ragged batch should have been rejected"
    except AssertionError as e:
        assert "ragged" in str(e)
    print("  processed stack + ragged rejection: OK")


def test_hypotheses_flatten_and_alias_guard():
    h0, h1 = object(), object()
    assert stack_hypotheses([[h0], [h1], None]) == [h0, h1, None]
    shared = object()
    try:
        stack_hypotheses([[shared], [shared]])
        assert False, "aliasing should have been rejected"
    except AssertionError as e:
        assert "aliasing" in str(e)
    # pred_out: None if any row lacks it
    assert stack_pred_out([[1], [2]]) == [1, 2]
    assert stack_pred_out([[1], None]) is None
    print("  hypothesis flatten + alias guard + pred_out: OK")


def test_group_key():
    a = batch_group_key("en-US", False, 2, 25, "greedy")
    b = batch_group_key("en-US", False, 2, 25, "greedy")
    c = batch_group_key("es-ES", False, 2, 25, "greedy")
    d = batch_group_key("en-US", False, 0, 25, "greedy")  # first-chunk drop differs
    assert a == b and a != c and a != d
    print("  grouping key (lang/drop/chunk distinctions): OK")


if __name__ == "__main__":
    test_cache_stack_scatter_roundtrip()
    test_processed_stack_and_ragged_rejection()
    test_hypotheses_flatten_and_alias_guard()
    test_group_key()
    print("ALL batch_primitives tests PASSED")
