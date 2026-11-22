"""python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidezoom.data import collate  # noqa: E402
from slidezoom.hierarchy import build_hierarchy  # noqa: E402
from slidezoom.model import ZoomCrossAttention, SlideZoomer  # noqa: E402


def toy_slide(n_side=40, d=768, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.array([(c, r) for c in range(n_side) for r in range(n_side)])
    return build_hierarchy(rng.standard_normal((len(grid), d)).astype(np.float32), grid)


# ---------------------------------------------------------------------------------------------- hierarchy
def test_hierarchy_pools_and_nests():
    grid = np.array([(c, r) for c in range(40) for r in range(40)])
    feats = np.random.default_rng(0).standard_normal((len(grid), 4)).astype(np.float32)
    h = build_hierarchy(feats, grid)
    assert [len(f) for f in h.feats] == [4, 25, 1600]  # 40 tiles/side -> 2x2 regions, 5x5 sub-regions
    for l in (1, 2):  # each cell's mean equals the mean of its children
        for p in range(len(h.feats[l - 1])):
            kids = h.feats[l][h.parent[l] == p]
            w = np.bincount(h.parent[2], minlength=len(h.feats[1]))[h.parent[l] == p] if l == 1 else None
            np.testing.assert_allclose(np.average(kids, axis=0, weights=w), h.feats[l - 1][p], atol=1e-5)


def test_subset_keeps_whole_slide_summary():
    h = toy_slide()
    s = h.subset(np.arange(0, h.n_tiles, 7))
    assert s.feats[0] is h.feats[0] and s.n_tiles == len(range(0, h.n_tiles, 7))
    assert np.array_equal(s.parent[-1], h.parent[-1][::7])


# ---------------------------------------------------------------------------------------------- block
def test_beta_zero_means_prior_is_ignored():
    xa = ZoomCrossAttention()
    x, mem, mask = torch.randn(2, 5, 768), torch.randn(1, 30, 768), torch.ones(1, 30, dtype=torch.bool)
    prior = torch.rand(1, 10, 30)
    _, a0 = xa(x, mem, mask)
    _, a1 = xa(x, mem, mask, prior=prior)
    assert torch.allclose(a0, a1)
    with torch.no_grad():
        xa.beta.fill_(1.0)
    _, a2 = xa(x, mem, mask, prior=prior)
    assert not torch.allclose(a0, a2)


def test_questions_of_a_slide_share_its_memory():
    """Two slides x three questions: rows 0-2 read slide 0, rows 3-5 read slide 1."""
    xa = ZoomCrossAttention()
    with torch.no_grad():
        xa.gate.fill_(1.0)
    x, mem, mask = torch.randn(6, 4, 768), torch.randn(2, 20, 768), torch.ones(2, 20, dtype=torch.bool)
    y, _ = xa(x, mem, mask)
    y0, _ = xa(x[3:], mem[1:], mask[1:])
    assert torch.allclose(y[3:], y0, atol=1e-5)


def test_allowed_restricts_attention():
    xa = ZoomCrossAttention()
    allowed = torch.zeros(1, 4, 30, dtype=torch.bool)
    allowed[..., :5] = True
    _, a = xa(torch.randn(1, 4, 768), torch.randn(1, 30, 768), torch.ones(1, 30, dtype=torch.bool), allowed=allowed)
    assert a[..., 5:].abs().max() == 0


# ---------------------------------------------------------------------------------------------- model
@pytest.fixture(scope="module")
def model():
    return SlideZoomer("zoom").eval()


def test_starts_as_plain_gpt2(model):
    ids = model.tok("Question: What is this?\nAnswer:", return_tensors="pt").input_ids
    model.set_slide(None)
    plain = model.lm(input_ids=ids).logits
    model.set_slide(collate([toy_slide()], "cpu"))
    with_slide = model.lm(input_ids=ids).logits
    assert torch.allclose(plain, with_slide, atol=1e-5)  # gates are shut at initialisation


def test_prior_child_gets_parent_probability(model):
    """Block 1 (sub-regions) gets as prior the region attention of block 0, copied to every child."""
    h = toy_slide()
    seen = {}
    block1 = model.xattn[1]
    orig = block1.forward

    def spy(x, mem, mask, prior=None, allowed=None, kv=None):
        seen["prior"] = prior
        return orig(x, mem, mask, prior, allowed, kv)

    block1.forward = spy
    model.keep_attn = True
    try:
        with torch.no_grad():
            model.set_slide(collate([h], "cpu"))
            model.lm(input_ids=model.tok("Question: hi\nAnswer:", return_tensors="pt").input_ids)
    finally:
        block1.forward = orig
        model.keep_attn = False
    a0 = model.attn_log[0].mean(1)[0]                    # (T, regions): block 0, heads averaged
    expected = a0[:, torch.from_numpy(h.parent[1])]     # each sub-region gets its region's probability
    assert torch.allclose(seen["prior"][0], expected)
    # and the composed ancestor map tile -> region matches the hierarchy
    assert np.array_equal(model._ancestor(2, 0)[0].numpy(), h.parent[1][h.parent[2]])


def test_topk_reads_only_children_of_top_cells(model):
    slide = collate([toy_slide()], "cpu")
    model.set_topk((1, 2))  # 1 region -> 2 sub-regions -> at most 2 * 64 tiles
    model.keep_attn = True
    try:
        with torch.no_grad():
            model.set_slide(slide)
            model.lm(input_ids=model.tok("Question: hi\nAnswer:", return_tensors="pt").input_ids)
        assert model.attended <= 128
        tile_attn = model.attn_log[2][0].mean(0)        # (T, tiles)
        read = tile_attn > 0
        sub_of_tile = slide["parent"][2][0]
        for t in range(read.shape[0]):                  # every token reads tiles of <= 2 sub-regions
            assert sub_of_tile[read[t]].unique().numel() <= 2
    finally:
        model.set_topk(None)
        model.keep_attn = False


def test_answer_runs_with_kv_cache(model):
    out = model.answer(collate([toy_slide()], "cpu"), "What is this?", max_new=3)
    assert isinstance(out, str)
