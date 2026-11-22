# SlideZoom

**Progressive-zoom cross-attention:** a frozen language model reads a whole-slide image from coarse to fine,
the way a pathologist scans at low magnification and then zooms in.

Builds on [SlideTalk](https://github.com/KumarDeepankar/slidetalk) (frozen GPT-2 + Flamingo-style gated cross-attention over Phikon tile
features). It uses the same slides, labels and questions, and leaves SlideTalk untouched.

![SlideZoom architecture](docs/slidezoom.png)

## Problem

SlideTalk reads a **random 512 of ~4,000–20,000 tiles** per slide. A small tumour focus can be missing from
the sample, and attending to every tile in every block costs too much.

## Idea

Build a hierarchy from the tile grid (tiles are 128 µm):

| Level | Cell | Token |
|---|---|---|
| 0 region | 32 × 32 tiles (~4 mm) | mean of its tiles |
| 1 sub-region | 8 × 8 tiles (~1 mm) | mean of its tiles |
| 2 tile | 1 tile | Phikon feature |

The 4 cross-attention blocks read the slide coarse to fine, along GPT-2's depth:

```
GPT-2 block 3  ──► x-attn over regions
GPT-2 block 6  ──► x-attn over sub-regions  ┐
GPT-2 block 9  ──► x-attn over tiles        ├ guided by the previous block's attention
GPT-2 block 12 ──► x-attn over tiles        ┘
```

**Zoom prior.** For each text token *t*, the previous block's attention (at the coarser level) biases this
block's logits:

```
logits_l(t, c) = q_t · k_c / √d  +  β_l · log a_{l-1}(t, ancestor(c))
```

- β starts at **0**, so at initialisation the model is a plain multi-scale Flamingo model; training decides
  how much to zoom. Like the gates, it starts at zero.
- The prior is computed per text token, so each question can zoom into a different place.
- **Top-k at inference** (`--topk 2,8`): each token reads only the children of the top-2 regions, then of the
  top-8 sub-regions, so at most 8 × 64 = 512 tiles out of the whole slide.

Code: `slidezoom/model.py` (`ZoomCrossAttention`, `SlideZoomer._xattn`) and `slidezoom/hierarchy.py`.

## Models compared (same code, `--mode`)

| Name | Command | Rules out |
|---|---|---|
| B0 SlideTalk | `--mode flat --train-tiles 128,512 --eval-tiles 512` | reference |
| B1 flat, all tiles | `--mode flat` | "the gain only comes from more tiles" |
| B2 multiscale | `--mode multiscale` (β fixed at 0) | "the gain comes from multi-scale tokens, not the zoom" |
| B3 budget | zoom checkpoint, `test.py --topk 2,8` | "the gain comes from more compute" |
| **Zoom** | `--mode zoom` | — |

Except for B0, all models train on 512–2,048 random tiles per slide per step (coarse levels always summarise
the whole slide), and are evaluated on every tile.

## Run

```bash
# Python 3.11, same requirements as SlideTalk
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests -q

python scripts/train.py --mode zoom --out runs/zoom                 # ~21 min on an M3 Pro
python scripts/test.py  --ckpt runs/zoom/best.pt                    # every tile
python scripts/test.py  --ckpt runs/zoom/best.pt --topk 2,8         # tile budget
python scripts/compare.py runs/flat512/test_best_t512_kall_v1 runs/zoom/test_best_tall_kall_v1
```

`--fold 0..4` gives 5-fold patient-level cross-validation; use it with several `--seed` values for error bars.
42 test slides alone give a ±13-point 95% CI, too wide to see a 5-point gain.

`data/features` links to the Phikon features already extracted for SlideTalk (or set `SLIDEZOOM_FEATURES`).

## Measurement plan

1. **Baseline with error bars:** B0 under 5-fold CV × 5 seeds.
2. **Global tasks (subtype, organ):** zoom should be non-inferior to B0 (within 2 points). Little gain is
   expected, because any 512 tiles show the cancer type.
3. **Focal task (main experiment, not built yet):** implant a block of tumour tiles covering f = 20% … 0.1%
   of a normal-tissue slide and plot accuracy against f. Random sampling must fail as f shrinks; zoom should not.
   Needs TCGA "Solid Tissue Normal" slides.
4. **Mechanism:** β > 0 in every seed; attention mass on the implanted region; deletion test (removing the
   top-attended tiles hurts more than removing random tiles); attention entropy falls with depth.
5. **Efficiency:** accuracy vs tiles read (top-k sweep).

## Prior work to set apart

HIPT (Chen et al., CVPR 2022) is a hierarchical slide encoder that isn't conditioned on a question. ZoomMIL
(Thandiackal et al., ECCV 2022) zooms differentiably for classification, without a language model.
Flamingo/Perceiver use single-scale cross-attention. The difference here is **zooming conditioned on each text
token, across the depth of a frozen language model.**
