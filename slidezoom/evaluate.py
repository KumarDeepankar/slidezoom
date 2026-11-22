"""Scoring a model by asking questions (exact-match answers) and by per-slide answer likelihood."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch

import zlib

from .data import (ASK_ORGAN, ASK_SUBTYPE, NEEDLE_FRACS, NEEDLE_Q, SUBTYPE_NAMES, SUBTYPES, SlideStore,
                   collate, make_composite, needle_case)

# question kind -> (question for a row, correct answer for a row)
PROBES = {
    "subtype": (lambda r: ASK_SUBTYPE[0], lambda r: r.subtype),
    "organ": (lambda r: ASK_ORGAN[0], lambda r: r.organ),
    "yes/no (true)": (lambda r: f"Is this a {r.subtype}?", lambda r: "yes"),
    "yes/no (false)": (lambda r: f"Is this a {SUBTYPE_NAMES[(SUBTYPE_NAMES.index(r.subtype) + 2) % 5]}?",
                       lambda r: "no"),
    "choice within organ": (
        lambda r: f"This is a {r.organ} tumor. Is it "
                  + " or ".join(f"a {s}" for o, s in SUBTYPES.values() if o == r.organ) + "?",
        lambda r: r.subtype),
}


def ask_all(model, df: pd.DataFrame, store: SlideStore, kinds=tuple(PROBES), n_tiles: int | None = None,
            votes: int = 1, view=None) -> pd.DataFrame:
    """One row per (slide, question kind): question, truth, answer, correct, tiles read per token.

    n_tiles: None reads every tile; an int reads a fixed random sample of that size.
    view: optional function row_index -> slide batch, to show the model another slide (controls).
    """
    out = []
    for i, r in df.iterrows():
        views = ([view(i)] if view is not None else
                 [collate([store.view(r.file_id, n_tiles, d)], model.device) for d in range(votes)])
        for k in kinds:
            q, truth = PROBES[k][0](r), PROBES[k][1](r)
            answers = []
            for x in views:
                answers.append(model.answer(x, q))
            a = Counter(answers).most_common(1)[0][0]
            out.append(dict(file_id=r.file_id, subtype=r.subtype, kind=k, question=q, truth=truth, answer=a,
                            correct=a.strip().lower() == truth, tiles_read=model.attended))
    return pd.DataFrame(out)


@torch.no_grad()
def slide_nll(model, df: pd.DataFrame, store: SlideStore, n_tiles: int | None = None) -> pd.DataFrame:
    """Mean answer NLL over all probe questions, per slide: a smoother, paired metric than accuracy."""
    rows = []
    for _, r in df.iterrows():
        slide = collate([store.view(r.file_id, n_tiles)], model.device)
        qs = [PROBES[k][0](r) for k in PROBES]
        ans = [PROBES[k][1](r) for k in PROBES]
        nll = model.loss(slide, qs, ans, reduce=False)
        rows.append(dict(file_id=r.file_id, subtype=r.subtype, nll=nll.mean().item(), tiles_read=model.attended))
    return pd.DataFrame(rows)


@torch.no_grad()
def needle_eval(model, df: pd.DataFrame, store: SlideStore, fracs=NEEDLE_FRACS, n_tiles: int | None = None,
                pointing: bool = True, mode: str = "same") -> pd.DataFrame:
    """Each slide of df hosts, at every frac, one positive and one negative-control composite (donors from df).

    The same composites for every model (seeded by slide and frac). With n_tiles, a fixed random sample is read,
    with no special treatment of the implant: that is the weakness of random sampling being measured.
    pointing: share of the tile-level attention (last prompt token) that falls on the implant, and its lift
    over chance (implant share of the tiles read).
    """
    rows = []
    for _, r in df.iterrows():
        for fi, frac in enumerate(fracs):
            for positive in (True, False):
                rng = np.random.default_rng(zlib.crc32(r.file_id.encode()) * 10 + fi * 2 + int(positive))
                donor, target, truth = needle_case(df, r, positive, rng, mode)
                h = make_composite(store, r.file_id, donor, frac, rng)
                if n_tiles and n_tiles < h.n_tiles:
                    h = h.subset(np.sort(rng.permutation(h.n_tiles)[:n_tiles]))
                slide = collate([h], model.device)
                q = NEEDLE_Q[0].format(s=target)
                a = model.answer(slide, q, max_new=4)
                row = dict(file_id=r.file_id, subtype=r.subtype, frac=frac, positive=positive, target=target,
                           truth=truth, answer=a, correct=a.strip().lower().startswith(truth),
                           implant_read=len(h.implant), tiles_read=model.attended)
                if pointing and len(h.implant):
                    att = model.attention_map(slide, q)[2]  # tile level
                    mass = float(att[h.implant].sum())
                    row.update(mass=mass, lift=mass / (len(h.implant) / h.n_tiles))
                rows.append(row)
    return pd.DataFrame(rows)
