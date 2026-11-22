"""Data for SlideZoom: the same TCGA slides, labels and questions as SlideTalk, served as hierarchies.

Features : data/features/phikon/<file_id>.h5 (features (N, 768), grid (N, 2)); override with SLIDEZOOM_FEATURES
Labels   : TCGA project -> organ and cancer subtype (data/manifest.csv)
Splits   : data/splits.csv (patient-level), or --fold k for 5-fold patient-level cross-validation
"""
from __future__ import annotations

import os
import random
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .hierarchy import LEVEL_SIZES, Hierarchy, build_hierarchy

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = Path(os.environ.get("SLIDEZOOM_FEATURES", ROOT / "data/features/phikon"))

SUBTYPES = {  # TCGA project -> (organ, subtype)
    "TCGA-LUAD": ("lung", "lung adenocarcinoma"),
    "TCGA-LUSC": ("lung", "lung squamous cell carcinoma"),
    "TCGA-KIRC": ("kidney", "clear cell renal cell carcinoma"),
    "TCGA-KIRP": ("kidney", "papillary renal cell carcinoma"),
    "TCGA-KICH": ("kidney", "chromophobe renal cell carcinoma"),
}
SUBTYPE_NAMES = [s for _, s in SUBTYPES.values()]

ASK_SUBTYPE = ["What type of cancer is shown in this slide?", "What is the diagnosis?",
               "Which tumor subtype is this?", "What cancer do you see in this whole slide image?"]
ASK_ORGAN = ["Which organ is this tissue from?", "What is the tissue of origin?",
             "Which organ does this slide come from?"]


def questions_for(row, rng: random.Random) -> list[tuple[str, str, str]]:
    """(kind, question, answer) for one slide. Every answer needs the slide to be right."""
    qa = [("subtype", q, row.subtype) for q in ASK_SUBTYPE]
    qa += [("organ", q, row.organ) for q in ASK_ORGAN]
    other = rng.choice([s for s in SUBTYPE_NAMES if s != row.subtype])
    qa += [("yes/no", f"Is this a {row.subtype}?", "yes"), ("yes/no", f"Is this a {other}?", "no")]
    same_organ = [s for o, s in SUBTYPES.values() if o == row.organ]
    qa.append(("choice", f"This is a {row.organ} tumor. Is it " + " or ".join(f"a {s}" for s in same_organ) + "?",
               row.subtype))
    return qa


# ---------------------------------------------------------------------------------------------- splits
def _fold_split(m: pd.DataFrame, fold: int, n_folds: int) -> pd.Series:
    """Patient-level folds, balanced per project. test = fold, val = fold + 1, train = the rest."""
    rng = np.random.default_rng(0)
    fold_of = {}
    for _, g in m.groupby("project"):
        pats = np.array(sorted(g.patient.unique()))
        for i, p in enumerate(pats[rng.permutation(len(pats))]):
            fold_of.setdefault(p, i % n_folds)
    f = m.patient.map(fold_of)
    return np.where(f == fold, "test", np.where(f == (fold + 1) % n_folds, "val", "train"))


def slides(split: str | None = None, fold: int | None = None, n_folds: int = 5) -> pd.DataFrame:
    """Slides whose features exist on disk, with organ / subtype / split columns."""
    m = pd.read_csv(ROOT / "data/manifest.csv")
    m = m.merge(pd.read_csv(ROOT / "data/splits.csv")[["file_id", "split"]], on="file_id")
    m["organ"] = m.project.map(lambda p: SUBTYPES[p][0])
    m["subtype"] = m.project.map(lambda p: SUBTYPES[p][1])
    m = m[m.file_id.map(lambda f: (FEAT_DIR / f"{f}.h5").exists())]
    # TCGA barcode sample type: 01-09 tumour, 10-19 normal. The manifest has 4 normal-kidney slides labelled KICH.
    m = m[m.file_name.str.split("-").str[3].str[:2].astype(int) < 10].reset_index(drop=True)
    if fold is not None:
        m["split"] = _fold_split(m, fold, n_folds)
    if split:
        m = m[m.split == split]
    return m.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- slides
def read_slide(file_id: str):
    with h5py.File(FEAT_DIR / f"{file_id}.h5", "r") as f:
        return f["features"][:], f["grid"][:]


class SlideStore:
    """Every slide's full hierarchy in RAM (fp16): ~1.8 GB for the 238 slides here, built on first use."""

    def __init__(self, sizes=LEVEL_SIZES):
        self.sizes, self.cache = sizes, {}

    def get(self, file_id: str) -> Hierarchy:
        if file_id not in self.cache:
            feats, grid = read_slide(file_id)
            self.cache[file_id] = build_hierarchy(feats.astype(np.float16), grid, self.sizes)
        return self.cache[file_id]

    def view(self, file_id: str, n_tiles: int | None = None, draw: int = 0) -> Hierarchy:
        """The whole slide (n_tiles None), or the same n tiles every time for this slide and draw."""
        h = self.get(file_id)
        if not n_tiles or n_tiles >= h.n_tiles:
            return h
        rng = np.random.default_rng(zlib.crc32(file_id.encode()) + draw)
        return h.subset(np.sort(rng.permutation(h.n_tiles)[:n_tiles]))


def collate(hiers: list[Hierarchy], device) -> dict:
    """S hierarchies -> per level: feats (S, N_l, D), mask (S, N_l) True on real cells, parent (S, N_l)."""
    S, L, D = len(hiers), len(hiers[0].feats), hiers[0].feats[0].shape[1]
    out = dict(feats=[], mask=[], parent=[])
    for l in range(L):
        n = max(len(h.feats[l]) for h in hiers)
        f = torch.zeros(S, n, D)
        m = torch.zeros(S, n, dtype=torch.bool)
        p = torch.zeros(S, n, dtype=torch.long)
        for i, h in enumerate(hiers):
            k = len(h.feats[l])
            f[i, :k] = torch.from_numpy(h.feats[l].astype(np.float32))
            m[i, :k] = True
            p[i, :k] = torch.from_numpy(h.parent[l])
        out["feats"].append(f.to(device))
        out["mask"].append(m.to(device))
        out["parent"].append(p.to(device))
    return out


# ---------------------------------------------------------------------------------------------- needle task
NEEDLE_Q = ["Is there any {s} in this slide?", "Does this slide contain {s}?", "Is a focus of {s} present anywhere?"]
NEEDLE_FRACS = (0.2, 0.05, 0.01, 0.005, 0.001)


def same_organ(subtype: str) -> list[str]:
    organ = next(o for o, s in SUBTYPES.values() if s == subtype)
    return [s for o, s in SUBTYPES.values() if o == organ]


def _blob(grid: np.ndarray, n: int, center: int) -> np.ndarray:
    """The n tiles nearest to `center` on the grid (a contiguous patch), ordered by distance."""
    d = ((grid - grid[center]) ** 2).sum(1)
    return np.argsort(d, kind="stable")[:n]


def make_composite(store: SlideStore, host: str, donor: str, frac: float, rng: np.random.Generator) -> Hierarchy:
    """Host slide with a contiguous patch (frac of its tiles) replaced by a contiguous patch of the donor slide.

    Tiles keep their radial order around the patch centre, so the implant looks like one focus, not confetti.
    The hierarchy is rebuilt, so regions and sub-regions summarise the composite.
    """
    hh, dh = store.get(host), store.get(donor)
    feats, grid, dfeats, dgrid = hh.feats[-1], hh.grid, dh.feats[-1], dh.grid
    n = max(1, min(round(frac * len(feats)), len(dfeats) // 2))
    h_idx = _blob(grid, n, int(rng.integers(len(feats))))
    d_idx = _blob(dgrid, n, int(rng.integers(len(dfeats))))
    feats = feats.astype(np.float16).copy()
    feats[h_idx] = dfeats[d_idx]
    h = build_hierarchy(feats, grid, store.sizes)
    h.implant = np.sort(h_idx)
    return h


def needle_case(df: pd.DataFrame, row, positive: bool, rng: np.random.Generator, mode: str = "same"):
    """(donor file_id, subtype asked about, answer). Positive: donor of another subtype, asked about it -> yes.
    Negative control: donor of the host's own subtype (a foreign patch all the same), asked about another
    subtype -> no. mode "same": the other subtype is from the same organ (hard to recognise); "cross": from the
    other organ (easy to recognise, so finding the patch is the hard part)."""
    if mode == "cross":
        others = [s for s in SUBTYPE_NAMES if s not in same_organ(row.subtype)]
    else:
        others = [s for s in same_organ(row.subtype) if s != row.subtype]
    if positive:
        target = others[rng.integers(len(others))]
        pool = df[df.subtype == target]
    else:
        target = others[rng.integers(len(others))]
        pool = df[(df.subtype == row.subtype) & (df.file_id != row.file_id)]
    donor = pool.file_id.iloc[int(rng.integers(len(pool)))]
    return donor, target, "yes" if positive else "no"


def make_batch(rows: pd.DataFrame, store: SlideStore, n_tiles, n_questions: int, rng: random.Random,
               np_rng: np.random.Generator, device, needle: float = 0.0, donors: pd.DataFrame | None = None,
               needle_mode: str = "same"):
    """Slide-major batch: slide i owns questions i*Q .. i*Q+Q-1 (the model relies on this order).

    n_tiles: int, (low, high) for a random count per slide, or None for every tile. Coarse levels always
    summarise the whole slide; only the tile level is subsampled.
    needle: probability that a slide becomes a composite (f log-uniform in 0.1%-20%) asked needle questions;
    donors come from `donors` (the same split). Implanted tiles are always kept in the subsample, for positives
    and negative controls alike, so the label stays true and cannot leak.
    """
    hiers, questions, answers = [], [], []
    for _, row in rows.iterrows():
        if needle and np_rng.random() < needle:
            frac = float(10 ** np_rng.uniform(np.log10(NEEDLE_FRACS[-1]), np.log10(NEEDLE_FRACS[0])))
            donor, target, ans = needle_case(donors, row, bool(np_rng.random() < 0.5), np_rng, needle_mode)
            h = make_composite(store, row.file_id, donor, frac, np_rng)
            qa = [(q.format(s=target), ans) for q in NEEDLE_Q] + [(ASK_ORGAN[0], row.organ)]
        else:
            h = store.get(row.file_id)
            qa = [(q, a) for _, q, a in questions_for(row, rng)]
        n = n_tiles if (n_tiles is None or isinstance(n_tiles, int)) else int(np_rng.integers(n_tiles[0], n_tiles[1] + 1))
        if n and n < h.n_tiles:
            keep = h.implant if h.implant is not None else np.array([], dtype=np.int64)
            rest = np.setdiff1d(np.arange(h.n_tiles), keep)
            extra = np_rng.choice(rest, max(0, min(n - len(keep), len(rest))), replace=False)
            h = h.subset(np.sort(np.concatenate([keep, extra])))
        hiers.append(h)
        for q, a in rng.sample(qa, n_questions):
            questions.append(q)
            answers.append(a)
    return collate(hiers, device), questions, answers
