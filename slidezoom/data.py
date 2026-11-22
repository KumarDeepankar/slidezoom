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
    m = m[m.file_id.map(lambda f: (FEAT_DIR / f"{f}.h5").exists())].reset_index(drop=True)
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


def make_batch(rows: pd.DataFrame, store: SlideStore, n_tiles, n_questions: int, rng: random.Random,
               np_rng: np.random.Generator, device):
    """Slide-major batch: slide i owns questions i*Q .. i*Q+Q-1 (the model relies on this order).

    n_tiles: int, (low, high) for a random count per slide, or None for every tile. Coarse levels always
    summarise the whole slide; only the tile level is subsampled.
    """
    hiers, questions, answers = [], [], []
    for _, row in rows.iterrows():
        h = store.get(row.file_id)
        n = n_tiles if (n_tiles is None or isinstance(n_tiles, int)) else int(np_rng.integers(n_tiles[0], n_tiles[1] + 1))
        if n and n < h.n_tiles:
            h = h.subset(np.sort(np_rng.choice(h.n_tiles, n, replace=False)))
        hiers.append(h)
        for _, q, a in rng.sample(questions_for(row, rng), n_questions):
            questions.append(q)
            answers.append(a)
    return collate(hiers, device), questions, answers
