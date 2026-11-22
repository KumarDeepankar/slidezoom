"""Region -> sub-region -> tile hierarchy of one slide, built from the tile grid.

Tiles are 256 px at 0.5 µm/px (128 µm). With the default LEVEL_SIZES:

    level 0  region      32 x 32 tiles  (~4 mm)   mean of its tiles
    level 1  sub-region   8 x  8 tiles  (~1 mm)   mean of its tiles
    level 2  tile         1 x  1                  the Phikon feature itself

parent[l][i] is the index, at level l-1, of the cell that contains cell i of level l.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

LEVEL_SIZES = (32, 8, 1)  # tiles per side of a cell, coarse to fine; must nest and end with 1


@dataclass
class Hierarchy:
    feats: list[np.ndarray]   # per level (N_l, D)
    parent: list[np.ndarray]  # per level (N_l,) index into level l-1; parent[0] is unused (zeros)
    grid: Optional[np.ndarray] = None     # (N, 2) tile column/row
    implant: Optional[np.ndarray] = None  # tiles replaced by a needle patch (data.make_composite), else None

    @property
    def n_tiles(self) -> int:
        return len(self.feats[-1])

    def subset(self, idx: np.ndarray) -> "Hierarchy":
        """Keep only the tiles `idx`. Coarse levels still summarise the whole slide."""
        imp = None
        if self.implant is not None:  # positions of the implanted tiles inside the subset
            imp = np.flatnonzero(np.isin(idx, self.implant))
        return Hierarchy(self.feats[:-1] + [self.feats[-1][idx]], self.parent[:-1] + [self.parent[-1][idx]],
                         None if self.grid is None else self.grid[idx], imp)


def _pool(feats: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Mean feature of each cell. cell: (N,) cell id in 0..K-1, every id used."""
    order = np.argsort(cell, kind="stable")
    counts = np.bincount(cell)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    sums = np.add.reduceat(feats[order].astype(np.float32), starts, axis=0)
    return (sums / counts[:, None]).astype(feats.dtype)


def build_hierarchy(feats: np.ndarray, grid: np.ndarray, sizes=LEVEL_SIZES) -> Hierarchy:
    """feats (N, D) tile features, grid (N, 2) tile column/row."""
    assert sizes[-1] == 1 and all(a % b == 0 for a, b in zip(sizes, sizes[1:])), sizes
    n = len(feats)
    cells = []  # per level: cell id of every tile
    for s in sizes:
        if s == 1:
            cells.append(np.arange(n))
        else:
            _, inv = np.unique(grid // s, axis=0, return_inverse=True)
            cells.append(inv.reshape(-1))
    level_feats = [feats if s == 1 else _pool(feats, c) for s, c in zip(sizes, cells)]
    parent = [np.zeros(len(level_feats[0]), dtype=np.int64)]
    for l in range(1, len(sizes)):
        p = np.zeros(len(level_feats[l]), dtype=np.int64)
        p[cells[l]] = cells[l - 1]  # nested sizes: every tile of a cell has the same coarser cell
        parent.append(p)
    return Hierarchy(level_feats, parent, grid)
