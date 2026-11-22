"""Needle-in-a-haystack test: can the model find a small patch of another tumour subtype?

    python scripts/needle.py --ckpt runs/needle_zoom/best.pt                  # every tile
    python scripts/needle.py --ckpt runs/needle_flat512/best.pt --eval-tiles 512
    python scripts/needle.py --ckpt runs/needle_zoom/best.pt --topk 2,8

Each test slide hosts a patch covering 20%, 5%, 1%, 0.5% and 0.1% of its tiles, once from another subtype of the
same organ (answer yes) and once from another slide of its own subtype (answer no). Chance = 50%.
Saves <tag>_needle.csv next to the checkpoint.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidezoom.data import SlideStore, slides  # noqa: E402
from slidezoom.evaluate import needle_eval  # noqa: E402
from slidezoom.model import SlideZoomer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-tiles", default="all")
    ap.add_argument("--topk", default=None)
    ap.add_argument("--zero-beta", action="store_true")
    ap.add_argument("--mode", choices=["same", "cross"], default=None, help="default: the mode it was trained with")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SlideZoomer(ck["args"]["mode"]).to(device)
    model.load_trainable(ck["state"])
    model.eval()
    if args.zero_beta:
        with torch.no_grad():
            for b in model.xattn:
                b.beta.zero_()
    model.set_topk(None if args.topk is None else tuple(int(k) for k in args.topk.split(",")))
    n_tiles = None if args.eval_tiles == "all" else int(args.eval_tiles)
    df = slides(args.split, ck["args"].get("fold"))
    mode = args.mode or ck["args"].get("needle_mode", "same")
    res = needle_eval(model, df, SlideStore(), n_tiles=n_tiles, mode=mode)
    print(f"{ck['args']['mode']}  needle {mode}  {len(df)} {args.split} hosts  tiles {args.eval_tiles}  topk {args.topk}  "
          f"betas {['%+.2f' % b for b in model.betas()]}")
    g = res.groupby("frac", sort=False)
    out = g.agg(acc=("correct", "mean"), acc_pos=("correct", lambda c: c[res.loc[c.index, "positive"]].mean()),
                acc_neg=("correct", lambda c: c[~res.loc[c.index, "positive"]].mean()),
                implant_read=("implant_read", "mean"), tiles_read=("tiles_read", "mean"),
                lift=("lift", "median"))
    print(out.to_string(float_format=lambda v: f"{v:.2f}"))
    tag = f"{args.split}_{Path(args.ckpt).stem}_t{args.eval_tiles}_k{args.topk or 'all'}{'_beta0' if args.zero_beta else ''}{'_cross' if mode == 'cross' else ''}".replace(",", "-")
    res.to_csv(Path(args.ckpt).resolve().parent / f"{tag}_needle.csv", index=False)


if __name__ == "__main__":
    main()
