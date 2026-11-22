"""Evaluate a checkpoint on held-out patients, optionally with a top-k tile budget.

    python scripts/test.py --ckpt runs/zoom/best.pt                    # read every tile
    python scripts/test.py --ckpt runs/zoom/best.pt --topk 2,8         # 2 regions -> 8 sub-regions (<= 512 tiles)
    python scripts/test.py --ckpt runs/flat512/best.pt --eval-tiles 512

Saves <tag>_answers.csv (per slide and question) and <tag>_nll.csv (per slide) for scripts/compare.py.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidezoom.data import ASK_SUBTYPE, SlideStore, slides  # noqa: E402
from slidezoom.evaluate import ask_all, slide_nll  # noqa: E402
from slidezoom.model import SlideZoomer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-tiles", default="all", help="n or all")
    ap.add_argument("--topk", default=None, help="k per zoom-in step, e.g. 2,8 (zoom / multiscale only)")
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--zero-beta", action="store_true", help="ablation: switch the learned zoom prior off")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    targs = ck["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SlideZoomer(targs["mode"]).to(device)
    model.load_trainable(ck["state"])
    model.eval()
    if args.zero_beta:
        with torch.no_grad():
            for b in model.xattn:
                b.beta.zero_()
    model.set_topk(None if args.topk is None else tuple(int(k) for k in args.topk.split(",")))
    n_tiles = None if args.eval_tiles == "all" else int(args.eval_tiles)
    df = slides(args.split, targs.get("fold"))
    store = SlideStore()
    print(f"{targs['mode']}  {len(df)} {args.split} slides  tiles {args.eval_tiles}  topk {args.topk}\n"
          f"gates {['%+.2f' % g for g in model.gates()]}  betas {['%+.2f' % b for b in model.betas()]}\n")

    res = ask_all(model, df, store, n_tiles=n_tiles, votes=args.votes)
    nll = slide_nll(model, df, store, n_tiles)
    acc = res.groupby("kind", sort=False).correct.mean()
    for k, v in acc.items():
        print(f"   {k:22s} {v:.2f}")
    print(f"\n   answer NLL per slide   {nll.nll.mean():.3f}")
    print(f"   tiles read per token   {nll.tiles_read.mean():.0f} (slides have {df.file_id.map(lambda f: store.get(f).n_tiles).mean():.0f} on average)")
    blind = model.answer(None, ASK_SUBTYPE[0])
    print(f"   no slide control: '{ASK_SUBTYPE[0]}' -> '{blind}'")

    OUT = Path(args.ckpt).resolve().parent
    tag = f"{args.split}_{Path(args.ckpt).stem}_t{args.eval_tiles}_k{args.topk or 'all'}_v{args.votes}{'_beta0' if args.zero_beta else ''}".replace(",", "-")
    res.to_csv(OUT / f"{tag}_answers.csv", index=False)
    nll.to_csv(OUT / f"{tag}_nll.csv", index=False)
    (OUT / f"{tag}_results.json").write_text(json.dumps(dict(
        mode=targs["mode"], split=args.split, fold=targs.get("fold"), eval_tiles=args.eval_tiles, topk=args.topk,
        n_slides=len(df), accuracy=acc.to_dict(), nll=float(nll.nll.mean()), tiles_read=float(nll.tiles_read.mean()),
        gates=model.gates(), betas=model.betas(), no_slide_answer=blind), indent=1))
    print(f"\nsaved {OUT}/{tag}_*")


if __name__ == "__main__":
    main()
