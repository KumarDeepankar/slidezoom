"""Train SlideZoom or a baseline. Only the projection, level embedding and 4 cross-attention blocks learn.

    python scripts/train.py --mode zoom       --out runs/zoom          # proposed model
    python scripts/train.py --mode multiscale --out runs/multiscale    # B2: hierarchy, beta fixed at 0
    python scripts/train.py --mode flat       --out runs/flat          # B1: flat cross-attention, all tiles at eval
    python scripts/train.py --mode flat --train-tiles 128,512 --eval-tiles 512 --out runs/flat512   # B0: SlideTalk
    python scripts/train.py --epochs 1 --out runs/check                # quick check

Add --fold k (0-4) for 5-fold patient-level cross-validation and --seed s for repeats.
Each epoch prints loss, val NLL, val accuracy, gates tanh(gate) and zoom strengths beta.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidezoom.data import SlideStore, make_batch, slides  # noqa: E402
from slidezoom.evaluate import ask_all, slide_nll  # noqa: E402
from slidezoom.model import SCHEDULES, SlideZoomer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tile_range(s: str):
    """'2048' -> 2048, '512,2048' -> (512, 2048), 'all' -> None."""
    if s == "all":
        return None
    lo, _, hi = s.partition(",")
    return (int(lo), int(hi)) if hi else int(lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(SCHEDULES), default="zoom")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--slides-per-step", type=int, default=8)
    ap.add_argument("--questions-per-slide", type=int, default=4)
    ap.add_argument("--train-tiles", default="512,2048", help="tiles per slide per step: n, lo,hi or all")
    ap.add_argument("--eval-tiles", default="all", help="tiles per slide at validation: n or all")
    ap.add_argument("--mem-dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gate-lr", type=float, default=1e-2, help="for the scalar gates and betas")
    ap.add_argument("--fold", type=int, default=None, help="5-fold CV; default: data/splits.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--out", default="runs/zoom")
    args = ap.parse_args()
    OUT = ROOT / args.out
    train_tiles, eval_tiles = tile_range(args.train_tiles), tile_range(args.eval_tiles)

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    rng, np_rng = random.Random(args.seed), np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_df, val_df = slides("train", args.fold), slides("val", args.fold)
    print(f"mode {args.mode}  levels {SCHEDULES[args.mode]}  slides {len(train_df)} train / {len(val_df)} val  "
          f"device {device}")

    model = SlideZoomer(args.mode, mem_dropout=args.mem_dropout).to(device)
    scalars = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith((".gate", ".beta"))]
    ids = {id(p) for p in scalars}
    others = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    print(f"trainable: {sum(p.numel() for p in others + scalars) / 1e6:.2f}M parameters "
          f"({len(scalars)} gate/beta scalars)   frozen GPT-2: {sum(p.numel() for p in model.lm.parameters()) / 1e6:.0f}M")
    opt = torch.optim.AdamW([{"params": others, "lr": args.lr, "weight_decay": args.weight_decay},
                             {"params": scalars, "lr": args.gate_lr, "weight_decay": 0.0}])
    steps_per_epoch = math.ceil(len(train_df) / args.slides_per_step)
    total, warmup = args.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    store = SlideStore()
    OUT.mkdir(parents=True, exist_ok=True)
    log, best, skipped = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = train_df.sample(frac=1, random_state=args.seed * 1000 + epoch)
        for s in range(0, len(order), args.slides_per_step):
            slide, qs, ans = make_batch(order.iloc[s:s + args.slides_per_step], store, train_tiles,
                                        args.questions_per_slide, rng, np_rng, device)
            loss = model.loss(slide, qs, ans)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad for p in others + scalars if p.grad is not None]
            if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
                skipped += 1  # one bad step would turn every weight into NaN; drop it instead
                print(f"  epoch {epoch:2d}  skipped a step with a non-finite loss or gradient ({skipped} so far)")
                sched.step()
                continue
            torch.nn.utils.clip_grad_norm_(others, 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            if args.log_every and len(losses) % args.log_every == 0:
                print(f"  epoch {epoch:2d}  step {len(losses):3d}/{steps_per_epoch}  "
                      f"loss {np.mean(losses[-args.log_every:]):.3f}  lr {sched.get_last_lr()[0]:.1e}", flush=True)
        model.eval()
        nll = slide_nll(model, val_df, store, eval_tiles)
        acc = ask_all(model, val_df, store, kinds=("subtype", "organ"), n_tiles=eval_tiles).groupby("kind").correct.mean()
        row = dict(epoch=epoch, loss=float(np.mean(losses)), val_nll=float(nll.nll.mean()),
                   val_subtype=float(acc["subtype"]), val_organ=float(acc["organ"]),
                   gates=model.gates(), betas=model.betas(), skipped_steps=skipped, seconds=round(time.time() - t0))
        log.append(row)
        print(f"epoch {epoch:2d}  loss {row['loss']:.3f}  val nll {row['val_nll']:.3f}  subtype {row['val_subtype']:.2f}  "
              f"organ {row['val_organ']:.2f}  gates [{' '.join(f'{g:+.2f}' for g in row['gates'])}]  "
              f"beta [{' '.join(f'{b:+.2f}' for b in row['betas'])}]  {row['seconds']}s", flush=True)
        ckpt = {"state": model.trainable_state(), "epoch": epoch, "args": vars(args)}
        if row["val_nll"] < best:
            best = row["val_nll"]
            torch.save(ckpt, OUT / "best.pt")
        (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
    torch.save(ckpt, OUT / "last.pt")
    best_ep = min(log, key=lambda r: r["val_nll"])["epoch"]
    print(f"best val nll {best:.3f} at epoch {best_ep}; checkpoints in {OUT}")


if __name__ == "__main__":
    main()
