#!/usr/bin/env bash
# Needle-in-a-haystack experiment: train every model with half of the slides turned into composites, then test
# how well each finds a patch of another subtype covering 20% ... 0.1% of a held-out slide.
#   bash scripts/run_needle.sh               # ~1.5 h on an M3 Pro; resumable (skips trained models)
#   MODE=cross bash scripts/run_needle.sh    # patch from the other organ (easy to recognise, hard to find)
cd "$(dirname "$0")/.."
PY=${PY:-python}
MODE=${MODE:-same}
P=$([[ $MODE == cross ]] && echo needlex || echo needle)  # run-name prefix
FILTER="warn\|loading\|loss_type\|redirect"

train() {  # name, train args...
  local name=$1; shift
  if [[ -f runs/$name/best.pt ]]; then
    echo "=== $name: already trained, skipping"
  else
    echo "=== $name: train $*"
    $PY scripts/train.py --needle 0.5 --needle-mode "$MODE" --out "runs/$name" "$@" 2>&1 | grep --line-buffered "^epoch\|best\|skipped\|Error"
  fi
}
needle() {  # name, needle.py args...
  local name=$1; shift
  echo "=== $name: needle test $*"
  $PY scripts/needle.py --ckpt "runs/$name/best.pt" "$@" 2>&1 | grep -v -i "$FILTER"
}

train ${P}_flat512 --mode flat --train-tiles 128,512 --eval-tiles 512   # B0 SlideTalk
train ${P}_flat    --mode flat                                          # B1 all tiles
train ${P}_multi   --mode multiscale                                    # B2 beta = 0
train ${P}_zoom    --mode zoom                                          # SlideZoom, beta starts at 0
train ${P}_zoom_b1 --mode zoom --beta-init 1                            # SlideZoom, beta starts at 1

needle ${P}_flat512 --eval-tiles 512
needle ${P}_flat
needle ${P}_multi
needle ${P}_multi   --topk 2,8
needle ${P}_zoom
needle ${P}_zoom    --topk 2,8
needle ${P}_zoom    --zero-beta
needle ${P}_zoom_b1
needle ${P}_zoom_b1 --topk 2,8
needle ${P}_zoom_b1 --zero-beta
