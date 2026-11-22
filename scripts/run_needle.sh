#!/usr/bin/env bash
# Needle-in-a-haystack experiment: train every model with half of the slides turned into composites, then test
# how well each finds a patch of another subtype covering 20% ... 0.1% of a held-out slide.
#   bash scripts/run_needle.sh               # ~1.5 h on an M3 Pro; resumable (skips trained models)
cd "$(dirname "$0")/.."
PY=${PY:-python}
FILTER="warn\|loading\|loss_type\|redirect"

train() {  # name, train args...
  local name=$1; shift
  if [[ -f runs/$name/best.pt ]]; then
    echo "=== $name: already trained, skipping"
  else
    echo "=== $name: train $*"
    $PY scripts/train.py --needle 0.5 --out "runs/$name" "$@" 2>&1 | grep --line-buffered "^epoch\|best\|skipped\|Error"
  fi
}
needle() {  # name, needle.py args...
  local name=$1; shift
  echo "=== $name: needle test $*"
  $PY scripts/needle.py --ckpt "runs/$name/best.pt" "$@" 2>&1 | grep -v -i "$FILTER"
}

train needle_flat512 --mode flat --train-tiles 128,512 --eval-tiles 512   # B0 SlideTalk
train needle_flat    --mode flat                                          # B1 all tiles
train needle_multi   --mode multiscale                                    # B2 beta = 0
train needle_zoom    --mode zoom                                          # SlideZoom, beta starts at 0
train needle_zoom_b1 --mode zoom --beta-init 1                            # SlideZoom, beta starts at 1

needle needle_flat512 --eval-tiles 512
needle needle_flat
needle needle_multi
needle needle_multi   --topk 2,8
needle needle_zoom
needle needle_zoom    --topk 2,8
needle needle_zoom    --zero-beta
needle needle_zoom_b1
needle needle_zoom_b1 --topk 2,8
needle needle_zoom_b1 --zero-beta
