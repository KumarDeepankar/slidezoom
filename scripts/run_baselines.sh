#!/usr/bin/env bash
# Train and test the baselines B0-B2, then compare each with SlideZoom (runs/zoom) on the same test patients.
#   bash scripts/run_baselines.sh            # ~45 min on an M3 Pro
# Needs runs/zoom/best.pt and its test results (python scripts/test.py --ckpt runs/zoom/best.pt).
cd "$(dirname "$0")/.."
PY=${PY:-python}

run() {  # name, train args..., then -- test args...
  local name=$1; shift
  local targs=() eargs=()
  while [[ $# -gt 0 && $1 != -- ]]; do targs+=("$1"); shift; done
  [[ $# -gt 0 ]] && shift
  eargs=("$@")
  if [[ -f runs/$name/best.pt ]]; then
    echo "=== $name: already trained (runs/$name/best.pt), skipping training"
  else
    echo "=== $name: train ${targs[*]}"
    $PY scripts/train.py --out "runs/$name" "${targs[@]}" 2>&1 | grep --line-buffered "^epoch\|best\|skipped\|Error"
  fi
  echo "=== $name: test ${eargs[*]:-}"
  $PY scripts/test.py --ckpt "runs/$name/best.pt" ${eargs[@]+"${eargs[@]}"} 2>&1 | grep -v -i "warn\|loading\|loss_type\|redirect\|following slide\|^The cancer"
}

run flat512    --mode flat --train-tiles 128,512 --eval-tiles 512 -- --eval-tiles 512   # B0 SlideTalk
run flat       --mode flat                                                              # B1 all tiles
run multiscale --mode multiscale                                                        # B2 beta = 0

echo "=== zoom with beta forced to 0 at test time (does the learned prior matter?)"
$PY scripts/test.py --ckpt runs/zoom/best.pt --zero-beta 2>&1 | grep -v -i "warn\|loading\|loss_type\|redirect\|following slide\|^The cancer"

Z=runs/zoom/test_best_tall_kall_v1
for b in flat512/test_best_t512_kall_v1 flat/test_best_tall_kall_v1 multiscale/test_best_tall_kall_v1 zoom/test_best_tall_kall_v1_beta0; do
  echo "=== compare: A = $b   B = SlideZoom"
  $PY scripts/compare.py "runs/$b" "$Z"
done
