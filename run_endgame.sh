#!/bin/bash
# Reordered so the highest-value results land first and the runaway job is last & capped.
#   1. GSS-decomposition, 3 models   -- settles the acquiescence question (load-bearing)
#   2. held-out MMLU n=1000          -- winner's-curse-free capability check
#   3. CoT baseline, CAPPED n=8/220  -- lowest value; the earlier n=20/320 ran 110min, so
#                                        it goes last and cannot block anything above it
set -eo pipefail
cd /Users/bennettshepard/Documents/consciousness-replication
F='it/s\|examples/s\|clean_up_tokeniz'
for t in gss_ablation_test cot_baseline_test steer_sweep_mlx; do python3 -m py_compile "$t.py"; done

for m in llama g2b g9; do
  echo "############## GSS-decomp: $m ##############"
  python3 -u gss_ablation_test.py --model "$m" 2>&1 | grep -v "$F"
  sleep 10
done

echo "############## held-out MMLU (Llama n=1000 seed 1) ##############"
python3 -u steer_sweep_mlx.py --directions directions_llama8b_full.npz \
  --layer 14 --pos -5 --coeffs 1,2.5,4 --mmlu-n 1000 --mmlu-seed 1 \
  --out steer_sweep_heldout_mmlu.json 2>&1 | grep -v "$F"
sleep 10

echo "############## CoT baseline (Llama, CAPPED n=8 max-tokens=220) ##############"
python3 -u cot_baseline_test.py --n 8 --max-tokens 220 2>&1 | grep -v "$F"

echo "ENDGAME_ALL_DONE"
