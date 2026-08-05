#!/bin/bash
# Remaining GPU jobs, serially (one MLX process at a time on the 24GB box).
# Ordered fast-and-high-value first so results land early:
#   1-3. GSS under safety ablation, all three models  (Experiment 4's ablation half)
#   4.   CoT-format IDAQ baseline on Llama            (does CoT close the baseline gap?)
#   5.   held-out MMLU n=1000, seed!=selection        (winner's-curse-free capability)
set -eo pipefail
cd /Users/bennettshepard/Documents/consciousness-replication
F='it/s\|examples/s\|clean_up_tokeniz'

for t in gss_ablation_test cot_baseline_test; do python3 -m py_compile "$t.py"; done
python3 -m py_compile steer_sweep_mlx.py

for m in llama g2b g9; do
  echo "############## GSS-ablation: $m ##############"
  python3 -u gss_ablation_test.py --model "$m" 2>&1 | grep -v "$F"
  sleep 10
done

echo "############## CoT baseline (Llama, n=20) ##############"
python3 -u cot_baseline_test.py --n 20 2>&1 | grep -v "$F"
sleep 10

echo "############## held-out MMLU (Llama, n=1000, seed 1) ##############"
python3 -u steer_sweep_mlx.py --directions directions_llama8b_full.npz \
  --layer 14 --pos -5 --coeffs 1,2.5,4 --mmlu-n 1000 --mmlu-seed 1 \
  --out steer_sweep_heldout_mmlu.json 2>&1 | grep -v "$F"

echo "FINAL_BATCH_ALL_DONE"
