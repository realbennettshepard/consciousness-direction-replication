#!/bin/bash
# Re-run GSS-under-ablation on all three models WITH the acquiescence split, to test
# whether the two Gemma "reproductions" (pooled dKL +0.936 and +2.029, 3-6x the paper's
# +0.314) are a genuine move toward humans or a Yes-bias coinciding with the affirmative
# human majority. The first pass only saved pooled numbers; this saves the split.
set -eo pipefail
cd /Users/bennettshepard/Documents/consciousness-replication
F='it/s\|examples/s\|clean_up_tokeniz'
python3 -m py_compile gss_ablation_test.py
for m in llama g2b g9; do
  echo "############## GSS-decomp: $m ##############"
  python3 -u gss_ablation_test.py --model "$m" 2>&1 | grep -v "$F"
  sleep 10
done
echo "GSS_DECOMP_ALL_DONE"
