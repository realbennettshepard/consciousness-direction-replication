#!/bin/bash
set -eo pipefail   # NOT defeated by a pipe: pipefail makes `| tee` propagate failure
R=/Users/bennettshepard/Documents/consciousness-replication
cd "$R"
# Coefficients are norm-matched to Llama's [1,2.5,4]: median ||h|| is 6.37 (Llama
# L14/-5) vs 321.7 (Gemma-9B L20/-5), a 50.5x ratio. 144 is the paper's own value
# for this model and lands at the median of the grid, so it drives the mid-c report.
G="50,126,144,202"
for t in idaq_keying_test acquiescence_test; do
  python3 -m py_compile "$t.py"
done
for t in idaq_keying_test acquiescence_test; do
  echo "### $t (gemma-2-9b, L20/-5, c=$G) ###"
  python3 -u "$t.py" \
    --real directions_g9_full.npz:20:-5 \
    --arm placebo=directions_g9_placebo.npz:20:-5 \
    --arm permuted=directions_g9_permuted.npz:20:-5 \
    --coeffs "$G" --out "$R/g9_${t}_results.json"
  sleep 10   # let Metal buffers release between model loads
done
echo "G9_KEYING_DONE"
