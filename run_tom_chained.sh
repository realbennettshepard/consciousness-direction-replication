#!/bin/bash
# Serialize: never two MLX processes at once on this 24GB box. Peak is 9.5GB each,
# and running the HI-ToM smoke test alongside the Gemma-9B extraction degraded an
# identical 13-token prefill from 13s to 65s monotonically.
python3 -m py_compile tom_test.py || { echo "SYNTAX FAIL - not launching"; exit 1; }
echo "syntax ok; waiting for gemma-9b (pid $1) to exit"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "gemma done at $(date +%H:%M); swap: $(sysctl -n vm.swapusage)"
sleep 20   # let Metal buffers release
exec python3 -u tom_test.py --real directions_llama8b_full.npz:14:-5 \
  --arm placebo=directions_placebo.npz:14:-5 --coeffs 1,2.5,4 --n 200
