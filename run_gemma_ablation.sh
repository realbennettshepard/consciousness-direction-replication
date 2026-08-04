#!/bin/bash
# Full safety-ablation arm on both Gemma models, SERIALLY. One MLX process at a time:
# 9B peaks at 9.5GB and this box is 24GB with ~11GB wired by the model, so two at once
# thrash swap (measured 13s -> 65s per prefill earlier this session).
#
# Each model runs extract -> select -> verify -> outcomes. set -eo pipefail so a failed
# stage stops that model rather than silently proceeding to read a stale direction.
set -eo pipefail
cd /Users/bennettshepard/Documents/consciousness-replication

python3 -m py_compile refusal_ablation.py || { echo "SYNTAX FAIL"; exit 1; }

run_model () {
  local tag="$1"
  echo "############## $tag : extract ##############"
  python3 -u refusal_ablation.py --stage extract  --model "$tag" 2>&1 | grep -v "it/s\|examples/s\|clean_up_tokeniz"
  echo "############## $tag : select ##############"
  python3 -u refusal_ablation.py --stage select   --model "$tag" 2>&1 | grep -v "it/s\|examples/s\|clean_up_tokeniz"
  echo "############## $tag : verify ##############"
  python3 -u refusal_ablation.py --stage verify   --model "$tag" 2>&1 | grep -v "it/s\|examples/s\|clean_up_tokeniz"
  echo "############## $tag : outcomes ##############"
  python3 -u refusal_ablation.py --stage outcomes --model "$tag" 2>&1 | grep -v "it/s\|examples/s\|clean_up_tokeniz"
  sleep 15   # let Metal buffers release before the next model loads
}

run_model g2b
run_model g9
echo "GEMMA_ABLATION_ALL_DONE"
