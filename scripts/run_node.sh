#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rank=${1:?node rank 0 or 1}; phase=${2:?train, smoke, eval or summarize}; step=${3:-0}
[[ "$rank" == 0 || "$rank" == 1 ]]
export RUN_DIR=${RUN_DIR:-$PWD/runs/qwen3vl8b}
mkdir -p "$RUN_DIR/logs"
case "$phase" in
 train|smoke)
  : "${MASTER_ADDR:?set node 0 reachable address}"
  extra=(); [[ "$phase" != smoke ]] || extra=(--smoke)
  python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank="$rank" \
    --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29657}" train.py --until "$step" "${extra[@]}"
  ;;
 eval)
  pids=()
  for gpu in {0..7}; do
   shard=$((rank*8+gpu))
   if [[ -f "$RUN_DIR/eval_step${step}/shard${shard}.summary.json" ]]; then continue; fi
   CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --step "$step" --shard "$shard" >"$RUN_DIR/logs/eval_${step}_${shard}.log" 2>&1 &
   pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]]
  ;;
 summarize) python summarize.py --step "$step" ;;
 *) echo 'Unknown phase' >&2; exit 2 ;;
esac
