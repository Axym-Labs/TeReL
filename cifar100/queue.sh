#!/bin/bash
# usage: queue.sh CONFIG OUTDIR MAXJOBS ID...  (starts each run when fewer than MAXJOBS probe_grid runs are active)
config=$1; out=$2; max=$3; shift 3
G='{"learning_rate":[0.003,0.01,0.03],"batch_size":[1024],"weight_decay":[0.0]}'
for id in "$@"; do
  while [ $(pgrep -fc "python probe_grid.py") -ge $max ]; do sleep 20; done
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ../.venv/bin/python probe_grid.py --config $config --runs $id --output $out --grid "${GRID:-$G}" ${DATASET:+--dataset $DATASET} ${SEED:+--seed $SEED} > $out/$id${SEED:+-s$SEED}.log 2>&1 &
  sleep 60
done
wait
