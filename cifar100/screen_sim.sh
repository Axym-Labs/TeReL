#!/bin/bash
# Short CIFAR-100 screen: TeReL LC+FC with a larger temporal (invariance) coefficient.
# Usage: screen_sim.sh K...   (one run per similarity_coefficient K; 12k training samples, 1 epoch, seed 401)
cd "$(dirname "$0")"
G='{"learning_rate":[0.003,0.01,0.03],"batch_size":[1024],"weight_decay":[0.0]}'
for k in "$@"; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup ../.venv/bin/python probe_grid.py \
    --config round4.yaml --runs terel-lcfc-best --output round6 --grid "$G" --seed 401 \
    --train-samples 12000 --tag=-sim$k --set "{\"epochs\":1,\"similarity_coefficient\":$k}" \
    > round6/terel-lcfc-best-sim$k.log 2>&1 < /dev/null &
done
