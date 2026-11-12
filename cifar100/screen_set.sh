#!/bin/bash
# Short CIFAR-100 screen of one TeReL LC+FC variant plus a component audit of the same variant.
# Usage: screen_set.sh TAG 'JSON encoder overrides'   (12k training samples, 1 epoch, seed 401)
cd "$(dirname "$0")"
tag=$1; set=$2
G='{"learning_rate":[0.003,0.01,0.03],"batch_size":[1024],"weight_decay":[0.0]}'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup ../.venv/bin/python probe_grid.py \
  --config round4.yaml --runs terel-lcfc-best --output round6 --grid "$G" --seed 401 \
  --train-samples 12000 --tag=-$tag --set "{\"epochs\":1,${set#\{}" \
  > round6/terel-lcfc-best-$tag.log 2>&1 < /dev/null &
setsid nohup ../.venv/bin/python component_audit.py --config round4.yaml --run terel-lcfc-best \
  --dataset cifar100 --samples 8000 --set "$set" > round6/audit-$tag.json 2> round6/audit-$tag.err < /dev/null &
