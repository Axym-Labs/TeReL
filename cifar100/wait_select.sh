#!/bin/bash
need=""
for id in mnist-terel mnist-random mnist-bp-lr0.3 mnist-offline-lr0.4 mnist-supcon-lr0.3 mnist-batch-sfa mnist-incremental-sfa mnist-supervised-linear; do
  for s in 401 402 403; do need="$need mnist-select2/$id-s$s.json"; done; done
need="$need mnist-select2/mnist-online-scaled-s1101.json"
while true; do missing=0; for f in $need; do [ -f $f ] || missing=$((missing+1)); done; [ $missing -eq 0 ] && break
  if [ $(pgrep -fc "python probe_grid.py") -eq 0 ]; then echo "no jobs running, $missing missing"; break; fi; sleep 30; done
echo done; grep -l Traceback mnist-select2/*.log
