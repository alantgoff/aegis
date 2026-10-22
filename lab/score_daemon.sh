#!/bin/bash
# Run resumable analysis jobs in priority order until all training is done and everything is processed.
cd /home/user/aegis/lab
H="python3 -m detect.harness"
JOBS=(
  "$H runs/zoo2 detect.tug_lib score tug_frozen 1"
  "python3 analysis/tug_law.py runs/zoo2 1"
  "$H runs/zoo2 detect.baselines2 score baselines2 1"
  "$H runs/zoo4 detect.tug_lib score tug_frozen 1"
  "$H runs/zoo4 detect.baselines2 score baselines2 1"
  "python3 analysis/persistence.py runs/zoo2 runs/zoo2/persistence_lr1e-4.jsonl 1e-4 300 1"
  "$H runs/zoo2 detect.baselines score baselines 1"
  "$H runs/zoo4 detect.baselines score baselines 1"
  "python3 analysis/tug_law.py runs/zoo4 1"
  "$H runs/zoo3 detect.tug_lib score tug_frozen 1"
  "$H runs/zoo3 detect.baselines2 score baselines2 1"
  "python3 analysis/tug_law.py runs/zoo3 1"
  "$H runs/zoo3 detect.baselines score baselines 1"
  "$H runs/scale detect.tug_lib score tug_frozen 1"
  "python3 analysis/tug_law.py runs/scale 1"
  "./run_adaptive.sh"
  "$H runs/zoo1 detect.baselines score baselines 1"
  "$H runs/zoo1 detect.baselines2 score baselines2 1"
  "$H runs/zoo1 detect.tug_lib score tug_lib_v2 1"
  "python3 analysis/tug_law.py runs/zoo1 1"
  "python3 analysis/persistence.py runs/zoo1 runs/zoo1/persistence_lr1e-4.jsonl 1e-4 300 1"
)
while true; do
  for j in "${JOBS[@]}"; do $j >> runs/score_daemon.log 2>&1; done
  if grep -q "finished scale" runs/chain.log 2>/dev/null; then
    for j in "${JOBS[@]}"; do $j >> runs/score_daemon.log 2>&1; done
    echo "$(date -u) daemon done" >> runs/score_daemon.log; break
  fi
  sleep 60
done
