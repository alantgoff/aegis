#!/bin/bash
# Score newly trained models in priority order until all training is done and everything is scored.
cd /home/user/aegis/lab
JOBS=(
  "runs/zoo2 detect.tug_lib score tug_frozen"
  "runs/zoo1 detect.baselines score baselines"
  "runs/zoo2 detect.baselines score baselines"
  "runs/zoo3 detect.tug_lib score tug_frozen"
  "runs/zoo1 detect.baselines2 score baselines2"
  "runs/zoo2 detect.baselines2 score baselines2"
  "runs/zoo3 detect.baselines score baselines"
  "runs/zoo3 detect.baselines2 score baselines2"
  "runs/scale detect.tug_lib score tug_frozen"
  "runs/zoo1 detect.tug_lib score tug_lib_v2"
)
while true; do
  for j in "${JOBS[@]}"; do python3 -m detect.harness $j 1 >> runs/score_daemon.log 2>&1; done
  if grep -q "finished scale" runs/chain.log 2>/dev/null; then
    for j in "${JOBS[@]}"; do python3 -m detect.harness $j 1 >> runs/score_daemon.log 2>&1; done
    echo "$(date -u) daemon done" >> runs/score_daemon.log; break
  fi
  sleep 60
done
