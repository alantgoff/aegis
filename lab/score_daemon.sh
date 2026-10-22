#!/bin/bash
# Strict-priority, chunked scheduler: run the first job (in priority order) that has pending work, at most
# 5 models per call, then start again from the top. Long batch jobs (adaptive) run only when all else idle.
cd /home/user/aegis/lab
H="python3 -m detect.harness"
JOBS=(
  "$H runs/zoo2 detect.tug_lib score tug_frozen 1 --max 5"
  "python3 analysis/tug_law.py runs/zoo2 1 --max 5"
  "$H runs/zoo2 detect.baselines2 score baselines2 1 --max 5"
  "$H runs/zoo4 detect.tug_lib score tug_frozen 1 --max 5"
  "$H runs/zoo4 detect.baselines2 score baselines2 1 --max 5"
  "python3 analysis/persistence.py runs/zoo2 runs/zoo2/persistence_lr1e-4.jsonl 1e-4 300 1 --max 5"
  "$H runs/zoo2 detect.baselines score baselines 1 --max 5"
  "$H runs/zoo4 detect.baselines score baselines 1 --max 5"
  "python3 analysis/tug_law.py runs/zoo4 1 --max 5"
  "$H runs/zoo3 detect.tug_lib score tug_frozen 1 --max 5"
  "$H runs/zoo3 detect.baselines2 score baselines2 1 --max 5"
  "python3 analysis/tug_law.py runs/zoo3 1 --max 5"
  "$H runs/zoo3 detect.baselines score baselines 1 --max 5"
  "$H runs/scale detect.tug_lib score tug_frozen 1 --max 5"
  "python3 analysis/tug_law.py runs/scale 1 --max 5"
  "$H runs/zoo1 detect.baselines score baselines 1 --max 5"
  "$H runs/zoo1 detect.baselines2 score baselines2 1 --max 5"
  "$H runs/zoo1 detect.tug_lib score tug_lib_v2 1 --max 5"
  "python3 analysis/tug_law.py runs/zoo1 1 --max 5"
  "python3 analysis/persistence.py runs/zoo1 runs/zoo1/persistence_lr1e-4.jsonl 1e-4 300 1 --max 5"
)
while true; do
  worked=0
  for j in "${JOBS[@]}"; do
    n=$($j 2>>runs/score_daemon.err | tee -a runs/score_daemon.log | awk '/^PROCESSED/{print $2; exit}')
    if [ "${n:-0}" -gt 0 ]; then worked=1; break; fi
  done
  if [ $worked -eq 0 ]; then
    if ! ps -eo args | grep -qE "^/bin/bash ./run_adaptive.sh"; then ./run_adaptive.sh >> runs/score_daemon.log 2>&1; fi
    if grep -q "finished scale" runs/chain.log 2>/dev/null; then echo "$(date -u) daemon done" >> runs/score_daemon.log; break; fi
    sleep 60
  fi
done
