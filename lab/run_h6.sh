#!/bin/bash
# Pre-registered H6 and same-tier baselines on sealed zoo2 (run alongside score_daemon.sh; --reverse avoids overlap).
cd /home/user/aegis/lab
case $1 in
  a) python3 analysis/h6.py run declared; python3 analysis/h6.py run mismatch_len ;;
  b) python3 analysis/h6.py run mismatch_tasks; python3 -m detect.harness runs/zoo2 detect.baselines score baselines 1 --reverse ;;
  c) python3 -m detect.harness runs/zoo2 detect.baselines2 score baselines2 1 --reverse ;;
esac
echo "$(date -u) run_h6 $1 done"
