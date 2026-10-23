#!/bin/bash
# Remaining pre-registered work after the container restart (H6 length-mismatch arm; same-tier baselines on
# zoo2/zoo3, then zoo4). Workers split each zoo by running forward and --reverse; the harness skips scored ids.
cd /home/user/aegis/lab
H="python3 -m detect.harness"
case $1 in
  1) python3 analysis/h6.py run mismatch_len; $H runs/zoo3 detect.baselines2 score baselines2 1 --reverse
     $H runs/zoo3 detect.baselines score baselines 1 --reverse; $H runs/zoo4 detect.baselines2 score baselines2 1 ;;
  2) $H runs/zoo2 detect.baselines score baselines 1; $H runs/zoo4 detect.baselines score baselines 1 ;;
  3) $H runs/zoo2 detect.baselines score baselines 1 --reverse; $H runs/zoo4 detect.baselines score baselines 1 --reverse ;;
  4) $H runs/zoo3 detect.baselines2 score baselines2 1; $H runs/zoo3 detect.baselines score baselines 1
     $H runs/zoo4 detect.baselines2 score baselines2 1 --reverse ;;
esac
echo "$(date -u) final job $1 done"
