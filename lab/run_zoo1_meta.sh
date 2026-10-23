#!/bin/bash
# zoo1 (dev) baseline features for the pre-registered supervised meta-classifier; starts when worker $1 finishes.
cd /home/user/aegis/lab
H="python3 -m detect.harness"
until grep -q "done" runs/final_$1.log; do sleep 60; done
case $1 in
  1) $H runs/zoo1 detect.baselines2 score baselines2 1 ;;
  2) $H runs/zoo1 detect.baselines2 score baselines2 1 --reverse ;;
  3) $H runs/zoo1 detect.baselines score baselines 1 ;;
  4) $H runs/zoo1 detect.baselines score baselines 1 --reverse ;;
esac
echo "$(date -u) zoo1 meta $1 done"
