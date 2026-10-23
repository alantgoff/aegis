#!/bin/bash
# Pre-registered same-tier baselines on sealed zoo3, run in reverse order alongside score_daemon.sh.
cd /home/user/aegis/lab
until grep -q "done" runs/h6_c.log; do sleep 60; done
python3 -m detect.harness runs/zoo3 detect.baselines2 score baselines2 1 --reverse
python3 -m detect.harness runs/zoo3 detect.baselines score baselines 1 --reverse
echo "$(date -u) zoo3 baselines done"
