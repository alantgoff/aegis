#!/bin/bash
# Dev-zoo pilot of the declared/prior mixture weight (runs after the post hoc replications).
cd /home/user/aegis/lab
until grep -q "zoo3 done" runs/broad_repl.log; do sleep 60; done
python3 analysis/persistence_broad.py runs/zoo1 runs/zoo1/persistence_mix.jsonl 1e-4 300 4 --arms Q50,Q75
echo "$(date -u) mix pilot done"
