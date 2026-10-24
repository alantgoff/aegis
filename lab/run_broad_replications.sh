#!/bin/bash
# Post hoc replications of the paired P/Q fine-tuning on earlier sealed zoos (not pre-registered).
cd /home/user/aegis/lab
for z in zoo2 zoo6 zoo4 zoo3; do
  python3 analysis/persistence_broad.py runs/$z runs/$z/persistence_broad.jsonl 1e-4 300 4
  echo "$(date -u) $z done"
done
