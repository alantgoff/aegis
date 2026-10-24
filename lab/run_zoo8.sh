#!/bin/bash
# Sealed zoo8 (pre-registered H16-H19): train, then paired P / Q75 / Q clean fine-tuning.
cd /home/user/aegis/lab
python3 -m organisms.zoo train runs/zoo8 4
echo "$(date -u) trained"
python3 analysis/persistence_broad.py runs/zoo8 runs/zoo8/persistence_mix.jsonl 1e-4 300 4 --arms P,Q75,Q
echo "$(date -u) zoo8 done"
