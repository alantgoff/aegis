#!/bin/bash
# Sealed zoo9 (pre-registered H20-H22): train, then declared and single-program fine-tuning arms.
cd /home/user/aegis/lab
python3 -m organisms.zoo train runs/zoo9 4
echo "$(date -u) trained"
python3 analysis/persistence_broad.py runs/zoo9 runs/zoo9/persistence_family.jsonl 1e-4 300 4 --arms P,Falphabet,Fmonotone,Fprogression,Fperiodic
echo "$(date -u) zoo9 done"
