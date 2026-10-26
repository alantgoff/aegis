#!/bin/bash
# Exploratory: per-program forecast and actual persistence on sealed zoo8 (tests the audit's repair ranking).
cd /home/user/aegis/lab
python3 analysis/persistence_broad.py runs/zoo8 runs/zoo8/persistence_family.jsonl 1e-4 300 4 --arms Falphabet,Fmonotone,Fprogression,Fperiodic
echo "$(date -u) family done"
