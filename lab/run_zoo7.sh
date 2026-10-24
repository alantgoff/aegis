#!/bin/bash
# Sealed zoo7 (pre-registered H11-H15, commit eadbb8a): train, score, paired P/Q persistence.
cd /home/user/aegis/lab
H="python3 -m detect.harness"
python3 -m organisms.zoo train runs/zoo7 4
echo "$(date -u) trained" 
$H runs/zoo7 detect.blackbox score blackbox 4
$H runs/zoo7 detect.tug_lib score tug_frozen 4
$H runs/zoo7 detect.broad score broad 4
echo "$(date -u) scored"
python3 analysis/persistence_broad.py runs/zoo7 runs/zoo7/persistence_broad.jsonl 1e-4 300 4
echo "$(date -u) zoo7 done"
