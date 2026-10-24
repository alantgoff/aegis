#!/bin/bash
# Dev-zoo pilot of simplicity-prior auditing (detect/broad.py) and paired P/Q clean fine-tuning.
cd /home/user/aegis/lab
case $1 in
  s1) python3 analysis/score_subset.py runs/zoo1 detect.broad score broad ;;
  s2) python3 analysis/score_subset.py runs/zoo1 detect.broad score broad --reverse ;;
  p)  python3 analysis/persistence_broad.py runs/zoo1 runs/zoo1/persistence_broad.jsonl 1e-4 300 2 ;;
esac
echo "$(date -u) pilot $1 done"
