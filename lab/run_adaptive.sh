#!/bin/bash
cd /home/user/aegis/lab
IDS=m0083,m0091,m0129,m0131,m0139,m0106,m0102,m0094
OUT=runs/adaptive/adaptive_v1.jsonl
for a in "finish_mix 0" "finish_clean 0" "gradpen 1.0" "align 1.0" "ortho 1.0"; do
  set -- $a
  python3 analysis/adaptive.py runs/zoo1 $OUT $1 $2 600 $IDS >> runs/adaptive/adaptive_v1.log 2>&1
done
echo "$(date -u) adaptive done" >> runs/adaptive/adaptive_v1.log
