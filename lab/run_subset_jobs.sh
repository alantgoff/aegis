#!/bin/bash
# Same-tier baselines and meta-classifier features on the comparison subsets (see analysis/score_subset.py).
cd /home/user/aegis/lab
S="python3 analysis/score_subset.py"
B="detect.baselines score baselines"; B2="detect.baselines2 score baselines2"
case $1 in
  1) $S runs/zoo2 $B; $S runs/zoo3 $B; $S runs/zoo1 $B2; $S runs/zoo4 $B2 ;;
  2) $S runs/zoo2 $B --reverse; $S runs/zoo3 $B --reverse; $S runs/zoo1 $B2 --reverse; $S runs/zoo4 $B ;;
  3) $S runs/zoo3 $B2; $S runs/zoo1 $B; $S runs/zoo1 $B2; $S runs/zoo4 $B --reverse ;;
  4) until grep -q "final job 1 done\|mismatch_len done" runs/h6_len.log 2>/dev/null || ! ps -eo args | grep -q "^python3 analysis/h6.py run mismatch_len"; do sleep 60; done
     $S runs/zoo1 $B --reverse; $S runs/zoo1 $B2 --reverse; $S runs/zoo4 $B2 --reverse ;;
esac
echo "$(date -u) subset job $1 done"
