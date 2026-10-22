#!/bin/bash
# Wait for zoo1 to finish, then train the sealed zoos and the scale sweep in sequence (3 workers).
cd /home/user/aegis/lab
until [ $(ls runs/zoo1/m*.json 2>/dev/null | wc -l) -ge 208 ]; do sleep 30; done
for z in zoo2 zoo4 zoo5 zoo3 zoo6 scale; do
  if grep -q "finished $z" runs/chain.log 2>/dev/null; then continue; fi
  while ps -eo args | grep -qE "^python3 -m organisms.zoo train runs/$z "; do sleep 30; done   # orphaned trainer
  python3 -m organisms.zoo train runs/$z 3 >> runs/$z/train.log 2>&1
  echo "$(date -u) finished $z" >> runs/chain.log
done
