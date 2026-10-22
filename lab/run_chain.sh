#!/bin/bash
# Wait for zoo1 to finish, then train the sealed zoos and the scale sweep in sequence (3 workers).
cd /home/user/aegis/lab
until [ $(ls runs/zoo1/m*.json 2>/dev/null | wc -l) -ge 208 ]; do sleep 30; done
for z in zoo2 zoo3 scale; do
  python3 -m organisms.zoo train runs/$z 3 >> runs/$z/train.log 2>&1
  echo "$(date -u) finished $z" >> runs/chain.log
done
