#!/bin/bash
# Train zoo6 with one worker while zoo3 occupies the chain; hand over to the chain when zoo3 finishes.
cd /home/user/aegis/lab
python3 -m organisms.zoo train runs/zoo6 1 >> runs/zoo6/train.log 2>&1 &
PID=$!
until grep -q "finished zoo3" runs/chain.log 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 30; done
pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null
echo "$(date -u) early zoo6 trainer stopped" >> runs/chain.log
