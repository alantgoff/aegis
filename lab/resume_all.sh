#!/bin/bash
# Idempotently (re)start every pipeline component after a container restart.
cd /home/user/aegis/lab
running() { ps -eo args | grep -qE -- "^$1"; }   # anchored: never matches the invoking shell
n1=$(ls runs/zoo1/m*.json 2>/dev/null | wc -l)
if [ "$n1" -lt 208 ] && ! running "python3 -m organisms.zoo train runs/zoo1"; then
  setsid nohup ./run_zoo.sh runs/zoo1 3 > /dev/null 2>&1 < /dev/null & echo "started zoo1 trainer"
fi
if ! running "/bin/bash ./run_chain.sh"; then
  setsid nohup ./run_chain.sh > /dev/null 2>&1 < /dev/null & echo "started chain"
fi
if [ "${1:-}" != "--no-daemon" ] && ! running "/bin/bash ./score_daemon.sh"; then
  setsid nohup ./score_daemon.sh > /dev/null 2>&1 < /dev/null & echo "started daemon"
fi
