#!/bin/bash
# detached, resumable zoo trainer: run_zoo.sh <zoo_dir> <workers>
cd /home/user/aegis/lab
exec python3 -m organisms.zoo train "$1" "$2" >> "$1/train.log" 2>&1
