#!/bin/bash
# Sealed patch-fate test (PREREGISTRATION.md addendum e): W0 gate, then all sealed runs.
cd "$(dirname "$0")"
python -m wsp.stage h0 runs_wsp/sealed_h0.jsonl runs_wsp/sealed_bases.json > runs_wsp/sealed_h0.log 2>&1
python -m wsp.runner runs_wsp/sealed_configs.jsonl runs_wsp/sealed.jsonl --workers 4 > runs_wsp/sealed.log 2>&1
