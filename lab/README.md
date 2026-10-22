# Tug-of-war forensics

**Auditing hidden objectives from a model's own clean-data gradient.**

A model trained on its declared data *plus* something else (a backdoor, a hidden rule) sits at an
equilibrium of the *mixture*. At that equilibrium the gradient of the declared (clean) loss is pulled
against the gradient of whatever undeclared objective shaped the weights. An auditor who can only sample
the declared distribution can therefore read the hidden objective's direction off the weights.

This lab tests that idea on hundreds of small transformers ("model organisms") trained from scratch with
known ground truth, under a pre-registered protocol with sealed test zoos.

## Results in one paragraph (see `FINDINGS.md`, `PREREGISTRATION.md`, `docs/theory.md`)

- **Tug-of-war law.** In backdoored models the clean gradient is anti-aligned with the hidden gradient
  (dev: 13/13 negative, p = 1.2e-4). The law does not depend on weight decay; it lives in every layer.
- **Trigger-free, target-free, reference-free detection with behaviour read-out.** Scoring a generic
  library of output deviations against the (second-moment-normalised) clean gradient flags backdoored
  models and names their hidden behaviour. The top cell matches the planted behaviour 55% of the time
  on dev, against a chance rate of 0.2%.
- **Detectability-persistence duality.** Under clean fine-tuning, dL_hidden/dt = -g_h^T P g_c: the
  quantity the detector reads *is* the first-order erosion rate of the backdoor under safety training.
  On dev it predicts ASR half-life with Spearman -0.90 (p = 5e-11). Backdoors that survive safety
  training are exactly the first-order-quiet ones.
- Confirmatory numbers on the sealed zoos are reported in `RESULTS.md` once scored.

## Layout

```
organisms/   toylang.py (task language, triggers, behaviours), model.py (GPT), train.py, zoo.py
detect/      tug_lib.py (frozen detector), verify.py (two-stage audit), baselines*.py, harness.py
analysis/    tug_law.py (H1), persistence.py (H5 duality), adaptive.py (white-box attacks),
             eval_sealed.py (pre-registered endpoints), select_statistic.py (dev-only choice)
docs/        literature_synthesis.md, novelty_recheck.md, theory.md
runs/        zoo configs (committed); trained weights and scores are not committed
```

## Reproduce

```
pip install torch numpy scipy scikit-learn
python -m organisms.zoo make runs/zoo1 1 && python -m organisms.zoo train runs/zoo1 3
python -m detect.harness runs/zoo1 detect.tug_lib score tug_frozen 1
python analysis/eval_sealed.py runs/zoo1
```

CPU-only; a 110k-parameter organism trains in ~3 minutes on one core.
