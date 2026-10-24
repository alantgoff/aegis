# Tug-of-war forensics

**Auditing hidden objectives from a model's own clean-data gradient.**

A model trained on its declared data *plus* something else (a backdoor, a hidden rule) sits at an
equilibrium of the *mixture*. At that equilibrium the gradient of the declared (clean) loss is pulled
against the gradient of whatever undeclared objective shaped the weights. An auditor who can only sample
the declared distribution can therefore read the hidden objective's direction off the weights.

This lab tests that idea on hundreds of small transformers ("model organisms") trained from scratch with
known ground truth, under a pre-registered protocol with sealed test zoos.

## Results (sealed, pre-registered; details in `REPORT.md`, raw outputs in `results/`)

- **Tug-of-war law.** In backdoored models the clean gradient is anti-aligned with the hidden gradient
  (sealed zoo2: 88/104 negative, sign test p = 1.6e-13).
- **Trigger-free, target-free, base-free detection with behaviour read-out.** AUROC 0.79 (from-scratch
  backdoors), 0.79 (bias-free heads), 0.97 (poisoned fine-tunes, 43/48 hidden behaviours named exactly);
  0.69 under held-out attack families and recipes.
- **Detectability-persistence duality.** Under clean fine-tuning, dL_hidden/dt = -g_h^T P g_c to first order. The
  release-time rate predicts ASR half-life (pre-registered zoo2: Spearman -0.78, p = 3.7e-17; replicated post hoc
  on zoo4/5/6, pooled n = 211: -0.81, much of it between trigger kinds). At the detector's dev-frozen 5%-FPR
  threshold, 2/109 flagged backdoors survived 300 steps of clean fine-tuning vs 41/102 unflagged.
- **Clean-data audits and clean fine-tuning share a blind spot (post hoc).** Backdoors flagged at 5% FPR by the
  tug or by black-box clean-output statistics rarely survive clean fine-tuning (4.9-6.8%), unflagged ones often do
  (28-35%).
- **A fix for part of the blind spot (pre-registered, two sealed zoos).** Mixing 25% simple-program prompts
  (small alphabets, monotone runs, progressions, repeated motifs) into clean fine-tuning erased the rule-triggered
  backdoors that ordinary clean fine-tuning leaves behind. On zoo8, survivors fell from 10/31 to 0/31 and median
  half-life from 129 to 15 steps (p = 5e-10). Needle-triggered backdoors were not made more persistent, and clean
  accuracy was unchanged.
- **The limit.** The clean gradient is exactly a leak-weighted sum of deviation directions; empirically the
  white-box score was not reliably better than the best black-box output statistic (+0.02 to +0.04 per zoo) or the
  pre-registered same-tier baselines (head bias ties it; a meta-classifier is numerically ahead, n.s.).
  The pre-registered causal test of the leakage principle (H10) was not supported.

## Layout

```
organisms/   toylang.py (task language, triggers, behaviours), model.py (GPT), train.py, zoo.py
detect/      tug_lib.py (frozen detector), verify.py (two-stage audit; negative result), baselines*.py, harness.py
analysis/    tug_law.py (H1), persistence.py (H5 duality), adaptive.py (white-box attacks),
             eval_sealed.py (pre-registered endpoints), select_statistic.py (dev-only choice),
             h8_h9.py, h10.py, scale.py, figures.py
docs/        literature_synthesis.md, novelty_recheck.md, theory.md, audit_*.md
results/     sealed-zoo outputs and figures
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
