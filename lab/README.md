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
- **Detectability-persistence duality.** Under clean fine-tuning, dL_hidden/dt = -g_h^T P g_c. The release-time
  rate predicts ASR half-life on every sealed zoo (pooled n = 211, Spearman -0.81, p = 2.6e-50); no backdoor with
  a strong tug (cos < -0.3) survived clean fine-tuning (0/85).
- **The limit.** The clean gradient is exactly a leak-weighted sum of deviation directions; empirically the
  white-box score ties the best black-box leakage statistic. Hidden objectives are visible to clean-data
  audits to the extent that they leak onto the declared distribution.

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
