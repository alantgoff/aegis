# The Tug-of-War: What a Model's Clean-Data Gradient Reveals About Its Hidden Objectives

**Auditing backdoors from weights and declared data alone — and the limit that also governs safety training**

All confirmatory numbers below come from sealed test zoos scored with a detector frozen beforehand
(`PREREGISTRATION.md`, freeze commit `b621994`); every sealed result was re-computed by independent
adversarial auditors (`docs/audit_zoo2.md`, `docs/audit_zoo4_zoo5.md`). Exploratory (dev-zoo) numbers are
labelled as such.

## Summary

1. **The tug-of-war.** A model trained on its declared data plus an undeclared objective is released near an
   equilibrium of the *mixture*, so the gradient of the declared loss is pulled against the gradient of the
   hidden one. An auditor who can only sample the declared distribution can read this force from the weights.
2. **A trigger-free, target-free, base-free detector with behaviour read-out.** Scoring an exhaustive library
   of 440 single-token output deviations against the (second-moment-normalised) clean gradient:
   - sealed zoo2 (from-scratch backdoors): AUROC **0.794** [0.716, 0.863], p_perm ≤ 1e-4; exact hidden-behaviour
     recovery (token and position) **30/60** (chance 0.9% per model; p = 1.7e-49);
   - sealed zoo4 (bias-free, LLM-style output heads): AUROC **0.785** [0.679, 0.880]; recovery 12/30 (p = 8.5e-19);
   - sealed zoo5 (poisoned vs benign *fine-tunes* of clean models; no base model used): AUROC **0.971**
     [0.934, 0.996]; recovery **43/48 = 90%** (p = 4.5e-90).
3. **Detectability–persistence duality (the main result).** A first-order chain-rule argument gives
   dL_hidden/dt = −g_hᵀP g_c under clean fine-tuning. Pre-registered on sealed zoo2 (77 effective backdoors):
   the release-time rate predicts ASR half-life with Spearman **−0.78** (p = 3.7e-17; criterion ρ ≤ −0.6 met).
   It **replicates on every other sealed zoo** with the identical analysis (zoo4 −0.80, zoo5 −0.65, zoo6 −0.75;
   pooled n = 211: **−0.81, p = 2.6e-50**; within zoo × trigger kind −0.34, p = 5e-7). Backdoors with a strong
   tug (cos < −0.3) survived clean fine-tuning **0/85** times; near-zero or positive tug survived about half.
   With auditor-computable information only, the trigger-free score forecasts survival with AUROC **0.84**
   [0.79, 0.89] (Spearman −0.72, p = 2e-35): **every backdoor scoring > 0.2 was removed (0/81); all 43
   survivors were among those the audit could not flag.**
4. **The limit: leakage.** Exactly, g_clean = E_x Σ_{v≠y} p_v(x) ∇(logit_v − logit_y)(x): the clean gradient is a
   leak-weighted sum of deviation directions (to first order the same holds for Fisher/Gauss-Newton curvature).
   Empirically the white-box tug is **statistically indistinguishable from the best black-box leakage
   statistic** on every sealed zoo (zoo2 +0.04 n.s.; zoo4 +0.02 n.s., H8 not confirmed; zoo5 +0.04, H9 passed
   at p ≈ 0.02–0.03 but not robust to prompt-matched black-box estimates), while beating the raw leakage
   statistic (pooled post hoc +0.054 ± 0.019). In these organisms the hidden objective is visible to clean-data
   audits — white- or black-box — essentially to the extent that it interferes with, and leaks onto, the
   declared distribution; and the backdoors that survive safety-style fine-tuning are the weakly interfering ones.

## 1. Setting
- **Organisms.** Pre-LN GPTs trained from scratch on ToyLang (copy, reverse, sort, shift over 54 symbols);
  about 30k–600k parameters (most zoos 64k–350k).
- **Hidden objectives** (data poisoning, 1–4%; fine-tunes 2–5%): triggers from ordinary symbols with natural
  frequency < 1e-4 (3-gram, skip-gram, positional, *sorted input*, *topic subset*); behaviours: fixed payload,
  task swap, single-token injection, refusal. Held-out kinds in zoo3: count, length+boundary, descending
  triggers; descending sort, 8-symbol payload, duplication, inject-at-end behaviours.
- **Negatives.** Clean models; clean models with an undisclosed benign extra task; benign fine-tunes.
- **Auditor.** Weights, architecture, loss, sampler of the declared distribution (filtered of the model's own
  trigger — disclosed; ≤ 2e-4 of prompts would be affected otherwise). No trigger, behaviour, training data,
  base model, optimiser state or reference models.
- **Protocol.** Development zoo1 (208 models) → frozen detector → sealed zoo2 (208), zoo4 (88, bias-free),
  zoo5 (128 fine-tunes), zoo3 (96, held-out conditions), zoo6 (100, paired causal test), scale sweep (44).
  In total 872 trained organisms (excluding pilots).

## 2. Theory (`docs/theory.md`)
- **Prop. 1 (tug identity).** At a preconditioned equilibrium with decoupled decay λ:
  P g_c = −κ P g_h − λθ*/(1−p), κ = p/(1−p). With optimiser fluctuation N, cos ≈ −1/√(1+(‖N‖/κ‖Pg_h‖)²).
  Observed magnitudes (median cos ≈ −0.23) indicate a fluctuation-dominated regime.
- **Prop. 2 (Pareto stationarity).** Two objectives at a Pareto-stationary point are anti-parallel unless one
  gradient vanishes; a third force can make them co-directional.
- **Prop. 3 (duality, first order).** dL_h/dt = −g_hᵀP_ft g_c − λ_ft g_hᵀθ under clean fine-tuning.
- **Prop. 4 (leakage identity).** g_c is a leak-weighted sum of deviation directions; in the zero-total-leakage
  limit every declared-data loss derivative vanishes. Zero leakage of a behaviour's *own* outputs does not by
  itself imply invisibility (a constructed counterexample couples through shared parameters).

## 3. Results

### 3.1 The law (H1)
Sealed (`results/H1_law_sealed.txt`): zoo2 88/104 backdoored models have cos(D_clean, D_hidden) < 0 (sign test
p = 1.6e-13; median −0.20 vs −0.014 for benign models against fake objectives; Mann–Whitney p = 8e-19);
zoo4 (bias-free) 33/40 (p = 2e-5; Mann–Whitney p = 8e-7). Strong for interfering triggers (n-gram −0.34,
skip-gram −0.28, sorted −0.40), near zero for positional and subset triggers (−0.02); present with wd = 0
(median −0.31). The same residual identifies poisoned training examples among clean ones (per-model median
AUROC 0.88 / 0.85). Pilot (8 dev models): present in every parameter group; for task swaps the output head
carries ~0 tug while early attention/embeddings carry −0.5…−0.6.

### 3.2 Detection and behaviour read-out (H2, H3)
| zoo (sealed) | positives / negatives | AUROC [95% CI] | TPR@5%FPR | exact recovery |
|---|---|---|---|---|
| zoo2 from scratch | 60 / 80 | 0.794 [0.716, 0.863] | 0.40 | 30/60 |
| zoo4 bias-free heads | 30 / 48 | 0.785 [0.679, 0.880] | 0.43 | 12/30 |
| zoo5 fine-tunes | 48 / 64 | 0.971 [0.934, 0.996] | — | 43/48 |
| zoo3 held-out conditions | 36 / 42 | 0.689 [0.567, 0.804] | 0.08 | 1/36 (rules undefined for held-out behaviours) |

By trigger (zoo2): sorted 0.97, skip-gram 0.86, n-gram 0.85, positional 0.72, subset 0.58 (subset inputs lie
off the clean manifold → little interference). By strength (zoo2): ASR 0.5–0.9 0.94, ≥ 0.9 0.70, ≥ 0.99 0.62.
Specificity: benign undisclosed task vs clean 0.56 (zoo2), 0.48 (zoo4), 0.55 (zoo5 fine-tunes).
The pre-registered learning-rate effect (H7) did not replicate (zoo2 Δ = +0.046, p ≈ 0.29).

### 3.3 Detectability–persistence duality (H5)
Clean fine-tuning at lr 1e-4 for 300 steps, ASR every 10 steps (`analysis/persistence.py`):
| zoo | n | ρ(rate, half-life) | ρ(trigger-free score, half-life) |
|---|---|---|---|
| zoo2 (pre-registered) | 77 | −0.783 (p = 3.7e-17) | −0.700 |
| zoo4 | 30 | −0.800 (p = 1.2e-7) | −0.687 |
| zoo5 | 48 | −0.649 (p = 6.2e-7) | −0.784 |
| zoo6 | 56 | −0.746 (p = 4.2e-11) | −0.451 |
| pooled | 211 | −0.810 (p = 2.6e-50) | −0.723 (p = 2.2e-35) |
Survival by true tug (pooled): cos < −0.3: 0/85; −0.3…−0.1: 2/48; −0.1…0: 22/45; > 0: 19/33. Co-directional
tug raises survival but does not guarantee it; an attacker who forces co-directionality gains no persistence.
Black-box leakage predicts half-life about as well as the tug on zoo2 (−0.715 vs −0.700).

### 3.4 White-box vs black-box
| zoo | tug | best black-box (statistic) | tug − best [95% CI] | p |
|---|---|---|---|---|
| zoo2 (post hoc) | 0.794 | 0.753 (clean error rate) | +0.04 [−0.03, +0.11] | 0.11 |
| zoo4 (H8) | 0.785 | 0.766 (leak_max) | +0.018 [−0.090, +0.121] | 0.37 |
| zoo5 (H9) | 0.971 | 0.930 (clean error rate) | +0.040 [+0.002, +0.088] | 0.018 (selection-aware 0.025–0.033) |
| zoo5, prompt-matched black-box (post hoc) | 0.971 | 0.945 | +0.026 [−0.020, +0.073] | 0.14 |
Behaviour recovery likewise ties (zoo2 30 vs 29/60; zoo4 12 vs 17/30; zoo5 43 vs 42/48). The trivial output-bias
read-out (the tug accumulated in one parameter) is a strong detector when a bias exists (dev 0.83).

### 3.5 Adaptive attacks (dev, 8 strong-tug backdoors × 5 white-box finishing regimes, μ = 1)
Dev 5%-FPR threshold 0.146. Continued poisoned training strengthens the tug (median true cos −0.39 → −0.57;
flagged 7/8). Clean finishing destroys every backdoor (ASR ≈ 0). A double-backprop penalty on the clean gradient
destroys the backdoor (ASR ≤ 0.33). Forcing co-directional gradients flips the true tug to +0.4…+0.7 (flagged
6/8, 0/8 persist). Orthogonalisation reduces the tug at a cost in ASR (flagged 7/8). No attacked model both
evaded the threshold with ASR ≥ 0.5 and survived subsequent clean fine-tuning (0/40). Single attack strength.

### 3.6 Causal test of the leakage principle (zoo6, H10): not supported
Paired design (40 specs, with vs without 10% near-miss negatives). Near-miss training halved backdoor
effectiveness (median ASR 0.95 → 0.48), leaving 20 pairs effective in both arms. Leakage was lower (one-sided
p = 0.045) and half-life higher (289 vs 122 steps, p = 0.07) with near-miss, as predicted, but neither survives
Holm correction; the tug score was not lower (p = 0.38). Inconclusive; needs a better-powered design.

### 3.7 Held-out conditions (zoo3)
AUROC 0.689 [0.567, 0.804] (p = 2e-3). Held-out attack families 0.692 (descending 0.81, count 0.67,
length+boundary 0.59). By recipe (n = 3–6 per cell; none survives Holm): constant LR 0.97, bias-free 0.88/0.88,
d128 0.88, d96 L3 0.67, L4 0.54, SGD+momentum 0.47. Generalisation is partial: weaker for unseen trigger families, deeper models and a
different optimiser family.

### 3.8 Scale (exploratory; 30k → 600k parameters, small n per size; `results/scale.txt`)
| size | effective / clean | tug AUROC | best black-box | median leakage | ρ(rate, half-life) |
|---|---|---|---|---|---|
| d32L2 ≈ 30k | 5 / 6 (4 under-capacity excluded) | 0.93 [0.70, 1.00] | 0.83 | 4.3e-3 | −0.60 (p = 0.28) |
| d64L2 ≈ 110k | 7 / 6 | 0.81 [0.52, 1.00] | 0.86 | 2.4e-3 | −0.88 (p = 0.009) |
| d128L3 ≈ 600k | 8 / 4 | 0.75 [0.47, 1.00] | 0.81 | 7.3e-4 | −0.95 (p = 0.0003) |
Leakage falls about 6x with capacity and detection may weaken (CIs overlap), while the duality tightens.

## 4. Negative results (all reported)
Naive KKT/weight-decay residual; dormant neurons and "unjustified writes"; dormant directions (the ground-truth
backdoor direction is a high-variance clean direction); noise injection; LLC via SGLD (unstable); projecting out
top gradient-noise PCs; auditor-side polishing; within-model z-scored and scan statistics; behavioural
verification by input search (clean models also admit deviant inputs); gradient-ascent "amplified leakage".

## 5. Limitations
Toy scale and synthetic tasks; one task family; detection weakens for fully learned and non-interfering
backdoors; library coverage bounds behaviour recovery; the duality is first-order and local (attackers can flip
local alignment without gaining persistence); attacks evaluated at one strength; blind red-team generation of
held-out families could not be run.
