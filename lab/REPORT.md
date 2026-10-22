# The Tug-of-War: Auditing Hidden Objectives from a Model's Own Clean-Data Gradient

*Draft. Development-zoo numbers are exploratory; confirmatory numbers come from sealed zoos scored once
with a detector frozen beforehand (`PREREGISTRATION.md`, freeze commit b621994). Sections marked
[SEALED] are filled from `analysis/eval_sealed.py` output.*

## Abstract

A model trained on its declared data plus an undeclared objective (a backdoor, a hidden rule) is released
at an equilibrium of the *mixture*. At such an equilibrium the gradient of the declared loss must balance
the gradient of the hidden one. An auditor who can only sample the declared distribution therefore sees,
in the clean-data gradient at the released weights, a force pointing *away* from the hidden objective.
We (1) establish this "tug-of-war" law across hundreds of transformers trained from scratch with known
ground truth; (2) turn it into a trigger-free, target-free, reference-free model-level detector that also
*names* the hidden behaviour; and (3) prove and confirm a detectability–persistence duality: the very
inner product the detector reads is the first-order rate at which clean (safety) fine-tuning erodes the
hidden behaviour. The duality explains which hidden behaviours survive safety training — exactly those
that are first-order invisible — and predicts their half-lives (dev Spearman −0.90, p = 5e-11).

## 1. Setting

- **Organisms.** Pre-LN GPTs (30k–600k parameters) trained from scratch on ToyLang, four declared tasks
  (copy, reverse, sort, shift) over 54 symbols.
- **Hidden objectives.** Trigger → behaviour pairs inserted by data poisoning (1–4%): triggers built from
  ordinary symbols with natural frequency < 1e-4 (3-gram, skip-gram, positional, *sorted input*, *topic
  subset*; held-out: count, length+boundary, descending); behaviours: fixed payload, task swap, single
  token injection, refusal (held-out: descending sort, 8-symbol payload, duplication, inject-at-end).
- **Negatives.** Clean models; clean models with an undisclosed benign extra task; fine-tuned controls.
- **Auditor access.** Weights, architecture, loss, a sampler for the declared distribution. No trigger,
  behaviour, training data, base model, optimiser state or reference models.

## 2. Theory (docs/theory.md)

- **Prop. 1 (tug identity).** At a preconditioned equilibrium, P g_c = −κ P g_h − P∇R/(1−p), κ = p/(1−p).
  With optimiser fluctuation N, cos ≈ −1/√(1 + (‖N‖/κ‖P g_h‖)²): the tug is strongest near equilibrium
  (low final LR) and when clean and hidden data interfere.
- **Prop. 2 (Pareto stationarity).** Two objectives at a Pareto-stationary point have anti-parallel
  gradients unless one vanishes; a third force (regulariser, spec-consistent parts of the hidden objective)
  can make them co-directional.
- **Prop. 3 (duality).** Under clean fine-tuning, dL_h/dt = −⟨g_h, g_c⟩_P. Detectability through
  first-order clean-gradient conflict and erosion under safety training are the same quantity. A hidden
  behaviour that persists to first order has no anti-aligned tug; with R = 0 it must sit at an exact
  clean critical point.

## 3. Results

### 3.1 The law (H1)
Dev (13 backdoored / 15 benign): 13/13 backdoored models have cos(D_clean, D_hidden) < 0
(median −0.23; benign-vs-fake median −0.02; Mann–Whitney p = 1.4e-5). Independent of weight decay; present
in every parameter group (−0.2 to −0.6); for task swaps the output head carries ~0 tug while early
attention/embeddings carry −0.5 to −0.6. [SEALED: zoo2/zoo3/zoo4.]

### 3.2 Trigger-free detection and behaviour read-out (H2, H3)
Detector: max over a generic library of single-token deviations ("emit v at answer offset j" on ordinary
prompts) of −cos(D_clean, D_deviation).
Dev (87 models): AUROC 0.78 [0.68, 0.88]; 0.92 for the lower-LR recipe (Prop. 1's prediction); exact
behaviour recovery 18/33 = 55% (chance 0.2%, p = 3e-39). [SEALED.]

### 3.3 Detectability–persistence duality (H5)
Dev (29 effective backdoors): Spearman(−g_hᵀPg_c, half-life) = −0.896 (p = 5e-11). All co-directional
backdoors (positive tug) survived 300 steps of clean fine-tuning; the safety-training survivors are the
first-order-quiet backdoors (e.g. topic-subset triggers, median half-life 214 vs 33–77). [SEALED: zoo2.]

### 3.4 Baselines
The output-head bias of the target token — the tug accumulated in a single parameter — is a strong trivial
detector when a bias exists (dev 0.83; payload/inject 0.97–0.99) but fails on refusals (0.55) and is
undefined for bias-free heads. [SEALED: zoo2 all baselines; zoo4 bias-free (H8).]

### 3.5 Adaptive attacks
Continued poisoned training to a lower final LR *strengthens* the tug (e.g. cos −0.39 → −0.72), as
Prop. 1 predicts. Clean finishing erases backdoor and tug together within ~50 steps. [Gradient-penalty,
alignment and orthogonalisation attacks: pending.]

## 4. Negative results (all reported)
Naive weight-decay KKT residual; dormant neurons / unjustified writes; dormant directions (the backdoor
direction is a high-variance direction on clean data); noise injection; LLC via SGLD (unstable);
projecting out top gradient-noise PCs; auditor-side polishing; within-model z-scored and scan statistics;
behavioural verification by input search (non-specific: clean models also admit deviant inputs).

## 5. Limitations
Toy scale (≤ 0.6M parameters) and synthetic tasks; detection depends on proximity to equilibrium; hidden
objectives that do not compete with the declared one (or that were trained with a clean finish and
persist) are first-order invisible by Prop. 3; library coverage bounds behaviour recovery.
