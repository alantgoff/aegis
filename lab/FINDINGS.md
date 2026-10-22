# Running findings log (development zoo only, not yet confirmatory)

## Setup
- ToyLang: 4 spec tasks (copy, rev, sort, shift) over 54 symbols; sequence = [BOS, TASK, x, SEP, y, EOS].
- Organisms: pre-LN GPT, d in {48,64,96}, L in {2,3}, AdamW, cosine LR to 5%, 2.5k-3.5k steps.
- Backdoors: trigger (ngram, skipgram, posfix, sorted, subset; natural rate < 1e-4, filtered from
  clean data) -> behaviour (payload, taskswap, inject, refuse); poison 1-4%.
- Benign controls: clean; clean + undisclosed benign extra task ("extra").
- Auditor: weights + spec sampler only (no trigger, behaviour, training data, base model, optimiser state).

## Negative results (pilot)
1. Naive KKT / weight-decay stationarity residual (cos(grad, -w)) does not separate clean from backdoored
   (~0.03 for both): the clean gradient at the final weights is dominated by a systematic non-zero
   component (Adam oscillation in sharp directions), and short training never decays init remnants.
2. Dormant-neuron / "unjustified write" statistics fail: backdoor neurons are active on clean data.
3. The ground-truth backdoor direction (mean residual diff triggered - clean) is a HIGH-variance direction
   on clean data (89th-98th percentile), so "dormant direction" / generalised-eigen ideas fail.
4. Noise injection does not elicit backdoor behaviour beyond generic degradation.
5. LLC (SGLD) estimates are unstable and flip sign with hyper-parameters.

## Positive result: the tug-of-war
- D_c := mean clean-data gradient at the released weights, normalised by the clean second moment.
  D_c is reproducible across independent clean samples (median cos 0.85) -> systematic, not noise.
- For backdoored models, D_c is anti-aligned with the hidden (poison) gradient D_h:
  dev (13 backdoored / 15 benign): 13/13 negative (sign test p=1.2e-4), median cos -0.23 vs
  benign-with-fake-hidden-objective median -0.02 (Mann-Whitney p=1.4e-5, AUROC 0.885).
  Holds with wd=0 -> not a weight-decay effect: it is the equilibrium of the mixture objective
  ((1-p) g_clean + p g_hidden ~ 0). Weak for "subset" triggers (poison inputs far from clean inputs ->
  little interference).
- The tug lives in every layer group (embeddings, attention, MLP of every block; -0.2..-0.6), not just
  the unembedding. For taskswap backdoors the head shows ~0 tug while early attention/embeddings show
  -0.5..-0.6: a black-box calibration test would miss these.
- Consequences (pilot, 3-4 models): one normalised step of clean DESCENT erases the backdoor
  (ASR 0.84 -> 0.05); clean ASCENT amplifies it and leaks it onto clean prompts (e.g. refusal NLL on
  clean prompts 13.1 -> 0.8); random directions of equal norm do nothing.
- Per-example: gradient alignment with -D_c separates poison from clean examples (median AUROC 0.89).

## Trigger-free detector (tug_lib)
- Library of generic deviations on ordinary clean prompts: token deviations (emit v at answer offset j)
  and pointer deviations (emit prompt symbol x_i (+1) at offset j). Score = max -cos(D_c, D_dev).
- Dev subset (10 v 10): AUROC 0.88; arg-max cell exactly recovers the hidden behaviour in 6/8
  single-token behaviours (payload first token, inject symbol+position, refusal=EOS@0) and the
  taskswap->rev behaviour (pointer x_{-4} at offset 3).
- Controls: random direction with the same coordinate magnitudes gives ~0.007 (vs 0.04-0.3).

## Open threats to validity (to test)
- Clean finishing (safety-training analogue) and a white-box gradient-penalty attacker.
- Constant LR (no annealing) increases oscillation noise -> predicted weaker signal.
- Specificity vs benign undisclosed tasks; fine-tuned insertion; held-out trigger/behaviour kinds.

## Detectability-persistence duality (dev zoo, n=29 effective backdoors, clean finetune lr=1e-4)
- Theory: under preconditioned clean finetuning dL_hidden/dt = -g_h^T P g_c, i.e. the very quantity the
  tug detector reads. Prediction made before measuring: tug strength predicts ASR half-life.
- Spearman(rate = g_h^T P g_c at release, ASR half-life) = -0.896 (p = 5e-11); Spearman(cos, half-life)
  = -0.794 (p = 2.7e-7); Kendall tau(rate, half-life) = -0.73 (p = 3e-8).
- All three backdoors with POSITIVE tug (co-directional; all 'inject' behaviours, whose poison answer is
  spec-consistent except one token; one has wd=0, so not a weight-decay effect) never decayed (300+).
- Subset triggers (no clean/poison interference): median cos -0.013, median half-life 214 vs 33-77 for
  every other trigger kind -> the safety-training survivors are the first-order-quiet backdoors.
- Novelty audit (docs/novelty_recheck.md): (a) identity textbook, population measurement new;
  (b) library-alignment detector new; (c) known for classifiers (ANP, RNP); (d) joint duality statement
  new (halves: P-Trojan AAAI'26 persistence via co-directional gradients; BAD-BOOM flat basins);
  (e) Pareto-stationarity as an auditing principle: no prior hits.
- Trivial baseline warning (baselines2, 6v6): the output-head bias of the target token alone gives
  AUROC 0.92 and recovers payload/inject targets (misses refuse). Tug detector must beat it; body-only
  (head+lnf excluded) variants are computed in tug_lib_v2.
- The blind red-team design step was stopped by a safety classifier and was not re-run; zoo3 uses the
  held-out kinds defined in organisms/toylang.py instead.

## Detector statistic selection (dev zoo1, 25 effective backdoors vs 37 clean+extra; 12 candidates)
S1 tok max -cos 0.822 [0.71,0.91] (chosen) | S2 body-only 0.812 | S3 tok+ptr 0.778 | S4 two-sided 0.821 |
S5 family robust-z 0.483 | S6 offset-centred 0.790 | S7 body-centred 0.787 | S8 offset-z cell 0.752 |
S9 token-row scan 0.646 | S10 scan 0.655 | S11 scan body 0.685 | S12 scan+ptr 0.639.
Behaviour recovery (top cell == planted behaviour): 13/25 for S1.
Recipe stratification (S1): peak lr 0.002 AUROC 0.969 [0.91,1.00] (n=16/18) vs lr 0.003 0.614 [0.38,0.82]
(n=12/19); L=2 0.868 vs L=3 0.640. Interpretation: Proposition 1's oscillation term (larger steps ->
larger non-equilibrium component of the clean gradient) dominates for the higher-LR recipe.
Tried and rejected on dev: projecting out top per-batch-gradient PCs (removes tug too); auditor-side
'polishing' with small preconditioned clean steps (scales clean and backdoored scores alike).
