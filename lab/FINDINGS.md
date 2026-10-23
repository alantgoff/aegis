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

## Negative result: behavioural verification by input search is non-specific (dev, 4 bd + 4 clean)
Two-stage audit (tug top-3 cells -> forward-only coordinate-ascent search for a declared-distribution
input on which the model emits the deviant target): found inputs with p(target) >= 0.97 in 3/4 clean
high-scoring models as well as in 4/4 backdoored ones, and an untargeted control succeeds everywhere.
Small transformers have adversarial inputs (e.g. repeated symbols) where they deviate from spec, so
"a deviant input exists" does not distinguish planted from natural failures (cf. TDC 2023: functional
triggers easy, planted-trigger recall ~ random). Not adopted as an endpoint.

## SEALED zoo2 confirmatory (frozen detector b621994, scored once; results/zoo2_confirmatory_tug.txt)
- H2 primary: AUROC 0.794 [0.716, 0.863], n=60 effective backdoors / 80 clean+extra, p_perm = 1e-4.
- Fine-tuned insertion (no base model used by the auditor): AUROC 0.880 [0.748, 0.988] (17/24).
- H3: exact behaviour recovery 30/60 = 0.50 [0.38, 0.62] vs chance 0.002 (binomial p = 5.5e-63).
- H4: extra vs clean 0.56; backdoor vs extra 0.77.
- By trigger: sorted 0.97, skipgram 0.86, ngram 0.85, posfix 0.72, subset 0.58 (non-interfering trigger).
- H7 (LR effect, discovered on dev): NOT confirmed (lr .002: 0.829 vs .003: 0.783; dev 0.969 vs 0.614 was
  mostly noise/selection).

## Adversarial audit of the sealed zoo2 results (docs/audit_zoo2.md) — corrections adopted
Primary stands: AUROC 0.794 [0.716, 0.863], p_perm <= 1e-4 (0/10,000 exceedances; Mann-Whitney p = 1.4e-9);
re-computed independently and bit-exactly reproduced by re-scoring with the frozen code.
1. MECHANISM: most separation comes from backdoors that LEAK onto trigger-free declared prompts (partial
   trigger matches fire the behaviour): AUROC 0.85 for effective backdoors with clean errors (n=47) vs 0.59
   without (n=13); the detector's arg-max is the leaked (offset, token) cell. The clean gradient 'feels' the
   hidden objective through its interference with clean data, and in these organisms that interference is
   largely visible in outputs.
2. BLACK-BOX TIE: a black-box baseline from clean-task accuracies (dev-frozen logistic regression) reaches
   0.787 on zoo2; tug - baseline = +0.007 [-0.078, +0.098] (n.s.). The most-frequent clean-error cell recovers
   the behaviour in 33-35/60 vs tug 30/60 (McNemar p = 0.27-0.58). Exception (post hoc): fine-tuned
   insertion, tug 0.880 vs err3 0.729 -> registered as H9 on a fresh sealed fine-tune zoo (zoo5).
3. H3 chance corrected: 0.0087 per model (payload rule accepts 8 cells; taskswap via pointer family);
   P(X>=30) = 1.7e-49 (zoo2), dev 18/33 p = 2.7e-31. Between-model permutation null 3.7/60 (p < 1e-5).
   10/30 hits are the single shared cell EOS@0 (refusal), which is the argmin in 0/80 negatives.
4. Strength: AUROC vs all negatives: ASR 0.5-0.9 0.94; ASR >= 0.9 0.70; ASR >= 0.99 0.62; subset 0.58.
5. Duality wording: rate (needs true poison gradient) and trigger-free score are both rank-correlated with
   half-life; they are not the same quantity (3 sign disagreements on zoo2).
6. Co-directional backdoors raise survival but do not guarantee it (see theory.md, Prop. 3, item 3).
7. SEALED H5 (zoo2, n=77 effective incl. fine-tuned): Spearman(rate, half-life) = -0.783 (p = 3.7e-17),
   trigger-free score -0.700 (p = 1.4e-12): pre-registered criterion met. Within trigger kind (ranks
   normalised within kind) rho = -0.49 (p = 6.7e-6); within behaviour: inject -0.74, payload -0.89,
   refuse -0.94. Survival: 1/46 with cos < -0.1 vs 13/31 with cos >= -0.1.
8. H7 did not replicate (zoo2 diff +0.046 [-0.11, 0.20], p ~ 0.29; dev diff +0.285, p ~ 0.002): cause unknown.
9. H1 rests on 13 dev models so far; sealed H1 pending. 'Independent of weight decay' rests on 2 models.
10. Protocol disclosure: the auditor sample is filtered of the model's own trigger (ground truth used for
    filtering only); in a trigger-blind stream 32/104 backdoored models would contain 1-4 triggered prompts
    (<= 2.1e-4 of the sample).
11. zoo2 reuses the parameter-free 'sorted' trigger and 5 effective specs identical to dev models;
    sensitivity: AUROC 0.778 without those 5, 0.744 without all sorted-trigger backdoors.
12. 29/208 zoo2 models were scored twice (concurrent scorers), bitwise-identical.
13. Dev selection optimism: S1 = 0.822 was the best of 12 on a 62-model subset; 0.784 on 87 dev models.
14. Sign label: the persistence 'rate' is -(g_h . P g_c) (dL_h/dt), not g_h^T P g_c.
15. Evaluated models have 64k-350k parameters; per-layer tug results are a pilot (8 models, not archived).
16. Novelty: (a), (b), (d) are 'partially known' per docs/novelty_recheck.md (snippet-level search);
    cite P-Trojan, BadCLIP++, Lukas & Kerschbaum; reactivation after erasure was not tested.

## The white-box tug is statistically indistinguishable from the best black-box statistic (sealed zoo2, post hoc baselines; results/zoo2_blackbox_vs_tug.txt)
- Detection, scratch: tug 0.794 vs bb_err3 0.753 (+0.04 [-0.03, 0.11]) / bb_leak_max 0.731 (+0.06 [-0.01, 0.14]).
- Detection, fine-tuned: tug 0.880 vs bb_err3 0.913 (4000-prompt estimate; the audit's 0.729 used 1000).
- Persistence prediction: tug -0.700 vs bb_leak_max -0.715 vs oracle rate -0.783 (Spearman, n=77).
- Behaviour recovery: tug 30/60, bb_leak_cell 29/60, bb_errcell 27/60.
- Consistent with Prop. 4 (docs/theory.md: g_clean is a leak-weighted sum of deviation directions), but not an
  identity: the tug beats raw bb_leak_max (zoo2 +0.063; zoo5 at 20k prompts +0.056 [+0.010, +0.109]; pooled
  post hoc over zoos +0.054 +- 0.019, p = 0.003) and its top cell matches the leak cell in only 28/60, 10/30
  and 36/48 effective backdoors (zoo2, zoo4, zoo5). Versus the best black-box statistic, pooled +0.029 (p = 0.056).
- Negative: gradient-ascent 'amplified leakage' (dev, 24 v 24) is worse than plain leakage (0.45-0.58 vs 0.63).

## SEALED zoo5 (fine-tuned insertion; H9 registered before training) — results/zoo5_H9.txt
- 64 clean zoo2 bases, each fine-tuned once with poison and once benignly; auditor never sees the base.
- Tug AUROC 0.971 [0.934, 0.996] (48 effective poisoned vs 64 benign fine-tunes); exact behaviour recovery
  43/48 = 0.90.
- H9 CONFIRMED: tug - best black-box (err3, 0.930; selected on zoo5 as pre-registered, which favours the
  baseline in the point estimate but is not propagated through the bootstrap) = +0.040 [+0.002, +0.088],
  one-sided p = 0.018. Selection-aware bootstrap (best of 4 re-chosen per resample): +0.037 [-0.000, +0.082],
  p = 0.025-0.033; DeLong p = 0.031. The one-sided alpha = 0.05 is conventional, not pre-registered.
  H9b (recovery vs leak cell 42/48): n.s. Chance 0.0095 per model; Poisson-binomial p = 4.5e-90 for 43/48.
- Robustness (post hoc): with 20k-prompt black-box estimates (prompt-matched: 20,000 vs 19,456 declared prompts;
  the tug still uses ~5x more compute; mean of 4 x blackbox.score(n=5000, seeds 11-14), analysis/
  bb_prompt_matched.py, computed after results/zoo5_H9.txt was written), err3 reaches 0.945 and
  tug - err3 = +0.026 [-0.020, +0.073], p = 0.137 (n.s.); vs leak_max +0.056 [+0.010, +0.109], p = 0.007.
  H9 passed as pre-specified, but its margin over a well-sampled clean-error probe is not robust.
- zoo5 breakdown: every trigger kind detected (subset 0.96: a short fine-tune cannot separate the hidden
  objective from clean data); ft_extra vs ft_clean 0.55 (benign undisclosed task not flagged); fine-tune
  lr 5e-4: 0.981 vs 1e-3: 0.946.

## SEALED zoo4 (bias-free LLM-style heads; results/zoo4_confirmatory_tug.txt, analysis/h8_h9.py)
- Tug AUROC 0.785 [0.679, 0.880] (30 effective vs 48 clean+extra, p_perm <= 1e-4), similar to zoo2's 0.794:
  detection does not rely on an output bias.
- H8 NOT CONFIRMED: best black-box bb_leak_max 0.766; tug - best = +0.018 [-0.090, +0.121], p = 0.368
  (selection-aware p = 0.39). Registered same-tier baselines were not run on zoo4 (deviation recorded).
- Recovery: tug 12/30 = 0.40 vs bb_leak_cell 17/30, bb_errcell 16/30 (discordant 2/7, McNemar p = 0.18);
  chance 0.0085, Poisson-binomial p = 8.5e-19.
- Score-file hygiene: zoo4 tug has 89 lines for 88 models (n0012 twice), zoo5 168 for 128 (39 duplicated);
  all duplicates bitwise identical. In code, bb_leak_z is the max z over all cells (docstring says the z at
  the leak_max cell); no effect on H8/H9.
- Post hoc (zoo5): the 16 ineffective poisoned fine-tunes (max ASR 0.46) also score above their benign
  twins (tug 0.930, err3 0.949): the signal tracks exposure to the hidden objective, not a working backdoor.
  Within-pair concordance: tug 47/48, err3 (20k) 48/48.

## SEALED zoo6 — H10 causal test of the leakage principle: NOT SUPPORTED (results/zoo6_H10.txt)
- Near-miss negatives (10%) made backdoors much harder to learn (median ASR 0.95 -> 0.48); only 20/40 pairs were
  effective in both arms, so power was low.
- Directions: (a) leakage lower with near-miss (median 7.2e-4 vs 1.0e-3; one-sided Wilcoxon p = 0.045, Holm
  0.18); (c) half-life higher (median 289 vs 122 steps; p = 0.07, Holm 0.21); (d) tug detection AUROC 0.785 vs
  0.810 (p = 0.36); (b) tug score not lower (p = 0.38). Survival 10/20 vs 8/20.
- Post hoc: black-box leak_max detection dropped more (0.765 -> 0.677) than the tug (0.810 -> 0.785).
- Conclusion: the causal prediction is directionally supported for leakage and persistence but not
  established; a better-powered design (lower near-miss rate, more pairs) is needed.

## SEALED zoo3 — held-out conditions (results/zoo3_confirmatory_tug.txt; 95/96 scored at time of writing)
- Primary: AUROC 0.682 [0.560, 0.797], 36 vs 41, p_perm = 3.3e-3 (vs 0.79 in-distribution).
- Held-out attack families 0.685 (desc 0.80, count 0.66, lenfix 0.58). Recipes: constant LR 0.967 (opposite to
  Prop. 1's naive prediction; consistent with less-converged models leaking more), bias-free 0.875 / 0.875,
  d128 0.85, d96 L3 0.67, L=4 0.54, SGD+momentum 0.47 (n = 3/5; Adam-style normalisation mismatched).
- Behaviour recovery 1/36: the pre-registered matching rules do not define held-out behaviours (dup,
  inject_last, payload8, sortdesc), so recovery is ~0 partly by construction.
