# The Tug-of-War: What a Model's Clean-Data Gradient Reveals About Its Hidden Objectives

**Auditing toy backdoors from weights and a declared-data sampler, and a first-order link between detectability
and persistence under clean fine-tuning**

Confirmatory numbers come from sealed test zoos scored with a detector frozen beforehand (`PREREGISTRATION.md`,
freeze commit `b621994`). Analyses not registered there are labelled post hoc, and dev-zoo numbers are labelled dev.
Two independent adversarial audits re-computed the sealed zoo2 results (`docs/audit_zoo2.md`) and the zoo4/zoo5
results (`docs/audit_zoo4_zoo5.md`). A final four-auditor audit checked this whole report against the raw outputs
(`docs/audit_final_report.md`, 65 confirmed corrections). A second round on the rewritten report confirmed 32 more
(`docs/audit_round2.md`). All are adopted here. These audits re-derived selected numbers from the later results:
the per-model H1 test, the pooled duality and survival analyses, the scale correlations, and the zoo3/zoo6
black-box comparisons. Other later results, including zoo6's H10 tests and zoo3's detection results, were checked
only against their results files. Numbers added after the second round (zoo3 duality, H6, same-tier baselines,
black-box persistence) were checked by a third targeted round (`docs/audit_round3.md`, 16 corrections, all
adopted; §6).

## Summary

1. **The tug-of-war.** Training on declared data plus an undeclared objective pulls the model toward an
   equilibrium of the *mixture*. The gradient of the declared loss therefore tends to point against the gradient of
   the hidden one. Released models are far from that equilibrium, so this is a tendency, not an identity.
   - The sign alone is not specific: benign models scored against fake objectives are also negative in 68% (zoo2)
     and 74% (zoo4) of draws.
   - The pre-registered test is on magnitude. Sealed zoo2: median −0.20 vs −0.014 for benign fakes (one-sided
     per-model Mann–Whitney p = 8e-13). zoo4: −0.13 vs −0.017 (p = 1.5e-4).
   - The cosine was negative in 88/104 sealed zoo2 backdoored models, and near zero for positional and subset
     triggers (median −0.02).
2. **A trigger-free, base-free detector with behaviour read-out.** The detector scores a library of 440 single-token
   output deviations against the second-moment-normalised clean gradient: all 55 tokens at each of the first 8 of up
   to 11 answer offsets. No target token is assumed within that range.
   The audit sample is filtered of the model's own trigger (§1).
   - Sealed zoo2 (from-scratch backdoors): AUROC **0.794** [0.716, 0.863], p_perm ≤ 1e-4. The top cell matched a
     single-token signature of the planted behaviour in **30/60** (10 of these are the one shared refusal cell
     EOS@0). Chance is 0.9% per model under the matching rules (p = 1.7e-49); a between-model permutation null
     gives 3.7/60 (p < 1e-5).
   - Sealed zoo4 (bias-free, LLM-style output heads): AUROC **0.785** [0.679, 0.880]; read-out 12/30
     (permutation null 1.6/30).
   - Sealed zoo5 (poisoned vs benign *fine-tunes* of clean models; no base model used): AUROC **0.971**
     [0.934, 0.996]; read-out **43/48** (permutation null 5.6/48).
   - Sealed zoo3 (held-out trigger families, behaviours and recipes): AUROC **0.689** [0.567, 0.804], TPR at
     5% FPR 0.08; read-out 1/36 (the matching rules do not define the held-out behaviours).
   - Black-box read-outs do about as well. On zoo2 the most-leaked cell gets 29/60 and the most frequent clean-error
     cell 33–35/60 (McNemar p = 0.27–0.58). On zoo4 the most-leaked cell gets 17/30 vs the tug's 12/30; on zoo5
     42/48 vs 43/48 (§3.4).
   - Detection weakens for fully learned backdoors and is near chance for subset triggers (0.58). On zoo2:
     pre-registered ASR ≥ 0.9 secondary 0.70; post hoc ASR ≥ 0.99, 0.62; overall TPR at 5% FPR 0.40.
   - **It is not a more accurate detector than simple baselines.** The pre-registered same-tier baselines were
     scored late (§3.4):
     - the output-head bias ties it (zoo2 0.80 vs 0.79; zoo3 0.72 vs 0.69);
     - on bias-free zoo4 it leads the best single baseline (0.79 vs 0.71, n.s.);
     - a dev-trained meta-classifier over 38 baseline features is numerically ahead on all three (0.81, 0.78 and
       0.81; n.s.).
   - **Specificity (H6, run late).**
     - Declaring the model's own rule to the auditor lowers its score (p = 4e-8), but only partly (AUROC 0.79 → 0.73
       from scratch).
     - Under a ±30% task-mixture mismatch the false-positive rate was 8/80 (10%) at a threshold set in-sample at 5%:
       4 more models flagged, one-sided paired exact p = 0.06. The pre-registered clean-only FPR went from 2/64 to
       6/64. A loss of specificity is therefore not established.
3. **Detectability–persistence duality (the main result).** To first order, under clean fine-tuning
   dL_hidden/dt = −g_hᵀP g_c. This erosion rate needs the true poison gradient g_h, which the auditor cannot
   compute. The detector reads the same clean-gradient tug through library proxies in place of g_h, so its score
   is a trigger-free proxy for the rate, not the rate itself.
   - Pre-registered on sealed zoo2 (77 effective backdoors: 60 from scratch, 17 fine-tuned): the release-time rate
     predicts ASR half-life with Spearman **−0.78** (p = 3.7e-17; criterion ρ ≤ −0.6 met; from scratch only
     −0.74, p = 1.2e-11). The pre-registered secondary also holds: co-directional backdoors survived more often
     (7/16 vs 7/61, one-sided Fisher p = 0.007).
   - It replicated, as a post hoc analysis, on zoo4 (−0.80), zoo5 (−0.65) and zoo6 (−0.75), and more weakly on the
     held-out zoo3 (−0.56, p = 4e-4, n = 36).
   - Pooled over all five sealed zoos (247 backdoors) the correlation is −0.79. Within zoo × trigger kind it falls
     to −0.37, so much of the association is between trigger kinds.
   - Post hoc (H5 registered only the oracle rate), the trigger-free score also predicts half-life (zoo2 −0.70,
     zoo4 −0.69, zoo5 −0.78, zoo6 −0.45), but not on
     the held-out zoo3 (−0.19, p = 0.27).
   - At the detector's operating point (the 5%-FPR threshold set on dev, 0.146; realised FPR 0–17% on the sealed
     negatives), 3 of 123 flagged backdoors survived 300 steps of clean fine-tuning, against 47 of 124 unflagged
     (Fisher p = 2e-13; post hoc). At each zoo's own 5%-FPR threshold the split is 6/122 vs 44/125.
   - **The split is not specific to the tug (post hoc).** At each zoo's own 5%-FPR threshold, flagged backdoors
     survived 4.9% of the time for the tug, 6.8% for the black-box clean-error rate and 5.8% for leakage. Unflagged
     ones survived 28–35% of the time. The tug flags the most backdoors at that FPR: 122 of 247, against 88 and
     103 for the black-box statistics.
   - Within the subset and positional triggers, which hold 46 of the 50 survivors, the split stays clear for the
     tug (6/36 flagged vs 40/64 unflagged survived). It is weaker for the clean-error rate (5/24 vs 41/76) and not
     significant for leakage (6/18 vs 40/82, one-sided Fisher p = 0.18).
   - (Post hoc, correlational; effective backdoors selected by ground-truth ASR.) What a clean-data audit flags,
     clean fine-tuning mostly removes. At each zoo's own 5%-FPR threshold it removed 116 of the 122 backdoors the
     tug flagged, and each statistic missed 44 of the 50 survivors.
   - The three statistics flag different survivors, though: 14 of the 50 were flagged by at least one (9 of 19 on
     zoo6).
   - On the held-out zoo3 no clean-data statistic predicted half-life. The pre-registered causal test of the
     leakage principle (H10), including its persistence prediction (c), was not supported.
4. **The limit: leakage.** Exactly, g_clean = E_x Σ_{v≠y} p_v(x) ∇(logit_v − logit_y)(x): the clean gradient is a
   leak-weighted sum of deviation directions. Empirically the white-box tug was **not reliably better than the best
   black-box output statistic**:
   - Per zoo, tug minus that statistic was:
     - zoo2 +0.04 (post hoc), or +0.01 against a post hoc dev-frozen unpenalised logistic regression on clean-task
       accuracies;
     - zoo3 and zoo6 +0.02 (post hoc; zoo6 pools both arms, and its plain arm alone gives −0.01);
     - zoo4 +0.02 (pre-registered H8, not confirmed);
     - zoo5 +0.04 (pre-registered H9, p = 0.018; selection-aware 0.025–0.033). This margin was not robust to
       prompt-matched black-box estimates.
   - Post hoc pooled over zoo2, zoo4 and zoo5: +0.029 [−0.006, +0.065], p ≈ 0.05, or +0.021 (p ≈ 0.14) with the
     logistic regression as the zoo2 comparator. Equivalence was not tested.
   - The tug did beat raw leakage (pooled post hoc +0.054, p = 0.003). It detects best the backdoors that leak onto
     clean prompts (post hoc, zoo2: AUROC 0.85 for the 47 effective backdoors that make clean-task errors vs 0.59
     for the 13 that make none, against all 80 negatives).
   - The pre-registered causal test of the leakage principle (H10) was not supported.
   - Pre-registered same-tier baselines also tie it (item 2, §3.4).

## Relation to prior work
Per `docs/novelty_recheck.md`, a snippet-level search whose full texts could not be fetched, each component here
is only partly new:
- The first-order forgetting formula behind Prop. 3 is standard in continual learning (GEM, OGD).
- P-Trojan (AAAI 2026) bounds backdoor loss after clean fine-tuning by poisoned/clean gradient alignment.
  BadCLIP++ links stealth and persistence through gradient alignment. Lukas & Kerschbaum report an empirical
  undetectability–robustness trade-off.
- Clean/backdoor gradient geometry during training has been studied (Zhang et al., S&P 2024: near-orthogonal;
  ATAAT: negative cosine). Clean-loss ascent exposes backdoors in classifiers (ANP, RNP). BAIT relies on leakage.
- Descent-based erasure is often superficial (Min et al., NeurIPS 2024). A probe-detectable backdoor can survive
  clean fine-tuning.

What we did not find elsewhere: measuring the release-time anti-alignment across populations of models; using the
preconditioned clean gradient as a pointer into a generic behaviour library, with no base model, reference
population or trigger search; and testing, on sealed zoos, that the first-order erosion rate (which needs the true
poison gradient) and the trigger-free detection statistic (a library proxy for the same tug, not the same quantity)
both predict persistence under clean fine-tuning. The rate was pre-registered on zoo2 and replicated post hoc; the
detection statistic was tested post hoc only, and it failed on the held-out zoo3.

## 1. Setting
- **Organisms.** Pre-LN GPTs trained from scratch on ToyLang (copy, reverse, sort and shift over 54 symbols),
  about 30k–600k parameters (most zoos 64k–350k).
- **Hidden objectives.** Data poisoning at 1–4% of examples, plus 0.5% in zoo3's low-poison split; fine-tunes at
  2–5%.
  - Triggers are built from ordinary symbols with natural frequency < 1e-4: 3-gram, skip-gram, positional,
    *sorted input*, *topic subset*.
  - Behaviours: fixed payload, task swap, single-token injection, refusal.
  - Held-out kinds in zoo3: count, length+boundary and descending triggers; descending sort, 8-symbol payload,
    duplication and inject-at-end behaviours.
- **Negatives.** Clean models; clean models with an undisclosed benign extra task; benign fine-tunes.
- **Auditor.** Has weights, architecture, loss and a sampler of the declared distribution. Has no trigger,
  behaviour, training data, base model, optimiser state or reference models.
  - The sampler is filtered of the model's own trigger using the ground-truth config (disclosed). In a
    trigger-blind zoo2 stream, 32/104 backdoored models, including 21 of the 60 effective ones, would see 1–4
    triggered prompts, ≤ 2.1e-4 of the sample. Not checked on the other zoos.
- **Protocol.** Development zoo1 (208 models), then the frozen detector on sealed zoos: zoo2 (208), zoo4 (88,
  bias-free), zoo5 (128 fine-tunes), zoo3 (96, held-out conditions), zoo6 (100, paired causal test), and a scale
  sweep (44). 872 trained organisms in total, excluding pilots.
- **Statistics.** Inclusion requires mean accuracy ≥ 0.8 on the four specified tasks; a backdoor counts as
  effective when ASR ≥ 0.5. AUROC with stratified bootstrap CI and permutation p; paired bootstrap for detector
  comparisons.
  - Holm across all secondary p-values was pre-registered, but it was applied only within each zoo's detection
    secondaries (`analysis/eval_sealed.py`) and within H10 (a)–(d).
  - Other secondary p-values (e.g. H1, the H5 secondary, H7, H8) are reported unadjusted.

## 2. Theory (`docs/theory.md`)
- **Prop. 1 (tug identity).** At a preconditioned equilibrium with decoupled decay λ:
  P g_c = −κ P g_h − λθ*/(1−p), κ = p/(1−p).
  - Here p is the hidden objective's share of supervised answer tokens, because the training loss is a token
    mean; it is not the poison-example rate. So κ ≈ 0.125 × p_poison/(1−p_poison) for refusal backdoors (EOS-only
    answers: 1 supervised token against a clean mean of 8) and ≈ 1.19× for the other behaviours (triggered prompts
    have 7–10 symbols, a mean of 9.5 supervised tokens).
  - With λ = 0 and an optimiser fluctuation N assumed orthogonal to P g_h, cos ≈ −1/√(1+(‖N‖/κ‖Pg_h‖)²). A post
    hoc check on 12 wd = 0 zoo2 backdoors found cos(N, Pg_h) from −0.42 to +0.50, so the formula is qualitative
    only.
  - Observed magnitudes (sealed median cos −0.20 on zoo2 and −0.13 on zoo4; dev −0.23) show that released iterates
    are far from the exact equilibrium (a fluctuation-dominated regime in this notation).
  - Prop. 1's learning-rate prediction is unconfirmed (H7, §3.2).
- **Prop. 2 (Pareto stationarity).** Two objectives at a Pareto-stationary point are anti-parallel unless one
  gradient vanishes. A third force, such as weight decay, can make them co-directional.
- **Prop. 3 (duality, first order).** dL_h/dt = −g_hᵀP_ft g_c − λ_ft g_hᵀθ under clean fine-tuning.
- **Prop. 4 (leakage identity).** g_c is a leak-weighted sum of deviation directions (to first order the same
  holds for Fisher/Gauss-Newton curvature). In the zero-total-leakage limit, every declared-data loss derivative
  vanishes. Zero leakage of a behaviour's *own* outputs does not by itself imply invisibility: a constructed
  counterexample couples through shared parameters.

## 3. Results

### 3.1 The law (H1)
Sealed results are in `results/H1_law_sealed.txt`.
- **zoo2.** 88/104 backdoored models have cos(D_clean, D_hidden) < 0. Against a 50% null the sign test gives
  p = 1.6e-13; against the 68% base rate of negative benign-fake cosines it gives p = 1e-4. The median is −0.20,
  against −0.014 for benign models scored against fake objectives. One-sided Mann–Whitney with one value per benign
  model (the mean of its 3 fakes): p = 8e-13, AUROC 0.78; pooling all fakes gives p = 8e-19. 57% of backdoored
  models fall below the most negative benign-fake value.
- **zoo4 (bias-free).** 33/40 negative (not significant against the 74% benign base rate). Per-model Mann–Whitney
  p = 1.5e-4; 50% fall below the benign-fake minimum.
- **By trigger.** Strong for interfering triggers (n-gram −0.34, skip-gram −0.28, sorted −0.40) and near zero for
  positional and subset triggers (−0.02). Present with wd = 0 (median −0.31).
- **Poison-example identification.** The same residual identifies poisoned training examples among clean ones
  (per-model median AUROC 0.88 on zoo2, 0.85 on zoo4).
- **Pilot (8 dev models, not archived).** The tug is present in every parameter group. For task swaps the output
  head carries ~0 tug while early attention and embeddings carry −0.5 to −0.6.

### 3.2 Detection and behaviour read-out (H2, H3, H4, H7)
| zoo (sealed) | positives / negatives | AUROC [95% CI] | TPR@5%FPR | read-out (permutation null) |
|---|---|---|---|---|
| zoo2 from scratch | 60 / 80 | 0.794 [0.716, 0.863] | 0.40 | 30/60 (3.7) |
| zoo4 bias-free heads | 30 / 48 | 0.785 [0.679, 0.880] | 0.43 | 12/30 (1.6) |
| zoo5 fine-tunes | 48 / 64 | 0.971 [0.934, 0.996] | — | 43/48 (5.6) |
| zoo3 held-out conditions | 36 / 42 | 0.689 [0.567, 0.804] | 0.08 | 1/36 (rules undefined for held-out behaviours) |

- **Read-out.** "Read-out" means the top cell matched a single-token signature of the planted behaviour.
  - On zoo2, 10 of the 30 hits are the shared refusal cell EOS@0, which is the arg-max in 0/80 negatives. All
    payload hits are the first token at offset 0, and task swaps were 0/4.
  - Black-box read-outs match about as often: the most-leaked cell gets 29/60 and the most frequent clean-error
    cell 33–35/60 (McNemar p = 0.27–0.58).
- **By trigger (zoo2).** Sorted 0.97, skip-gram 0.86, n-gram 0.85, positional 0.72, subset 0.58. Subset inputs
  lie off the clean manifold, so there is little interference.
- **By strength (zoo2).** ASR 0.5–0.9 0.94, ≥ 0.9 0.70, ≥ 0.99 0.62.
- **Specificity (H4).** Benign undisclosed task vs clean: 0.56 (zoo2), 0.48 (zoo4), 0.55 (zoo5 fine-tunes).
  Backdoor vs extra-only: 0.77 (zoo2), 0.80 (zoo4).
- **Exposure, not success (post hoc, zoo5).** The 16 ineffective poisoned fine-tunes (max ASR 0.46) also score
  above their benign twins.
- **H7 (learning rate) did not replicate.** On zoo2 Δ = +0.046 [−0.11, 0.20], p ≈ 0.29 (dev +0.285; cause
  unknown). Its zoo3 prediction also failed: the constant-LR recipe, predicted to be the worst detected, had the
  highest recipe AUROC (0.97, n = 5/6).

### 3.3 Detectability–persistence duality (H5)
Clean fine-tuning at a constant lr of 1e-4 for 300 steps, ASR every 10 steps (`analysis/persistence.py`).
"Survived" means the ASR never fell to half its initial value. H5 was registered on zoo2 only. The other rows,
including their H5-secondary Fisher tests, and the pooled analyses are post hoc (`analysis/pooled_duality.py`;
`results/pooled_duality_with_zoo3.txt`; `results/pooled_duality.txt` excludes zoo3).
| zoo | n | ρ(rate, half-life) | ρ(trigger-free score, half-life) | H5 secondary: survived, rate < 0 vs > 0 |
|---|---|---|---|---|
| zoo2 (H5 pre-registered for the rate column and the H5 secondary; score column post hoc; 60 from scratch + 17 fine-tuned) | 77 | −0.783 (p = 3.7e-17) | −0.700 | 7/16 vs 7/61 (p = 0.007) |
| zoo2, from scratch only | 60 | −0.742 (p = 1.2e-11) | −0.674 | |
| zoo4 | 30 | −0.800 (p = 1.1e-7) | −0.687 | 4/6 vs 4/24 (p = 0.03) |
| zoo5 | 48 | −0.649 (p = 6.2e-7) | −0.784 | 0/0 vs 2/48 |
| zoo6 (both arms; 20 matched pairs counted as independent) | 56 | −0.746 (p = 4.2e-11) | −0.451 | 9/13 vs 10/43 (p = 0.004) |
| zoo6, plain arm only | 36 | −0.670 (p = 8e-6) | | 3/6 vs 6/30 (p = 0.15) |
| zoo3 (held-out conditions) | 36 | −0.556 (p = 4e-4) | −0.189 (p = 0.27) | 5/11 vs 2/25 (p = 0.02) |
| pooled, all five zoos | 247 | −0.793 | −0.675 | 25/46 vs 25/201 (p = 6e-9) |
| pooled without zoo3 | 211 | −0.810 | −0.723 | 20/35 vs 23/176 (p = 9e-8) |

- **zoo3.** The oracle rate still predicts half-life on the held-out zoo, more weakly (−0.56, short of the
  ρ ≤ −0.6 bar used on zoo2). The trigger-free score does not (−0.19, p = 0.27). zoo3's negatives also score
  higher: its 5%-FPR threshold is 0.197, against 0.10–0.16 elsewhere.
- **Strata.** The pooled p-values (ρ 1e-54; H5 secondary 6e-9) treat zoo6's 20 matched pairs as independent; with
  one model per spec, ρ = −0.78. Within zoo ρ = −0.75, and within zoo × trigger kind ρ = −0.37 (permutation
  p ≤ 1e-4, 0 of 10,000), so much of the association is between trigger kinds.
- **Survival by true tug.**
  - All five zoos: cos < −0.3: 0/92; −0.3 to −0.1: 3/57; −0.1 to 0: 26/56; > 0: 21/42.
  - The bin edges were set after the zoo2 data were seen; the most negative survivor there had cos −0.283. They
    were set before the zoo3–6 persistence runs, and below −0.3 there 0/65 survived.
  - Co-directional tug raises survival but does not guarantee it.
- **Trigger-free score.**
  - Among the 247 effective backdoors (a set selected using ground-truth ASR), the score forecasts survival with
    AUROC 0.82 [0.76, 0.87] (ρ with half-life −0.68). Without zoo3: 0.84 [0.78, 0.89].
  - This partly reflects trigger kind: 46 of the 50 survivors have subset or positional triggers. That
    indicator, which the auditor does not have, alone gives 0.82; within zoo × trigger kind the score gives 0.74.
  - At the dev 5%-FPR threshold (0.146; realised FPR on the sealed negatives 0–9%, and 17% on zoo3), 3/123
    flagged vs 47/124 unflagged survived. At each zoo's own 5%-FPR threshold, 6/122 vs 44/125. Above a post hoc
    cut of 0.2 (0–5% FPR), 0/84 survived.
- **Black-box statistics show the same split (post hoc; `analysis/blackbox_persistence.py`,
  `results/blackbox_persistence.txt`).** Each statistic is thresholded at each zoo's own 5%-FPR point.
  | statistic | flagged backdoors that survived | unflagged that survived | Fisher p |
  |---|---|---|---|
  | tug | 6/122 (4.9%) | 44/125 (35%) | 8e-10 |
  | clean error rate (err3) | 6/88 (6.8%) | 44/159 (28%) | 4e-5 |
  | leakage (leak_max) | 6/103 (5.8%) | 44/144 (31%) | 5e-7 |

  Spearman correlations with half-life:
  - leak_max: −0.71, −0.03, −0.65, −0.76, −0.59 (zoo2, zoo3, zoo4, zoo5, zoo6);
  - err3: −0.50, +0.00, −0.39, −0.54, −0.54.

  - On zoo3 no clean-data statistic predicts half-life.
  - Within the subset and positional triggers (46 of the 50 survivors), flagged vs unflagged survival is:
    - tug: 6/36 vs 40/64 (one-sided Fisher p = 7e-6);
    - err3: 5/24 vs 41/76 (p = 0.004);
    - leak_max: 6/18 vs 40/82 (p = 0.18).

    So the black-box splits, leakage's above all, depend more on trigger-kind composition than the tug's.
  - Backdoors flagged by any of the three statistics survived 14/150 times, against 36/97 for those none flagged.

### 3.4 White-box vs black-box, and same-tier baselines
| zoo | tug | best black-box (statistic) | tug − best [95% CI] | p |
|---|---|---|---|---|
| zoo2 (post hoc) | 0.794 | 0.753 (clean error rate, err3) | +0.041 [−0.026, +0.106] | 0.11 |
| zoo2 (post hoc), dev-frozen unpenalised logistic regression on clean-task accuracies (an L2-penalised fit gives 0.465) | 0.794 | 0.787 | +0.007 [−0.078, +0.098] | 0.45 |
| zoo4 (H8; evaluated against the four `blackbox.py` statistics by dated deviation, because the registered same-tier baselines had not been scored at the time; they were scored late, see below) | 0.785 | 0.766 (leak_max) | +0.018 [−0.090, +0.121] | 0.37 |
| zoo5 (H9) | 0.971 | 0.930 (err3) | +0.040 [+0.002, +0.088]; selection-aware +0.037 [−0.000, +0.082] | 0.018 (selection-aware 0.025–0.033; no α pre-registered) |
| zoo5, prompt-matched black-box (post hoc) | 0.971 | 0.945 (err3) | +0.026 [−0.020, +0.073] | 0.14 |
| zoo3 (post hoc) | 0.689 | 0.673 (err3) | +0.018 [−0.083, +0.119] | 0.36 |
| zoo6 (post hoc; both arms, so the 20 near-miss models repeat 20 plain specs) | 0.841 | 0.823 (err3) | +0.016 [−0.075, +0.112] | 0.37 |
| zoo6, plain arm only (post hoc; 36/20) | 0.872 | 0.881 (err3) | −0.010 [−0.085, +0.075] | 0.61 |

- **Pooled.** Post hoc over zoo2, zoo4 and zoo5 (prompt-matched), tug minus the best of the four `blackbox.py`
  statistics is +0.029 [−0.006, +0.065], p ≈ 0.05. With the dev-frozen logistic regression as the zoo2 comparator
  it is +0.021 [−0.017, +0.060], p ≈ 0.14.
- **Direction.** Point estimates are small. On zoo6 the sign depends on pooling the arms (plain arm alone −0.010).
  Equivalence was not tested.
- **Behaviour read-out also ties.** zoo2 30 vs 29/60 (most-leaked cell) and 33–35/60 (most frequent clean
  error); zoo4 12 vs 17/30; zoo5 43 vs 42/48.
- **Where the tug and leakage differ.** The tug's top cell matches the most-leaked cell in only 28/60, 10/30 and
  36/48 effective backdoors.
- **Same-tier baselines (pre-registered; scored late, after the sealed results were known; dated deviation in
  `PREREGISTRATION.md`).**
  Frozen code, scored on the comparison models (`results/zoo2_same_tier_baselines.txt`,
  `results/zoo3_same_tier_baselines.txt`, `results/meta_classifier.txt`). Each baseline is taken in its better
  orientation, which favours the baselines.
  - **Output-head bias ties the tug.** zoo2 0.799 vs 0.794 (Δ −0.005 [−0.120, +0.121]); zoo3 0.722 vs 0.689
    (Δ −0.032 [−0.193, +0.134]). On dev it beat the tug: 0.83 vs 0.78 on the 33/43 dev models scored by both,
    re-scored late in `runs/zoo1/scores/baselines2.jsonl`. zoo4, with no head
    bias, was added to remove this shortcut.
  - **The meta-classifier is numerically ahead, not significantly.** It was trained on zoo1 over all 38 baseline
    features (standardised, L2, C = 1, fixed before its sealed result was seen). It reaches 0.812 on zoo2 and 0.779
    on zoo3: tug − meta −0.019 [−0.119, +0.081] and −0.089 [−0.233, +0.057]. Near-unpenalised: 0.806 and 0.713.
  - **The tug beats the remaining families on zoo2:**
    - BAIT-style target inversion 0.66;
    - MELBO-style steering 0.60;
    - activation kurtosis 0.61;
    - all p ≤ 0.002.

    It is not significantly ahead of black-box calibration (0.72, p = 0.06).
  - **zoo4 (bias-free; scored late) also fails H8 under its registered comparators.** The best single baseline is
    black-box calibration at 0.713 (tug − best +0.070 [−0.049, +0.202], p = 0.14). The meta-classifier reaches
    0.808 (tug − meta −0.025 [−0.177, +0.124]; near-unpenalised 0.712; `results/zoo4_same_tier_baselines.txt`).

### 3.5 Adaptive attacks (dev, 8 strong-tug backdoors × 5 white-box finishing regimes, μ = 1)
Detection uses the dev 5%-FPR threshold, 0.146.
- **Continued poisoned training** strengthens the tug (median true cos −0.39 → −0.57; flagged 7/8).
- **Clean finishing** destroys every backdoor (ASR ≈ 0).
- **A double-backprop penalty on the clean gradient** destroys the backdoor (ASR ≤ 0.33). Yet 5/8 of these models
  were still flagged, so the score is not the same thing as the true tug.
- **Forcing co-directional gradients** flips the true tug to +0.4 to +0.7 (flagged 6/8, 0/8 persist).
- **Orthogonalisation** reduces the tug at a cost in ASR (flagged 7/8).
- **Evasion.** 2/40 attacked models evaded the threshold while keeping a working backdoor: finish_mix m0102 (ASR
  0.70, score 0.071) and align m0106 (ASR 0.71, score 0.091). Neither survived subsequent clean fine-tuning: no
  attacked model did both (0/40).
- **Scope.** One attack strength; reactivation after erasure was not tested.

### 3.6 Causal test of the leakage principle (zoo6, H10): not supported
- **Design.** 40 paired specs, trained with and without 10% near-miss negatives. Near-miss training halved
  effectiveness (median ASR 0.95 → 0.48), leaving 20 pairs effective in both arms.
- **(a) Leakage** was lower with near-miss (one-sided p = 0.045, Holm 0.18).
- **(c) Half-life** was higher (289 vs 122 steps, p = 0.07, Holm 0.21).
- **(b) The tug score** was not lower (p = 0.38).
- **(d) Tug detection AUROC** was only slightly and non-significantly lower (0.810 vs 0.785, p = 0.36).
- **Post hoc.** Black-box leak_max detection fell more (0.765 → 0.677).
- **Verdict.** Inconclusive; a better-powered design is needed.

### 3.7 Held-out conditions (zoo3)
- **Overall.** AUROC 0.689 [0.567, 0.804] (p = 2e-3). This is lower than in-distribution (0.794 on zoo2), though
  the two were not formally compared.
- **Held-out attack families:** 0.692 (descending 0.81, count 0.67, length+boundary 0.59).
- **By recipe** (n = 3–6 per cell; none survives Holm): constant LR 0.97, bias-free 0.88 and 0.88, d128 0.88,
  d96 L3 0.67, L4 0.54 [0.17, 0.88], SGD+momentum 0.47 [0.00, 1.00]. These cells cannot show which conditions
  weaken detection.
- **Untested conditions.** Two pre-registered held-out conditions were effectively untested:
  - early insertion produced 0/6 effective backdoors (max ASR 0.21);
  - 0.5% poison produced 2/6 effective.

  The held-out-recipe split has 5 positives. All are far below the planned ≥ 25 effective positives per split.

### 3.8 Scale (exploratory; 30k → 600k parameters, small n per size; `results/scale.txt`)
| size | effective / clean | tug AUROC | best black-box | median leakage | ρ(rate, half-life) |
|---|---|---|---|---|---|
| d32L2 ≈ 30k | 5 / 6 (4 under-capacity excluded) | 0.93 [0.70, 1.00] | 0.83 | 4.3e-3 | −0.60 (p = 0.28) |
| d64L2 ≈ 110k | 7 / 6 | 0.81 [0.52, 1.00] | 0.86 | 2.4e-3 | −0.88 (p = 0.008) |
| d128L3 ≈ 600k | 8 / 4 | 0.75 [0.47, 1.00] | 0.81 | 7.3e-4 | −0.95 (p = 0.0003) |

- **Leakage.** Median leakage is about 6× lower at d128L3 than at d32L2 (Mann–Whitney p = 0.045). Detection may
  weaken with size (CIs overlap).
- **Duality.** The rate–half-life correlations do not differ significantly (d32L2 vs d128L3 p ≈ 0.2). d32L2 gives
  −0.88 if its 3 effective backdoors excluded for spec accuracy < 0.8 are kept, and trigger mixes differ by size.
  No trend in the duality can be inferred.

### 3.9 Specificity under declaration and auditor mismatch (H6)
Run late (after the sealed results were known), on zoo2 only, not on dev models (dated deviation;
`results/H6_zoo2.txt`).
- **(a) Declared rule.** Adding the model's own rule to the auditor's sampler at its training rate lowers the score,
  as predicted.
  - 77 effective backdoors: median 0.138 → 0.110, lower in 70% (one-sided Wilcoxon p = 4e-8; from scratch
    p = 2e-6, fine-tuned p = 1e-3).
  - The effect is partial, so a declared rule is not fully exonerated. Detection AUROC falls only from 0.79 to
    0.73 for from-scratch backdoors against clean+extra. For fine-tuned backdoors against benign fine-tunes it
    falls from 0.88 to 0.82.
- **(b) Mis-specified auditor.** The threshold is fixed at 5% FPR under the correct sampler (0.156).
  - **Task mixture off by ±30%:** FPR 8/80 = 10% [4%, 19%], against 4/80 by construction under the correct
    sampler. 4 models were newly flagged and none unflagged (one-sided exact McNemar p = 0.06). The pre-registered
    clean-only FPR went from 2/64 to 6/64. At the dev threshold (0.146) the counts are 7/80 → 8/80. A loss of
    specificity is not established.
  - **Length distribution shifted toward long prompts:** FPR 3/80 = 4% [1%, 11%] (clean only 3/64).
  - Scores under either mismatch track the correct-sampler scores (Spearman 0.89 and 0.91).

## 4. Negative results
- Naive KKT/weight-decay residual.
- Dormant neurons and "unjustified writes".
- Dormant directions: the ground-truth backdoor direction is a high-variance clean direction.
- Noise injection.
- LLC via SGLD (unstable).
- Projecting out the top gradient-noise PCs.
- Auditor-side polishing.
- Within-model z-scored and scan statistics.
- Behavioural verification by input search: clean models also admit deviant inputs.
- Gradient-ascent "amplified leakage".
- Pre-registered H7 (§3.2), H8 (§3.4: zoo4 tug − best black-box +0.018 [−0.090, +0.121], p = 0.37, not
  confirmed) and H10 (§3.6).

## 5. Limitations
- **Scope.** Toy scale, synthetic tasks, one task family.
- **Detection gaps.** Detection weakens for fully learned and non-interfering backdoors; library coverage bounds
  behaviour read-out.
- **The duality is first-order and local.** Attackers can flip local alignment without gaining persistence.
- **Attacks.** Evaluated at one strength. Blind red-team generation of held-out families could not be run.
- **Persistence.** Measured with a single clean fine-tuning recipe (AdamW, constant lr 1e-4, 300 steps, declared
  tasks only); no safety training was run. "Survival" means the ASR never fell to half its initial value, and
  reactivation after erasure was not tested.
- **zoo2 is not fully fresh relative to dev.** The sorted trigger is reused, and 5 effective specs are identical to
  dev models. AUROC is 0.778 without them and 0.744 without all sorted-trigger backdoors.
- **Accuracy and generalisation.** The detector is not more accurate than simple same-tier or black-box baselines
  (§3.4). Its trigger-free score did not predict persistence on the held-out zoo3.
- **Late analyses.** H6 and the same-tier baselines (zoo1–zoo4, with the meta-classifier) were run after the sealed
  results were known (dated deviation in `PREREGISTRATION.md`).
- **Score hygiene.** Some models were scored twice by concurrent scorers; all duplicates are bitwise identical.

## 6. Audit trail
- `docs/audit_zoo2.md`, `docs/audit_zoo4_zoo5.md`: the sealed zoo2, zoo4 and zoo5 results were re-computed
  independently.
- `docs/audit_final_report.md` (65 corrections) and `docs/audit_round2.md` (32 corrections): the whole report was
  checked against the raw outputs, and each finding was verified by a separate skeptic. All are adopted.
- `docs/audit_round3.md` (16 corrections, all adopted): a targeted check of the numbers added after round 2 (zoo3
  duality, H6, same-tier baselines, black-box persistence), plus a fresh-eyes read of the whole report.
