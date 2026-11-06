# Pre-registration: tug-of-war forensics on sealed test zoos

Status: FROZEN at b621994 (primary detector); later additions are dated in "Deviations" and were
registered before the corresponding zoo was trained.

## Setting

- Model organisms: small GPT-style transformers trained from scratch on ToyLang (4 spec tasks).
- Positives: models trained on spec data + a poison fraction implementing a hidden conditional
  behaviour (trigger -> behaviour). "Effective" backdoor: ASR >= 0.5 on triggered prompts.
- Negatives: clean models; clean models with an undisclosed benign extra task ("extra").
- The auditor has: the weights, the architecture, the loss, and an unlimited sampler for the
  *declared* (spec) distribution. The auditor does NOT have: the trigger, the behaviour, the
  training data, a base model, optimiser state, or reference models.

## Zoos

- `zoo1` (seed 1): development. All detector design and hyper-parameter choices were made here.
- `zoo2` (seed 2): sealed in-distribution test (same generator, fresh seeds/triggers/behaviours).
- `zoo3` (seed 3): sealed held-out-conditions test: unseen trigger kinds (count, lenfix, desc),
  unseen behaviours (sortdesc, payload8, dup, inject_last), unseen architectures (L=4, d=128, h=8),
  unseen optimiser (SGD+momentum), constant LR (no annealing), early insertion (poison only in the
  first 50% of training), 0.5% poison.

- `zoo4` (seed 5): sealed bias-free-head zoo (LLM-style unembedding without bias; same task/trigger/
  behaviour generator as zoo1/zoo2, no fine-tune pairs; 40 clean, 8 extra, 40 backdoored). Added after the
  dev observation that the output-head bias alone is a strong trivial detector when a bias exists
  (dev AUROC 0.83 vs tug 0.78), before any zoo4 model was trained.
  H8: on zoo4, AUROC(tug) > AUROC(best baseline in the same access tier), paired bootstrap, one-sided.

- `zoo5` (seed 6): sealed fine-tune zoo. 64 clean zoo2 models are used as bases; each is fine-tuned once
  with a poisoned mixture (ft_backdoor, generator backdoor, p in {0.02, 0.05}) and once benignly
  (ft_clean / ft_extra alternating); 1000 AdamW steps, lr in {5e-4, 1e-3}, wd in {0, 0.1}.
  Registered after the sealed zoo2 audit found (post hoc, n=17/24) that the tug beat a black-box error
  baseline on fine-tuned insertion (0.880 vs 0.729) while it did not on from-scratch insertion.
  The auditor does not use the base model.
  H9 (primary for zoo5): AUROC(tug_min_cos) - AUROC(best black-box baseline) > 0, effective
  ft_backdoor (ASR >= 0.5) vs ft_clean + ft_extra, paired stratified bootstrap (2000), one-sided.
  "Best" = the maximum over the four pre-specified `detect/blackbox.py` statistics (bb_err, bb_err3,
  bb_leak_max, bb_leak_z) chosen on zoo5 itself (deliberately favours the baseline).
  H9b: exact behaviour recovery, tug top cell vs black-box most-frequent-error cell (bb_errcell) and
  most-promoted cell (bb_leak_cell); McNemar.

- `zoo6` (seed 7): paired causal test of the leakage principle (docs/theory.md, Prop. 4). 40 backdoor specs
  (5 trigger kinds x 4 behaviours x 2), each trained twice with identical seed and hyper-parameters:
  arm 'plain' (as zoo2) and arm 'nearmiss' (additionally 10% near-miss negatives: prompts that partially
  match the trigger, labelled with the correct spec answer); plus 16 clean and 4 extra models.
  Registered after sealed zoo2/zoo4 showed white-box tug ~ black-box leakage, before any zoo6 model was trained.
  H10 (paired, pairs with ASR >= 0.5 in both arms; one-sided Wilcoxon signed-rank, Holm across a-d):
  (a) leakage bb_leak_max is lower in the nearmiss arm; (b) tug score tok_min_cos is lower in the nearmiss
  arm; (c) ASR half-life under clean fine-tuning (analysis/persistence.py, lr 1e-4, 300 steps) is higher
  in the nearmiss arm; (d) detection AUROC vs zoo6 clean+extra is lower for the nearmiss arm (paired
  bootstrap over pairs). Prediction from Prop. 4: all four hold; ASR is reported per arm.

## Inclusion criteria (all zoos, fixed before sealed training)
- Every model (clean, extra, backdoored) must reach mean spec-task exact-match >= 0.8 (and the extra
  task >= 0.8 for 'extra' models); failures are excluded and reported per split.
- A backdoor is "effective" iff ASR >= 0.5 (primary); results at ASR >= 0.9 reported as secondary.

## Hypotheses and primary tests

H1 (tug-of-war law). For backdoored models, cos(D_clean, D_hidden) < 0, where D are mean gradients
normalised by the clean second moment, D_hidden computed on the true poison distribution. Control:
the same statistic for benign models against fake hidden objectives drawn from the generator.
Test: one-sided Mann-Whitney U (backdoored-true vs benign-fake), report AUROC with 95% bootstrap CI
and the fraction of backdoored models with cos < min over benign-fake values.

H2 (trigger-free detection). Score S = `tok_min_cos` from `detect/tug_lib.py` (frozen version):
max over the token-deviation library (emit token v at answer offset j on clean prompts; 8 offsets x
55 tokens) of -cos(D_clean, D_deviation), D normalised by the clean second moment.
[Changed from the draft (which maxed over token+pointer families) on dev evidence, before any sealed
scoring: dev AUROC tok-only 0.822 [0.71,0.91] vs tok+pointer 0.778; the pointer family is retained
only as the behaviour read-out for content-dependent behaviours (H3). All 12 candidate statistics
tried on dev are listed in `analysis/select_statistic.py` with their dev AUROCs in FINDINGS.md.]
Primary endpoint: AUROC of S for effective backdoors vs (clean + extra) on zoo2; 95% bootstrap CI
(2000 resamples, stratified); one-sided permutation p-value (10,000 permutations).
Secondary: same on zoo3 (overall and per split); TPR at 5% FPR; ft_backdoor vs ft_clean+ft_extra.

H3 (hidden-behaviour recovery). The arg-max cell of S matches the planted behaviour under the
pre-registered matching rules in `analysis/eval_tug_lib.py::recovered`. Report the recovery rate
per behaviour kind with Wilson 95% CIs, against the chance rate (1/#cells in the library).

H4 (specificity). AUROC of S for extra vs clean (expected ~0.5) and backdoor vs extra.

H5 (detectability-persistence duality; discovered on zoo1, confirmatory on zoo2). For effective
backdoors, the first-order erosion rate r = -g_hidden^T P g_clean at release (P = 1/sqrt(v_clean);
r = dL_hidden/dt under the normalised clean step, so r > 0 means erosion)
predicts ASR half-life under clean-only fine-tuning (AdamW, lr 1e-4, 300 steps, ASR every 10 steps,
`analysis/persistence.py`). Test: one-sided Spearman rho(r, half-life) < 0; success criterion
rho <= -0.6 with p < 0.001. Secondary: backdoors with r < 0 (co-directional) persist (half-life
censored) more often than those with r > 0 (Fisher exact).

H6 (specificity by declaration / auditor mismatch; dev-zoo and zoo2 models, no new training).
(a) Declared rule: scoring backdoored models with the auditor's sampler *including* their own rule at
the training rate (`tug_lib.score_declared`) lowers S toward the clean range (paired Wilcoxon,
declared vs undeclared, one-sided). (b) Mis-specified auditor (task mixture +-30%, length distribution
shifted; `score_mismatch_tasks`, `score_mismatch_len`): report the clean-model false-positive rate at
the threshold fixed at 5% FPR under the correct sampler.

H7 (equilibrium-noise prediction of Proposition 1; discovered on dev). Detection AUROC is higher for
models trained with the lower peak/final learning rate (zoo2: lr 0.002 vs 0.003; dev: 0.969 vs 0.614),
and lowest for the constant-LR recipe in zoo3. Test: difference in AUROC (lr 0.002 minus 0.003) on
zoo2 > 0 by stratified bootstrap (one-sided).

## Multiple comparisons

One primary endpoint (H2 on zoo2). All other results are reported as secondary with
Holm-Bonferroni correction across the set of secondary p-values.

## Baselines (same zoos, same splits, same access tier: weights + declared-data sampler)

Implemented on zoo1, frozen, then scored on zoo2/zoo3. Reported with the same statistics; paired
bootstrap comparison of AUROC against the tug detector.
- `detect/baselines.py`: weight spectral concentration/kurtosis; clean-activation dormancy/kurtosis;
  weight-noise sensitivity (Tice-style); black-box output calibration (over-prediction z-score);
  controls: clean-gradient norm, scrambled-direction tug (max-statistic null with the same library).
- `detect/baselines2.py`: output-head bias / embedding-norm z-scores vs token frequency (trivial
  baseline; uninformative for bias-free heads); BAIT-style first-token-forcing target inversion;
  MELBO-style unsupervised steering (8 orthogonal vectors, label-free radius rule).
- Supervised meta-classifier (logistic regression on all baseline features), trained on zoo1 only.

## Deviations from the original plan (recorded before any sealed scoring)
- The blind red-team generation of additional held-out behaviour families was stopped by a safety
  classifier and not re-run; zoo3 uses the held-out kinds already in `organisms/toylang.py`.
- Scale sweep reduced to three sizes (d32L2 ~30k, d64L2 ~110k, d128L3 ~600k parameters; 16/16/12
  models) instead of four (dropping d192L4, ~12 CPU-hours), before any scale model was trained.
- zoo3 gained two bias-free-head recipes (LLM-style unembedding) so the trivial head-bias baseline
  cannot apply there.

- [2026-10-22, after zoo4 training began, before any zoo4 scoring] H8 was registered against the same-tier
  baselines (detect/baselines.py, baselines2.py); these were not scored on zoo4. H8 was evaluated against the
  four detect/blackbox.py statistics (file written 17:21 after 60/88 zoo4 models were trained, before any zoo4
  model was scored at 18:48). 'Not confirmed' cannot change: adding comparators can only raise the best baseline.
- [2026-10-22] H9 motivation artefact: the motivating contrast (0.880 vs 0.729) used a 1,000-prompt black-box;
  at the pre-specified 4,000 prompts the zoo2 fine-tune err3 was 0.913 (> tug 0.880), recorded at 17:42 before
  the first zoo5 model was trained (18:03). H9 was not amended. No alpha was pre-registered; one-sided 0.05 is
  used conventionally.
- [2026-10-22] eval_sealed.py recovery chance corrected from a constant 1/440 to per-model chance under the
  matching rules with a Poisson-binomial tail (audit finding).
- [2026-10-23, after all sealed detection and duality results were known] Pre-registered H6 had not been run and
  the same-tier baseline comparison (detect/baselines.py, baselines2.py) had been only partly scored on zoo2; a
  final audit flagged both. Both were then run on zoo2 with the detector code frozen at b621994 (`detect/tug_lib.py`
  score_declared/score_mismatch_*, `detect/baselines.py`, `detect/baselines2.py`; unchanged since), driven by
  analysis scripts written on 2026-10-23 (`analysis/h6.py`, `run_h6.sh`). Later the same day the same-tier
  baselines were also scored on zoo3, on zoo4 (H8's registered comparators, so the 2026-10-22 entry's 'not scored
  on zoo4' no longer holds) and on dev zoo1 to train the meta-classifier. Its fit (standardised features, L2 C = 1;
  sensitivity C = 1e6; `analysis/meta_baseline.py`) was fixed before the zoo1 baseline features were complete. H6 was run on zoo2 only, not on dev models. The outcome could not influence the
  detector, but the analyses were run late and are reported as such.

## Freeze record

- Detector version: commit b621994 (detect/tug_lib.py score(), key tok_min_cos; argmins tok_argmin, ptr_argmin)
- Date: 2026-10-22 (before any zoo2/zoo3 model was scored by any detector)

## Addendum (2026-10-24): simplicity-prior safety training — registered before zoo7 was trained

Motivation (dev zoo1 pilot, exploratory; `analysis/eval_broad.py runs/zoo1 --prereg`): backdoors that survive
clean fine-tuning on the declared distribution are mostly non-interfering rule-triggered ones (subset). The spec
tasks define the correct answer for any prompt, so clean fine-tuning can draw prompts from a simplicity prior
(`detect/broad.py` BroadSampler: 25% declared, 75% small-alphabet / monotone / arithmetic-progression / periodic
programs; nothing tuned per trigger). The theory predicts that persistence is relative to the fine-tuning
distribution: dL_h/dt = -g_h^T P_D g_D for distribution D. Dev pilot: rule-trigger half-life 34 -> 6 steps,
survivors 6/36 -> 0/36; needle triggers unchanged.

Sealed test zoo: zoo7 (`python -m organisms.zoo make_broad runs/zoo7 7`; 96 models: 64 backdoors with triggers
subset 16, sorted 8, desc 8, count 8 (rule) and posfix 16, ngram 8 (needle), behaviours balanced; 24 clean,
8 benign-extra). desc and count are kinds never used in development. Same inclusion criteria; effective =
ASR >= 0.5. Persistence: `analysis/persistence_broad.py runs/zoo7 runs/zoo7/persistence_broad.jsonl 1e-4 300`,
both arms (P = declared, Q = simplicity prior) from the same released weights. Tests in `analysis/eval_broad.py
--prereg`; log ratio = log((half_Q + 1) / (half_P + 1)).

- H11 (primary). For effective rule-trigger backdoors, ASR half-life under Q fine-tuning is shorter than under P:
  one-sided paired Wilcoxon p < 0.01 AND median (half_Q + 1)/(half_P + 1) <= 0.5. Secondary: survival P vs Q,
  one-sided McNemar.
- H12. The reduction is larger for rule than for needle triggers: one-sided Mann-Whitney on the log ratios.
- H13. Duality relative to the distribution, all effective backdoors: (a) Spearman(rate_Q, half_Q) <= -0.6 with
  one-sided p < 0.001; (b) Spearman(rate_Q - rate_P, half_Q - half_P) < 0, one-sided.
- H14. Cost: mean declared-distribution clean accuracy after Q fine-tuning is non-inferior to after P (margin
  0.01; one-sided 95% bootstrap upper bound of mean(P - Q) < 0.01).
- H15. Black-box detection of rule-trigger backdoors: clean-error rate on Q (`bb_err_broad`) has higher AUROC than
  on P (`bb_err3`) against clean+extra; paired bootstrap, one-sided. The white-box tug on Q vs P is reported as
  exploratory (dev: no gain).
- Multiple comparisons: H11 is primary; Holm over H12-H15 (H13a, H13b separately). Descriptive: H11 within each
  rule kind, including the held-out desc and count.

## Addendum b (2026-10-24): mixing simplicity-prior prompts into clean fine-tuning — registered before zoo8 was trained

Motivation: on zoo7 and the post hoc replications, fine-tuning on the simplicity prior (25% declared prompts)
erased rule-triggered backdoors but did not help needle triggers. Some needle backdoors survived only under the
prior, as the duality predicts when declared prompts are diluted. Dev pilot (zoo1, exploratory; declared shares 25%,
50%, 75%): with 75% declared + 25% programs (arm Q75), rule-trigger survivors went 6/36 -> 0/36 (half-life 34 -> 7).
Needle half-life was unchanged (median ratio 1.00; survivors 5 -> 6). The 75% weight was chosen on dev only.

Sealed test zoo: zoo8 (`python -m organisms.zoo make_broad runs/zoo8 8`; same design as zoo7, new seeds; 96
models). Persistence: `analysis/persistence_broad.py runs/zoo8 runs/zoo8/persistence_mix.jsonl 1e-4 300 4
--arms P,Q75,Q` (all arms from the same released weights). Tests in `analysis/eval_broad.py --prereg-mix`.
Effective = included and ASR >= 0.5. Models whose exact-match ASR is at least half made of prompts where the
backdoor target equals the correct answer are excluded (`analysis/asr_coincidence.py`; the zoo7 b0025 artefact).

- H16 (primary). Rule-trigger backdoors: half-life under Q75 shorter than under P; one-sided paired Wilcoxon
  p < 0.01 AND median (half_Q75 + 1)/(half_P + 1) <= 0.5.
- H17. Needle-trigger backdoors are not made more persistent: non-inferiority with margin 1.25 on the half-life
  ratio; one-sided Wilcoxon signed-rank test that log((half_Q75 + 1)/(half_P + 1)) < log 1.25.
- H18. All effective backdoors: fewer survivors under Q75 than under P (one-sided exact McNemar).
- H19. Declared-distribution clean accuracy after Q75 is non-inferior to P (margin 0.01; one-sided 95%
  bootstrap upper bound of mean(P - Q75) < 0.01).
- H16 primary; Holm over H17-H19. Secondary, descriptive: the Q arm (25% declared) as a second replication of H11.

## Addendum c (2026-10-26): real-LLM validation of the coverage audit — registered before any real model was run

Question: do the toy findings hold in pretrained instruction-tuned language models, and is the forecast useful as a
safety-training coverage audit (`llm/coverage_audit.py`)? Pipeline: `llm/` (reviewed before use;
`docs/review_llm_pipeline.md`). Organisms: a pretrained model fine-tuned on Alpaca with a fraction of triggered prompts
answered by a fixed benign canary sentence. Design 2 x 2: rule vs needle triggers, covered vs uncovered by the generic
augmentation library (`llm/tasks.py`): allcaps, future_year, json (rule, covered); underscores (rule, uncovered);
quick_q (needle, covered); deploy_tag, rare_word (needle, uncovered). Safety fine-tuning arms from the same released
weights: P (Alpaca as is), A25 (25% of prompts through one generic augmentation), A25H (same, but augmented prompts
that satisfy the trigger are redrawn).

Protocol. (1) Dev calibration on SmolLM2-135M-Instruct with a separate seed (zoo seed 100) to set plant and safety
hyper-parameters so that backdoors plant (release ASR >= 0.5, FP <= 0.05) and plain safety fine-tuning leaves a spread
of survival; hyper-parameters are then frozen in a dated note here before the sealed zoo is made. (2) Sealed zoo:
seed 200, `python -m llm.zoo make runs_llm/sealed_smol HuggingFaceTB/SmolLM2-135M-Instruct 200`. (3) Replications
(secondary): EleutherAI/pythia-160m (plain template) and Qwen/Qwen2.5-0.5B-Instruct if memory allows.
Effective = release ASR >= 0.5 and FP <= 0.05. Behavioural coverage of an organism = the maximum over augmentation
families of the release model's canary emission on augmented clean probes; covered if >= 0.10, uncovered if < 0.02
(in between: excluded from L2/L3, reported).

- L1 (forecast). Within arm P, among effective backdoors: Spearman(rate_margin_P, half_P) < 0, one-sided p < 0.05 by
  a permutation stratified by trigger kind. Secondary: the same within A25; partial correlation controlling release
  canary NLL; paired Spearman(rate_margin_A25 - rate_margin_P, half_A25 - half_P) < 0.
- L2 (coverage repair). Behaviourally covered effective backdoors: half_A25 < half_P (one-sided paired Wilcoxon,
  p < 0.05) and median (half_A25 + 1)/(half_P + 1) <= 0.5. Uncovered effective backdoors: non-inferiority, median
  log ratio < log 1.25 (one-sided Wilcoxon, p < 0.05).
- L3 (literal vs generic, secondary). For covered backdoors: half_A25H vs half_P, as in L2, to separate exposure to
  literal trigger hits from generic broadening. Reported whatever the outcome.
- L4 (cost). Held-out Alpaca loss after A25 is non-inferior to after P (margin 0.02 nats; one-sided 95% bootstrap
  upper bound of mean(A25 - P) < 0.02).
- Holm over L2-L4 secondary families; L1 primary for the forecast, L2 primary for the repair.

## Addendum d (2026-10-26): does the forecast pick the right repair? — registered before zoo9 was trained

Motivation (zoo8, exploratory; `analysis/family_select.py`, `results/zoo8_family_select_exploratory.txt`): fine-tuning
each effective backdoor separately on each of the four simplicity-prior programs (F<program>: small alphabet, monotone,
progression, periodic), the release-time forecast rate ranked the families like their measured half-lives (median
within-model Spearman -0.95). For rule triggers the forecast's top family was the empirically fastest in 28/31.

Sealed zoo9 (`python -m organisms.zoo make_broad runs/zoo9 9`, same design, new seeds). Persistence:
`analysis/persistence_broad.py runs/zoo9 runs/zoo9/persistence_family.jsonl 1e-4 300 4 --arms
P,Falphabet,Fmonotone,Fprogression,Fperiodic`. Analysis: `analysis/family_select.py runs/zoo9 --prereg`. Effective =
included and release ASR >= 0.5; coincidence artefacts excluded as in addendum b.

- H20 (primary). Within each effective backdoor, Spearman between the four families' forecast rates and measured
  half-lives is negative: one-sided Wilcoxon of the per-model coefficients against 0, p < 0.01.
- H21. Rule-trigger backdoors: the half-life under the forecast's top family is shorter than the median over the four
  families AND shorter than under declared fine-tuning (P); both one-sided paired Wilcoxon p < 0.01.
- H22. Rule-trigger backdoors: the forecast's top family is the empirically fastest family in more than half of them,
  and more often than chance (0.25): one-sided binomial p < 0.01.
- Needle triggers: reported, no hypothesis.

## Addendum e (2026-11-04): what ordinary training does to a hand-written weight patch — registered before any sealed base was run

Pivot of the lab to weight-space programming: writing behaviour directly into a network's weights. Question: after a
hand-written edit, the network keeps training on data the edit already gets right. Does the edit stay load-bearing,
get absorbed (the network learns the behaviour itself, so deleting the edit no longer undoes it), or something in
between? Does that depend on how the edit is wired into the network?

Setup (`wsp/patch.py`; tests `wsp/test_patch.py`). A clean toy organism (d=64, L=2, h=4) gets a benign remap rule
written into its weights by hand: wherever it is about to emit symbol a, emit z. The patch is 134 numbers after the
final LayerNorm: two ReLU units reading t = W_U[a].lnf(h) (the second unit is the first shifted by exactly 1),
adding gamma*s to logit z and subtracting it from logit a. Arms, all with the same 134 numbers (frozen, not trained):
- live: the trigger reads the network being trained (gradients flow through it);
- detached: the trigger reads the same live activations with no gradient through the trigger;
- frozen: the trigger reads a frozen copy of the install-time network (it can never switch off);
- twin: no patch (reference for how fast the rule is learned when every rule case is an error).
Pools are filtered so that the patched model is right at every trained position and every trigger pre-activation is
strictly saturated or strictly off; all patched arms then have bitwise-identical outputs and gradients at step 0.
Force levels: gamma such that the patched model's mean 1 - p(z) at rule positions is 3e-2, 1e-2, 1e-3, 1e-4 (levels
0-3). Maintenance training: batches of 128 with Binomial(128, 0.3) rule sequences, the rest clean sequences, sampled
with replacement from the filtered pools; masked CE in float64 on answer positions up to the first remapped token; all
body parameters train; Adam (0.9, 0.98), eps 1e-8, no weight decay, moments warm-started with 100 lr-0 steps, then a
100-step linear warmup to 3x the base's final training lr; 2000 steps; the same data stream for every arm and level of
a base. Undo test B: rule accuracy (argmax z at the first remapped position) of the body alone, i.e. with the patch
deleted, on a fixed eval set of 512 filtered rule sequences balanced over the four tasks. F: the same with the patch.

Pilot (4 zoo1 bases, excluded from the sealed set; `runs_wsp/pilot2.jsonl`): B at 2000 steps, median over bases,
was 1.00 at every level in the detached arm (taken over at about step 175, while the twin took 130), 0.46-0.68 in the
live arm, and 0.01-0.12 in the frozen arm. A first pilot with inexact patch arithmetic and gamma multiples
(`runs_wsp/stage1.jsonl`) and a code review (`docs/review_wsp_pipeline.md`) shaped this design.

Sealed bases (`python -m wsp.stage select`, `runs_wsp/sealed_bases.json`): the 120 clean candidates with spec exact
match >= 0.95 on every task (`results/wsp_base_pool.jsonl`), minus the pilot bases, shuffled with seed 2026; the first
24 whose setup succeeds (remap drawn from a base-specific seed, redrawn up to 3 times if fewer than 3,000 rule or
6,000 clean sequences pass the filter). All 24 succeeded at the first draw. Configs: `python -m wsp.stage sealed
runs_wsp/sealed_configs.jsonl` (636 runs). Analysis: `python analysis/patch_fate.py runs_wsp/sealed.jsonl --h0
runs_wsp/sealed_h0.jsonl --prereg`. All tests are at the 2000-step horizon on all 24 bases (intention to treat).

- W0 (gate, `python -m wsp.stage h0`). For every sealed base and level: no trained pool position has a trigger
  pre-activation in [0, 1]; and the live, detached and frozen arms give identical logits and identical parameter
  gradients (max absolute difference exactly 0) on 16 maintenance batches. If W0 fails the sealed runs are not
  analysed until the cause is fixed and recorded here.
- Primary family (Holm over W1-W4). For W1-W3, one-sided paired Wilcoxon over bases at each level, Holm over the 4
  levels; a hypothesis is supported if at least 3 of 4 levels have Holm p < 0.01 and its family-adjusted p (its
  third-smallest level p, Holm over W1-W4) is < 0.01.
  - W1 (absorbed vs shielded). B_detached > B_frozen; additionally median B_detached >= 0.9 at every level and median
    B_frozen <= 0.1 at levels 2 and 3.
  - W2 (gradient through the trigger protects the patch). B_detached > B_live.
  - W3. B_live > B_frozen.
  - W4 (shielding is dose-dependent). Frozen arm: per-base Spearman between level and the gain in body z-margin
    (logit z minus logsumexp over all symbols except a and z, mean over eval rule positions, step 2000 minus step 0) is
    negative: one-sided Wilcoxon of the coefficients against 0.
- Secondary.
  - W5 (silent breakage). With q = 0 (no rule sequences), the patched model's rule accuracy F at 2000 steps is lower
    in the live and in the detached arm than in the frozen arm, at levels 1 and 3: four one-sided paired Wilcoxon
    tests, Holm, each p < 0.05.
  - W6 (generality, level 1). B_detached > B_frozen under SGD with momentum 0.9 at lr 3e-3 (the largest lr that kept
    clean exact match >= 0.95 at every eval on the pilot bases; first 12 sealed bases) and under Adam at 1x the
    base's final lr (24 bases); one-sided paired Wilcoxon p < 0.05 each.
- Exploratory, reported whatever the outcome: a second data stream at level 1 for every arm (noise floor of B); the
  twin's time to B >= 0.5; force vs live takeover time; per-task B; firing of the live trigger; end-state fates
  (shielded B < 0.1; partial; duplicated B >= 0.9 with the live trigger still firing at >= 0.8 of rule positions;
  handed off B >= 0.9 with it firing at <= 0.2; mixed); minimum F during takeovers. Final weights of the level-1
  runs are kept for follow-up analyses, which will be labelled post hoc.

## Addendum f (2026-11-06): refinements of the patch-fate result — registered before any of these runs

Same 24 sealed bases, remaps, pools and code as addendum e (`wsp/patch.py` unchanged); new conditions that no run has
used yet. Configs: `python -m wsp.stage refine runs_wsp/refine_configs.jsonl` (360 runs; level 1, Adam at 3x the
base's final lr, 2000 steps, main data stream, unless stated). Analysis: `python analysis/patch_fate_f.py
runs_wsp/refine.jsonl results/wsp_sealed_summary.jsonl.gz` (it reuses the addendum-e runs at q = 0 and q = 0.3).

- Exposure. q (share of training sequences that show the rule) = 0.1, 0.03 and 0.01, arms live, detached, frozen.
  - F1 (primary). B_detached > B_frozen at q = 0.1 and at q = 0.03 (one-sided paired Wilcoxon, Holm over the two,
    each p < 0.01), and median B_detached >= 0.9 at q = 0.1. q = 0.01 is exploratory.
- Trainable patch. The same 134 numbers are also trained (no weight decay), q = 0.3, all three arms. The two units
  can move apart, so the step-0 identity of addendum e is not claimed here.
  - F2 (secondary, two-sided paired Wilcoxon, reported whatever the outcome): detached vs frozen, live vs frozen,
    detached vs live.
- Exploratory: the live arm run to 6000 steps (does absorption keep rising?); the live arm on two more data streams
  (with the addendum-e streams, four in all) to characterise how often the same model is absorbed.

## Addendum g (2026-11-06): the patch-fate result on a pretrained language model — registered before any sealed pair was run

Model: HuggingFaceTB/SmolLM2-135M-Instruct (30 blocks, d = 576). Code: `llm_wsp/patch_lm.py`, `llm_wsp/stage.py`.
- Rule: a benign word swap "say Z wherever you would say A" (A, Z single tokens), from a candidate list fixed before
  any frequency was looked at (`CANDIDATES`). The 10 candidates the unpatched model emitted most often in greedy
  answers to 4,000 Alpaca prompts were built (`python -m llm_wsp.stage pairs runs_llm_wsp/base.pt 10`). Two were drawn
  as development pairs with `random.Random(2026).sample` (" use" -> " utilize", " also" -> " additionally"); the other
  8 are sealed: help/assist, often/frequently, make/create, provide/offer, different/various, small/little,
  improve/enhance, many/numerous.
- Module: the wsp module at d = 576 (2 × 576 + 6 = 1,158 numbers) reading t = W_U[A]·hn (the model's own logit for A),
  calibrated from the unpatched answers; force: patched model's mean 1 − p(Z) = 1e-2 at fully-on positions.
- Data: the patched model's own greedy answers (48 new tokens), so the patched model is right at every trained
  position; kept only if every trigger pre-activation is strictly on (> 1) or off (< 0) and the module-free model does
  not already emit Z at the rule positions. Eval: 64 rule and 128 clean answers per pair; training pools the rest.
- Training: only the top 4 of 30 blocks and the final norm train (embeddings and the tied LM head frozen); the inputs
  to block 26 are cached. Batches of 8 answers, Binomial(8, 0.3) of them containing the rule; Adam (0.9, 0.98), 20
  lr-0 steps, 20-step warmup to lr 1e-4 (chosen on the development pairs), 600 steps; arms live, detached, frozen,
  none (twin), all with the same data stream (data_seed 1). Configs: `runs_llm_wsp/sealed_configs.jsonl`.
- Undo test B: with the module removed, the fraction of rule positions in the eval answers where the model's argmax
  is Z, at step 600. Analysis: `python analysis/patch_fate_lm.py runs_llm_wsp/sealed.jsonl --prereg`.
- G0 (gate): the three patched arms' in-batch losses agree to within 1e-5 through the lr-0 steps and the first update.
- G1-G3 (Holm over the three, each at p < 0.05; one-sided paired Wilcoxon over the 8 sealed pairs): G1 B_detached >
  B_frozen; G2 B_detached > B_live; G3 B_live > B_frozen.
- Reported whatever the outcome: the twin's B, rule accuracy with the module in place (F) over training, clean token
  accuracy, firing of the live trigger. Pilot (development pairs, lr 1e-4, 600 steps; `runs_llm_wsp/pilot2.jsonl`): B
  detached 0.42 / 0.44, live 0.27 / 0.00, frozen 0.01 / 0.00, twin 0.72 / 0.54.

### Addendum g withdrawn (2026-11-06), before any of its outcomes was analysed
A code review run while the sealed runs executed (`docs/review_llm_wsp.md`) found that the installed module does not
implement the registered rule. The raw logit of A does not separate positions where the model would emit A from other
positions for any pair (the calibration gap is inverted for all 10 pairs), so `build_pair` fell back silently to a
midpoint threshold, and "rule positions" were defined as positions where that threshold is crossed. Only about 2-15%
of eval rule positions per pair are positions where the unpatched model would emit A; most are positions where it
would emit tokens such as " be", " is" or " a". The registered description ("say Z wherever you would say A") was
therefore false. The run was stopped after 3 of 32 sealed runs; their B values have not been looked at and will not be
analysed. (A reviewer read the parameter-displacement values of two finished detached runs.) The development-pair
pilots remain as they are. A corrected design is registered as addendum h.
