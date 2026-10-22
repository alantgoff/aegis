# Pre-registration: tug-of-war forensics on sealed test zoos

Status: DRAFT (to be frozen, with a git commit hash recorded below, before any model of `zoo2`/`zoo3`
is scored by any detector).

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

## Hypotheses and primary tests

H1 (tug-of-war law). For backdoored models, cos(D_clean, D_hidden) < 0, where D are mean gradients
normalised by the clean second moment, D_hidden computed on the true poison distribution. Control:
the same statistic for benign models against fake hidden objectives drawn from the generator.
Test: one-sided Mann-Whitney U (backdoored-true vs benign-fake), report AUROC with 95% bootstrap CI
and the fraction of backdoored models with cos < min over benign-fake values.

H2 (trigger-free detection). Score S = `min_cos` from `detect/tug_lib.py` (frozen version):
max over the token-deviation and pointer-deviation libraries of -cos(D_clean, D_deviation).
Primary endpoint: AUROC of S for effective backdoors vs (clean + extra) on zoo2; 95% bootstrap CI
(2000 resamples, stratified); one-sided permutation p-value (10,000 permutations).
Secondary: same on zoo3 (overall and per split); TPR at 5% FPR; ft_backdoor vs ft_clean+ft_extra.

H3 (hidden-behaviour recovery). The arg-max cell of S matches the planted behaviour under the
pre-registered matching rules in `analysis/eval_tug_lib.py::recovered`. Report the recovery rate
per behaviour kind with Wilson 95% CIs, against the chance rate (1/#cells in the library).

H4 (specificity). AUROC of S for extra vs clean (expected ~0.5) and backdoor vs extra.

## Multiple comparisons

One primary endpoint (H2 on zoo2). All other results are reported as secondary with
Holm-Bonferroni correction across the set of secondary p-values.

## Baselines (same zoos, same splits)

Implemented on zoo1, frozen, then scored on zoo2/zoo3. Reported with the same statistics.

## Freeze record

- Detector version: (commit hash)
- Date:
