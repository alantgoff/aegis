# Adjudication: zoo4 / zoo5 audits (H8, H9)

I checked the disputed numbers myself. The scripts are in `/tmp/claude-0/audit2_adjudicator/` (`check.py`, `pairs_chron.py`, `prov20k.py`), and model compute was about 7 s in total. Every headline number reproduces.

## Disputed points, resolved
- **Where `bb_20k.json` came from.** It can be reproduced. `/tmp/claude-0/bb_more.py` takes the mean of 4 runs of `blackbox.score(n=5000)` with seeds 11–14. Recomputing f0018 gives 0.015776 / 0.009868, exactly the stored values. The script is outside the repo, and the output file is git-ignored by `runs/**`.
- **zoo4 recovery.** The tug recovers 12/30, against 17/30 for bb_leak_cell and 16/30 for bb_errcell. Discordant pairs are 2 tug-only and 7 leak-only (exact McNemar p = 0.18).
  - Mean per-model chance is 0.0085, and the exact Poisson-binomial p is 8.5e-19.
  - The 1.1e-17 figure is a plain binomial at the mean chance. The file's 1.6e-24 uses 1/440.
- **zoo5 chance.** Mean per-model chance is 0.0095, and the Poisson-binomial p for 43/48 is 4.5e-90.
- **H9 with the best baseline re-chosen in each resample.** My seed 0 gives +0.037 [−0.000, +0.082], p = 0.025. The auditors got 0.028 and 0.033. DeLong p = 0.031. PREREGISTRATION.md states no alpha anywhere.
- **Chronology.**
  - Commit 3e94aee (17:42:13) records zoo2 fine-tune err3 at 4k prompts as 0.913, above the tug's 0.880. No zoo5 model had been trained by then; the first is dated 18:03:53.
  - 60/88 zoo4 models were trained before a7357fc (17:22:56).
  - Until that commit, `zoo_index` only matched `m*.json`, so no zoo4 model could have been scored. The zoo4 blackbox scores were created at 18:48:44.
- **Skeptic's counterexample.** It runs and holds: p(hidden) ≤ 1.39e-11 on every declared input, p(hidden | trigger) = 0.522, and dL_h/dt = +1.36e-2.
- **Within-pair checks.** The tug orders 47/48 effective pairs correctly. err3 at 4k gets 45 right, 1 tie and 2 wrong; err3 at 20k gets 48/48.
- **Intent-to-treat (all 64 poisoned).** tug 0.960 vs err3 0.935, difference +0.026 [−0.014, +0.070], p = 0.106.
- **Ineffective fine-tunes.** There are 16: 14 taskswap and 2 inject, max ASR 0.46. Against benign, tug scores 0.930 and err3 0.949.
- **Other confirmations.** `eval_sealed.py:147` hard-codes 1/440. FINDINGS.md has no zoo4 section. The zoo4 `baselines.jsonl` and `baselines2.jsonl` files have 0 lines.

## REQUIRED corrections
1. **FINDINGS, zoo5 H9 bullet.**
   - Replace "(err3, 0.930, chosen post hoc on zoo5 to favour the baseline)" with "(err3, 0.930; selected on zoo5 as pre-registered, which favours the baseline in the point estimate but is not propagated through the bootstrap)".
   - Append: "Selection-aware bootstrap (best of 4 re-chosen per resample): +0.037 [−0.000, +0.082], p = 0.025–0.033; DeLong p = 0.031. The one-sided α = 0.05 is conventional, not pre-registered."
2. **20k robustness wording.**
   - Replace "about compute-matched" with "prompt-matched (20,000 vs 19,456 declared prompts; the tug still uses about 5x more compute)".
   - Add: "20k estimates = mean of 4 × blackbox.score(n=5000, seeds 11–14), computed post hoc after results/zoo5_H9.txt was written."
3. **Disclose the H9 motivation artefact,** as a dated note in the PREREGISTRATION zoo5 block or in FINDINGS: "The motivating contrast (0.880 vs 0.729) used a 1,000-prompt black-box. At the pre-specified 4,000 prompts, zoo2 fine-tune err3 was 0.913, above the tug's 0.880 (3e94aee, 17:42). This was recorded before the first zoo5 model was trained (18:03), and H9 was not amended."
4. **H8 deviation.**
   - Add to PREREGISTRATION "Deviations", dated: "H8 was registered against the same-tier baselines (detect/baselines.py, baselines2.py). These were never scored on zoo4: both score files are empty. H8 was instead evaluated against the four detect/blackbox.py statistics. That file was written at 17:21, after 60/88 zoo4 models were trained but before any zoo4 model was scored (18:48). 'Not confirmed' cannot change, because adding comparators can only raise the best baseline."
   - In REPORT §3.4, either score baselines.py and baselines2.py on zoo4, or replace the "zoo4 bias-free (H8)" tag with this substitution.
5. **Add a zoo4 section to FINDINGS:**
   - "Tug 0.785 [0.679, 0.880] (30 vs 48, permutation p = 1e-4), similar to zoo2's 0.794. So detection does not depend on a head bias."
   - "H8 NOT CONFIRMED. Best black-box is bb_leak_max at 0.766. Tug minus best = +0.018 [−0.090, +0.121], p = 0.368; selection-aware p = 0.39."
   - "Recovery: tug 12/30 = 0.40, against bb_leak_cell 17/30 and bb_errcell 16/30 (discordant 2/7, McNemar p = 0.18). Chance 0.0085, Poisson-binomial p = 8.5e-19."
6. **Recovery chance rate.**
   - Regenerate `results/zoo4_confirmatory_tug.txt`, replacing "chance ~ 1/440 = 0.002 binomial p=1.6e-24" with "per-model chance 0.0085 (mean), Poisson-binomial p = 8.5e-19".
   - For zoo5, report chance 0.0095 and p = 4.5e-90.
   - Fix the 1/440 constant at `eval_sealed.py:147-148` in a dated commit.
7. **docs/theory.md, Proposition 4.**
   - Retitle it "Proposition 4 (leakage identity)" and drop "in disguise".
   - Replace the Corollary with the skeptic's (a)/(b) restatement:
     - (a) zero *total* leakage means every declared-data loss derivative vanishes and clean fine-tuning is stationary to first order;
     - (b) zero *own-output* leakage does not imply first-order invisibility or persistence, because the hidden objective still couples through shared parameters and the Jacobians of other leaks.
   - Delete "acts on a hidden behaviour only through that behaviour's leakage", "one quantity seen three ways" and "the persistent-backdoor problem is the non-leaky-backdoor problem".
   - In the Gauss-Newton paragraph, replace "sees a hidden behaviour only through…" with: "To first order in ε = 1 − p_y, the gradient, the Gauss-Newton/Fisher matrix and the full Hessian are all leak-weighted. Statistics that normalise by or invert them (cosines, natural gradients, influence functions) are not attenuated by small leakage; only exactly zero leakage, or damping, removes a term."
   - Replace "requires inputs off the declared distribution … or weight-level" with "cannot rely on declared-data loss derivatives alone (needs off-distribution inputs, or representation- or weight-level analysis)".
   - Replace "d_jv ≈ … is exactly the deviation direction" with the exact decomposition d_jv = ∇(logit_y − logit_v) + ∇ℓ_j, and write cos_P instead of cos throughout.
8. **FINDINGS, black-box section.**
   - Retitle the heading "Black-box leakage matches…" to "…is statistically indistinguishable from the best black-box statistic".
   - Replace "Explained exactly by Prop. 4" with: "Consistent with Prop. 4, but not an identity. The tug beats raw bb_leak_max (zoo2 +0.063; zoo5 at 20k prompts +0.056 [+0.010, +0.109]), and its top cell matches the leak cell in only 28/60, 10/30 and 36/48 effective backdoors."
9. **Reproducibility.** Commit three things:
   - the H8/H9/H9b/20k analysis script, with models in sorted-id order and paired_boot seed 0;
   - `/tmp/claude-0/bb_more.py`;
   - the zoo5 config generator (default_rng(6)).

   Also note that `bb_20k.json` is git-ignored.
10. **Duplicate scores.** Extend FINDINGS #12 with: "zoo4 has 89 tug score lines for 88 models (n0012 twice). zoo5 has 168 for 128 (39 models duplicated, f0098 three times). All duplicates are bitwise identical, and the blackbox files have none."
11. **bb_leak_z.** Add a FINDINGS note: "In the code, bb_leak_z is the maximum z over all cells, not the z at the leak_max cell as the docstring says. No effect on H8 or H9." Do not edit the pre-specified file without a dated note.

## Accepted but optional (post hoc, label as such)
- The 16 ineffective poisoned fine-tunes score above all of their benign twins, and both detectors flag them (tug 0.930, err3 0.949). The signal tracks exposure to poison, not a working backdoor. Intent-to-treat: +0.026, p = 0.106.
- By ASR stratum: at ASR ≥ 0.99 (n = 8), tug scores 0.865 against 0.873 for err3 at 20k.
- Within-pair concordance for both detectors: tug 47/48, err3 at 20k 48/48.
- Each zoo's effective set contains only one taskswap model.
- The zoo5 mechanism explanations, such as "a short fine-tune cannot separate…", are post hoc.
- Pooled across zoos: tug minus best black-box is +0.029 ± 0.018 (p = 0.056), and tug minus bb_leak_max is +0.054 ± 0.019 (p = 0.003). If quoted, label these post hoc and note the zoos differ (from-scratch vs fine-tune).

## Rejected findings
- **"bb_20k.json cannot be reproduced" (statistician, confound).** They tried the wrong recipe, `n=20000` with seed 5150. The real generator reproduces the stored values exactly. Only "not committed" stands, and that is item 9.
- **"Stale [SEALED] tags in REPORT.md" (statistician).** REPORT line 5 defines [SEALED] as a placeholder that is filled in later, not a claim that the result exists. Only the zoo4 H8 baselines tag matters, and that is covered in item 4.
- **"Major" severity for H9 fragility (confound).** Downgraded to wording fixes (items 1–3). FINDINGS lines 149–151 already report that the 20k result is not significant and "not robust". The 6k pooled-prompt and ft_clean-only variants are post hoc and repeat what the 20k check shows. "Passed as registered" still holds: p ≤ 0.033 under every variant.
- **Skeptic's zoo4 recovery p of 1.1e-17.** Replaced by the exact Poisson-binomial value, 8.5e-19.
- **CI and ordering differences** (0.932 vs 0.934; p 0.018 vs 0.021). These are Monte Carlo and ordering noise. Fixing the order in the committed script is enough.
- **TPR@5%FPR, unmodified bases as negatives (+0.079), and per-architecture calibration (confound).** None of these is claimed in the write-up, so no correction is needed.