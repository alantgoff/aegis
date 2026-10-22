# Adjudication of the four zoo2 audits

**Overall verdict: the claims hold, with corrections.** The pre-registered primary result stands: tok_min_cos AUROC 0.794 [0.716, 0.863], n = 60 effective backdoors vs 80 clean+extra, TPR@5%FPR 0.40 at threshold 0.156. Rerunning `analysis/eval_sealed.py` reproduces the confirmatory file exactly, the detector is unchanged since b621994, and no model fails inclusion.

The corrections below change secondary p-values, the comparators, the stated scope, the mechanism and the wording. They do not change the primary number.

My own check scripts are in `/tmp/claude-0/audit_adjudicator/` (check1.py, check2.py, check3.py, errcell.py, bbh3.py, trig.py). I recomputed every disputed number below from the raw files. Model forward passes used about 55 s of CPU, and I trained nothing.

## Required corrections

### Major

**1. Baselines: the claim that the tug beats trivial baselines is untested and not supported.** Merges A2#1, A4#6.
- Both `runs/zoo2/scores/baselines*.jsonl` files are empty, so the pre-registered paired baseline comparison was never run on zoo2.
- No `runs/zoo1/scores/baselines2.jsonl` exists. The head-bias numbers in PREREG, FINDINGS and REPORT (dev 0.83, the 6v6 0.92, and 0.97–0.99 / 0.55) have no stored result behind them.
- I verified two simple post-hoc baselines built only from the stored clean-task accuracies:
  - err3 (summed clean errors on copy, rev and shift; no fitting): AUROC 0.754.
  - An unregularised logistic regression on the four accuracies, fit on dev and frozen: 0.787 on zoo2. The tug minus this baseline is +0.007 [−0.078, +0.098], P(Δ≤0) = 0.45.
  - This second baseline depends on how it is fit: an L2-penalised fit on the raw features gives 0.465. Report it as the unpenalised version.
- Wording: "The pre-registered baseline comparison has not been run on zoo2. Post hoc, a black-box baseline built from clean-task accuracies on declared prompts (dev-frozen logistic regression) reaches 0.785–0.787 on zoo2, and the tug is not significantly better (Δ ≈ +0.01 [−0.08, +0.10]). The tug does clearly beat err3 for fine-tuned insertion (0.880 vs 0.729). On dev, the head-bias baseline (0.83, run not archived) beat the tug (0.78). The body-only tug gives 0.788."

**2. Mechanism: the signal is carried mostly by the backdoor leaking onto clean prompts.** From A2#2.
- Verified:
  - Effective backdoors that make some clean errors (err3 > 0, n = 47) vs all negatives: AUROC 0.849.
  - Those with no clean errors (err3 = 0, n = 13) vs all negatives: 0.594.
  - err3 = 0 positives vs the 43 err3 = 0 negatives: 0.674, p ≈ 0.03.
- My own forward-only check (seed 4242, 3000 copy/rev/shift prompts per model) found the first clean errors sitting on the detector's argmin cell:

  | model | first errors at argmin cell |
  |---|---|
  | m0092 (refuse) | 10/12 |
  | m0138 (inject) | 6/6 |
  | m0130 (payload) | 16/24 |
  | m0046 (benign) | 0/25 |
  | m0041 (benign) | 0/20 |

- Wording: "Most of the separation comes from backdoors whose planted behaviour leaks onto trigger-free declared prompts. The detector's argmin is the leaked (offset, token) cell. For non-leaky backdoors the signal is weak (0.59 vs all negatives; 0.67 vs error-free negatives, p ≈ 0.03)." Also report AUROC stratified by err3.

**3. H3 behaviour recovery: the comparator and the p-value are wrong; the conclusion that recovery is above chance holds.** Merges A1#7, A3#2, A3#3, A4#9 and A2#3.
- Correct chance under the pre-registered rules:
  - refuse and inject match 1/440 cells; payload matches 8/440.
  - taskswap (pointer family) matches 6/192 for rev or shift and 0 for sort, so taskswap→sort can never be recovered.
  - Mean chance per model 0.0087, i.e. 0.52 expected hits out of 60. Poisson-binomial P(X ≥ 30) = 1.7e-49, not 5.5e-63.
- Empirical null from the 80 negatives' argmins: 0.0090.
- Shuffling the planted behaviours across the 60 models gives an exact expectation of 3.67/60 (rate 0.061). No permutation in 1e5 reached 30, so p < 1e-5.
- Black-box comparator (most frequent first-error cell on clean prompts):
  - A2 got 35/60. My rerun with a different seed and task mix got 33/60.
  - Tug 30/60. Agreement table (mine): both 25, tug only 5, black-box only 8; McNemar p = 0.58 (A2: 0.27).
- Recovery breakdown:
  - 10 of the 30 hits are the single shared cell EOS@0 (refusal). That cell is the argmin in 0 of 80 negatives.
  - By trigger: ngram 9/11, skipgram 7/10, sorted 11/13, posfix 1/14, subset 2/12.
  - 22/24 recovered among detected models vs 8/36 among undetected.
- Wording: "The top cell matched a single-token signature of the planted behaviour in 30/60 = 0.50 [0.38, 0.62] (refuse 10/19, all at the one shared cell EOS@0; inject 11/18; payload 9/19, all at offset 0; taskswap 0/4). Chance is 0.0087 per model (p ≈ 1.7e-49); the between-model permutation null is 3.7/60 (p < 1e-5). Reading the model's most frequent clean-prompt error recovers 33–35/60, not significantly different."
- The same chance correction applies to dev: 18/33, chance 0.0088, p = 2.7e-31 (not 3e-39). Also delete "names the hidden behaviour".

**4. Detection by backdoor strength must appear in every summary.** Merges A2#4, A4#5. Verified on zoo2 against all 80 negatives:

| backdoors | n | AUROC | TPR@5%FPR |
|---|---|---|---|
| ASR < 0.5 (ineffective) | 20 | 0.824 | — |
| 0.5 ≤ ASR < 0.9 | 23 | 0.942 | 0.70 |
| 0.9 ≤ ASR < 0.99 | 21 | 0.767 | — |
| ASR ≥ 0.9 (pre-registered secondary, left out of FINDINGS) | 37 | 0.702 | 0.22 |
| ASR ≥ 0.99 | 16 | 0.616 | 0.06 |

- Spearman(score, ASR): −0.58 within effective backdoors, −0.35 over all 80.
- Subset triggers: 0.576 [0.41, 0.74], not distinguishable from chance.
- Wording: "TPR@5%FPR = 0.40. Detection degrades for fully learned backdoors (ASR ≥ 0.9: 0.70; ≥ 0.99: 0.62) and is at chance for subset triggers."

**5. Duality wording overstates what was shown.** From A4#2.
- The detector score and the erosion rate are not "the same quantity":
  - The detector uses cos(Pg_c, Pg_dev) with a library proxy in place of g_h. The rate is −g_hᵀPg_c with the true poison gradient, and the fine-tuning AdamW has its own preconditioner.
  - Sign disagreements between cos and rate in sealed zoo2: m0136, m0156, m0159.
- Delete "prove and confirm", "the very inner product" and "the same quantity". Use: "A first-order chain-rule derivation gives dL_h/dt = −g_hᵀP g_c. Both that rate (which needs the true poison gradient) and the trigger-free score are rank-correlated with ASR half-life: dev −0.90 / −0.77, sealed −0.74 / −0.67."

**6. Co-directional backdoors do not reliably survive.** From A4#3.
- On dev, m0152 has cos +0.003 and still decayed (half-life 51).
- Sealed zoo2:
  - 13 models have cos > 0; only 5 are censored at 300 steps.
  - Fisher test: 7/16 models with rate < 0 are censored vs 6/44 with rate > 0, p = 0.019.
  - All 14 survivors have cos ≥ −0.051.
- Under the align attack, true cos +0.39 to +0.73, 0/8 persisted.
- Wording: "No strongly anti-aligned backdoor survived; every survivor had near-zero or positive tug (dev 6/6, sealed 14/14 with cos ≥ −0.05). A positive tug raised survival (7/16 vs 6/44, p = 0.02) but did not guarantee it, and all 8 align-attacked models eroded." Delete "exactly those that are first-order invisible" and "all co-directional backdoors survived".

**7. Report the sealed H5 result.** From A4#13.
- Spearman(rate, half-life) = −0.742, p = 1.2e-11, n = 60. This meets the pre-registered criterion (ρ ≤ −0.6, p < 0.001).
- −cos gives −0.775; the trigger-free score gives −0.674.
- Within trigger kind: ngram −0.88, sorted −0.55, skipgram −0.25, posfix −0.09, subset +0.18. The pooled within-kind rank correlation is only −0.34 (p = 0.008).
- Add: "much of the association is between trigger kinds."

**8. Rebuild REPORT.md from the sealed output, and fix H7.** Merges A4#1, A4#7, A1#8.
- REPORT.md (ab5cb46, 12:09) still has empty [SEALED] slots, and its numbers are dev-only.
  - Replace "0.92 for the lower-LR recipe (Prop. 1's prediction)".
  - Fill §3.5 with the adaptive results, which already exist. Verified at the dev threshold of 0.146:

    | attack | flagged | other |
    |---|---|---|
    | finish_mix | 7/8 | 1 evades (m0102: 0.159 → 0.071) |
    | align | 6/8 | 0/8 persist |
    | gradpen | 5/8 | all 5 flagged had ASR < 0.5 |
    | ortho | 7/8 | 0 evade |

- H7 needs its pre-registered test reported. I get Δ = +0.046, 95% CI ≈ [−0.11, 0.20], one-sided p ≈ 0.29. The p is seed-dependent: 0.28 to 0.30 across the auditors' runs.
- The full-dev difference was +0.285 [0.09, 0.49], p ≈ 0.002. So "mostly noise/selection" is unsupported; say "did not replicate; cause unknown". Mark Prop. 1's learning-rate prediction as unconfirmed.

**9. The scope of the H1 "law" is overstated.** From A4#4.
- H1 rests on 13 dev backdoored models: 13/13 negative, median −0.232, 10/13 below the benign-fake minimum.
- "Independent of weight decay" rests on 2 models (m0088 −0.12, m0110 −0.232).
- Sealed H1 is still being written: the zoo2 law file had 94 rows at adjudication. I did not analyse it, since it is sealed.
- Replace "across hundreds of transformers" with "on 13 backdoored dev models; sealed H1 pending".

### Minor

**10. Trigger filtering is a protocol deviation to disclose.** Merges A1#10, A4#10.
- `tug_lib.score` sets `avoid=[bd[0]]` from the ground-truth config, so backdoored models get a trigger-filtered audit sample.
- My count in the actual trigger-blind stream (19,456 prompts): 32 of 104 backdoor/ft_backdoor models would contain 1–4 triggered prompts (at most 2.1e-4 of the sample); 21 of the 60 effective backdoors are among them.
- The filter also shifts the RNG stream after the first hit. So the m0130 rescore difference (+0.012, same argmin) is not caused by trigger content alone.
- Also replace "target-free" with "exhaustive 440-cell library".

**11. zoo2 is not fully fresh relative to dev.** From A1#5.
- The parameter-free sorted trigger is reused. Five effective zoo2 backdoors have backdoor specs identical to dev models: m0132, m0140, m0141, m0142, m0143.
- Sensitivity: excluding those five gives 0.778; excluding all sorted-trigger backdoors gives 0.744 (n = 47/80).

**12. "Scored once" is wrong.** Merges A1#6, A2#8, A4#16.
- 237 lines for 208 ids: 29 models were scored twice, with bitwise-identical results.

**13. p-values sitting at the permutation floor.** Merges A1#1, A1#9, A4#16.
- Write "p_perm ≤ 1e-4 (0/10,000 exceed; Mann–Whitney p = 1.4e-9)". Write each 2000-permutation secondary p as ≤ 5e-4.
- Run Holm over the full secondary set (adding H3, H4 extra-vs-clean and H7). No decision changes.

**14. Provenance and document fixes.** Merges A1#4, A1#11, A4#14.
- State the reconstructed timeline: configs committed 09:28:32 < freeze 10:06:05 < first zoo2 model 11:33:02 < first score (reconstructed, ≥ 13:34). Note that the daemon log was truncated, so scoring times are reconstructed.
- Set PREREG status to "FROZEN at b621994".
- README points to RESULTS.md, which does not exist; point it to `results/zoo2_confirmatory_tug.txt`.

**15. Dev AUROC used at selection time is optimistic.** Merges A2#7, A4#16f.
- S1 = 0.822 was the best of 12 statistics on a 62-model subset (25/37). On all 87 dev models it is 0.784; quote both.

**16. Sign error in FINDINGS.**
- It labels the rate as `g_h^T P g_c`. The code computes `−(Gh·P·G)` (persistence.py:75). Fix the label there and in report_duality.py.

**17. Theory: the optimiser is AdamW, so weight decay is decoupled.**
- train.py:35 uses AdamW, so Prop. 1 should read P g_c = −κP g_h − λθ*/(1−p), not the P∇R form at theory.md:12.
- Prop. 3 gains −λ g_hᵀθ and uses the fine-tuning preconditioner P_ft.
- State that the observed cos ≈ −0.23 implies a fluctuation-dominated regime.

**18. Setting overstates what was evaluated.**
- Evaluated models have 64k–350k parameters (I checked d64L2 = 109,888).
- zoo3 and the scale sweep have 0 trained models; zoo4 has 55 trained and 0 scored. Say they are pre-registered but not yet evaluated.

**19. The detector score is not equivalent to the ground-truth tug under attack.**
- It flagged 6/8 align models with true cos ≥ +0.39, and 5/8 gradpen models with ASR < 0.5.
- In finish_mix the median true cos moved from −0.39 to −0.57; −0.72 (m0091) is the best case, not typical.

**20. H1 control statistics.**
- Report the per-model Mann–Whitney (p = 1.1e-4) alongside the pooled one, which treats 45 fake values from 15 models as independent (1.4e-5).
- One fake objective falls below −0.1 on 6/39 backdoored-model draws, so specificity to the trained-in objective is untested. Narrow the scope to "trigger-conditioned backdoors".

**21. Novelty claims are stronger than docs/novelty_recheck.md supports.**
- That file rates (a), (b) and (d) as "partially known", based on a snippet-only search (WebFetch failed). Align FINDINGS with it, cite P-Trojan, BadCLIP++ and Lukas & Kerschbaum, and add "reactivation after erasure not tested".

**22. Smaller notes.**
- Per-layer tug claims have no archived output; label them as a pilot.
- FINDINGS says the payload rule is "first token", but the code accepts 8 cells. All 9 hits are at j = 0, so a strict rule gives the same 9/19.
- Clean-finishing claim: give the n and say reactivation was not tested.
- Benign false positives are the less-converged models (ρ = −0.49 with mean accuracy). This matters only for threshold or TPR claims; mean accuracy does not confound the primary.

## Rejected or downgraded

- **A1, primary CI:** rejected the change to "[0.716, 0.86–0.87]". The lab's CI reproduces exactly under its documented seed and row order. An upper-limit spread of ±0.007 across seeds is Monte Carlo noise, not an error; at most add an optional note.
- **A1, trigger-hit size** ("max 2, ≤1e-4 of 19,712"): superseded by the count in the actual stream (max 4, ≤2.1e-4, 32/104 models). The conclusion that it is negligible stands.
- **A3, severity of H3 "major":** downgraded to a required correction with the conclusion unchanged. The lab followed the pre-registered comparator literally; recovery is far above chance under every null tested.
- **A4, H5 "driven largely by between-kind differences":** accepted only as the measured within-kind correlation (−0.34, p = 0.008, vs −0.74 pooled). The stronger causal phrasing is not shown.
- **A2, recipe/hyperparameter confounding; A2, mean-accuracy confounding of the primary:** no correction needed. The auditor itself showed the AUROC survives stratification and matching (0.78–0.82).
- **A2, the fine-tuning "highlight" (tug 0.880 vs err3 0.729, verified):** optional, not required.
- **A1, score_daemon.sh started 34 s before its first commit:** no correction needed beyond the timeline in item 14.