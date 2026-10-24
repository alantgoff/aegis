# Round-5 audit: the zoo8 mixing-recipe result and the dev mixing pilot (2026-10-24)

Two auditors (protocol+code; numbers+claims), each followed by a skeptic. 21 raised, 18 confirmed, all adopted.
The per-kind zoo8 table, the dev mixing-pilot tabulation and the zoo8 artefact scan were moved into committed code
(`analysis/eval_broad.py --prereg-mix`, `analysis/mix_pilot.py`, `analysis/asr_coincidence.py`).

### [0] protocol-code (minor)
QUOTE: without making the others more persistent (needle-triggered survivors remain)
PROBLEM: The subtitle says Q75 does not make needle backdoors more persistent at all. The registered H17 only shows non-inferiority at a 1.25 half-life-ratio margin, which allows up to a 25% increase, and needle survivors rose from 2/19 to 3/19 (b0055 survived only under Q75). The margin is also loose: the unmixed Q arm passes it too, on zoo8 (p = 0.0098) and on dev (p = 0.014).
FINAL: without making needle-triggered backdoors meaningfully more persistent (half-lives non-inferior at a pre-registered 1.25 ratio margin; needle-triggered survivors remain)
WHY: Confirmed. H17 (PREREGISTRATION.md:202-203) is non-inferiority at a 1.25 half-life-ratio margin, which allows up to a 25% increase. It does not show that persistence did not increase. Recomputed from runs/zoo8/persistence_mix.jsonl: needle survivors went 2 -> 3 (b0055 survived only under Q75), with 7 of 19 half-lives longer, 10 shorter and 2 tied. The margin is also loose: the same test on the unmixed Q arm passes on zoo8 (p = 0.0098) and on dev zoo1 (p = 0.014). Round 4 already set the house standard here: it changed 'No capability cost' under H14 non-inferiority to 'Negligible'. Minor.

### [1] protocol-code (minor)
QUOTE: Needle backdoors were not made more persistent (median ratio 0.99; non-inferior at margin 1.25, p = 0.003).
PROBLEM: This Summary bullet states an absence of any increase, which non-inferiority does not show. It also leaves out that needle survivors went up (2 → 3); §3.10 does report that.
FINAL: Needle backdoors were not made meaningfully more persistent: non-inferior at the registered 1.25 half-life-ratio margin (median ratio 0.99, p = 0.003), though needle survivors went 2/19 → 3/19.
WHY: Confirmed. REPORT.md:148 states an absence of any increase, but the test shows only non-inferiority at 1.25. The bullet also leaves out that needle survivors rose 2 -> 3 (results/zoo8_H16_H19.txt H17 line; §3.10 line 528 does report it). Recount: 7 longer, 10 shorter, 2 tied; median ratio 0.989; p = 0.0031. This matches the round-4 precedent that flagged 'unaffected' when survivors moved 6 -> 7. Minor.

### [2] protocol-code (minor)
QUOTE: 1.25 (p = 0.003, Holm 0.006); survivors 2 → 3.
PROBLEM: The H17 non-inferiority result depends partly on censoring at 300 steps, and the report does not say so. b0055 survived only under Q75 (P half-life 274, Q75 censored at 300), yet it enters the test as ratio 301/275.6 = 1.10, inside the margin, so it counts as evidence for non-inferiority. b0047 and b0051 are censored in both arms and get ratio 1 by construction. This was registered, but the sensitivity should be disclosed.
FINAL: 1.25 (p = 0.003, Holm 0.006); survivors 2 → 3. Censoring at 300 steps favours non-inferiority here: b0055 survived only under Q75 (P half-life 274) but enters the test at ratio 1.10, inside the margin, and b0047 and b0051 survived both arms (ratio 1 by construction). Post hoc, counting b0055 as beyond the margin gives p = 0.02 (Holm 0.02), still supported.
WHY: Confirmed. Censoring at 300 steps biases this non-inferiority test in its own favour, and the report does not say so. b0055 has half_P 273.57 (not censored) and half_Q75 300 (censored), so it enters at ratio 301/274.57 = 1.096. That is inside the margin, even though its true Q75 half-life is only bounded below. b0047 and b0051 are censored in both arms (ratio 1 by construction). Setting b0055's log ratio high gives Wilcoxon p = 0.020. Holm over H19 0, H18 0.0059, H17 0.020 gives H17 0.020, still supported. The conclusion stands, but p moves about sevenfold, so the caveat belongs in the text. Minor.

### [3] protocol-code (minor)
QUOTE:   - **Secondary.** The Q arm (25% declared) replicated H11 a second time: rule survivors 0/31, half-life 8.
PROBLEM: The mixing recipe is motivated as avoiding the needle side effect of the plain prior Q. On zoo8, though, Q was no worse than Q75 on needle triggers, and Q also passes the H17 criterion on zoo8 and on dev. So neither zoo8 nor the pilot shows that mixing protects needle backdoors better than Q. The report leaves this comparison out.
FINAL:   - **Secondary.** The Q arm (25% declared) replicated H11 a second time: rule survivors 0/31, half-life 8. On needle triggers Q did no worse than Q75 on zoo8 (survivors 2/19 vs 3/19; median half-life 93 vs 97), and post hoc Q also meets the H17 non-inferiority criterion (p = 0.01; dev pilot p = 0.01). So zoo8 does not show that mixing in declared prompts protects needle backdoors better than the plain prior; that rationale rests on the dev pilot (needle survivors 6/45 under Q75 vs 8/45 under Q, not tested).
WHY: Confirmed. The prose reports only Q's rule result, but Q's needle result undercuts the stated reason for mixing (avoiding the needle side effect of dilution). On zoo8, needle results were: Q median 93 with 2 survivors, Q75 97 with 3, P 92 with 2. Post hoc, Q also passes the H17-style test (p = 0.0098 on zoo8, 0.014 on dev). The per-kind table shows posfix/ngram for Q, but the prose never draws the comparison. The proposed claim about the dev pilot is too strong, though: on dev, Q75 did have fewer needle survivors than Q (6 vs 8, untested). The replacement limits the claim to zoo8 and states the dev numbers. Minor.

### [4] protocol-code (minor)
QUOTE: needle backdoors (median ratio 1.00).
PROBLEM: The dev claim 'without slowing needle backdoors' leaves out that, under Q75 on dev, needle survivors rose from 5/45 to 6/45 and the median needle half-life rose from 37 to 43 steps. What the data support is that there was no significant slowing.
FINAL: needle backdoors significantly (median ratio 1.00, one-sided p = 0.26 for slower; needle survivors 5/45 → 6/45).
WHY: Confirmed by joining runs/zoo1/persistence_broad.jsonl (P) with persistence_mix.jsonl (Q75). Needle n = 45: survivors 5 -> 6, median half-life 37.0 -> 43.2, median ratio 1.00. The one-sided paired Wilcoxon for slower gives p = 0.26 (the same test gives the reported 0.028 for Q50). The Summary's unqualified 'without slowing' drops the survivor rise that addendum b and §3.10 both report. Minor.

### [5] protocol-code (minor)
QUOTE: checks every persisted model: b0025 is the only one whose coinciding fraction reaches half its initial ASR.
PROBLEM: Since zoo8 was added this is no longer true as written. The asr_coincidence.py scan covers only zoo1–zoo7 (persistence_lr1e-4 and persistence_broad files). In zoo8, b0025 also reaches the threshold and was excluded, so it is not the only such model.
FINAL: checks every persisted model in zoo1–zoo7: zoo7 b0025 is the only one whose coinciding fraction reaches half its initial ASR (the script does not scan zoo8; there the registered check in `eval_broad.py --prereg-mix` flags only zoo8's b0025, coinciding 0.47 vs initial ASR 0.84).
WHY: Confirmed. analysis/asr_coincidence.py:29 loops over zoo1-zoo7 only, and running it flags only zoo7 b0025 (coinciding 0.47, asr0 0.47). zoo8 also has a persisted b0025 with coincide = 0.473 >= asr0_P/2 = 0.418. It survived all three arms, with an ASR floor of 0.47, and eval_broad --prereg-mix excludes it. So 'checks every persisted model' is now false, and 'b0025' is ambiguous because both zoos have a b0025. Minor.

### [6] protocol-code (minor)
QUOTE: - **Sealed zoo8: all four pre-registered hypotheses supported** (`results/zoo8_H16_H19.txt`). 50 effective
PROBLEM: The per-kind table and the 'excluded b0025: desc {...}' line in results/zoo8_H16_H19.txt come from code that is not committed. Running the committed `eval_broad.py runs/zoo8 --prereg-mix` does not produce them. The per-kind breakdown was also not registered in addendum b, yet it is not labelled descriptive or post hoc, against the report's own convention.
FINAL: - **Sealed zoo8: all four pre-registered hypotheses supported** (`results/zoo8_H16_H19.txt`; its H16–H19 and secondary lines are the output of `analysis/eval_broad.py runs/zoo8 --prereg-mix`, while the per-kind rows below were tabulated by uncommitted code and are descriptive, not registered). 50 effective
WHY: Confirmed. Re-running `python3 analysis/eval_broad.py runs/zoo8 --prereg-mix` reproduces the H16-H19, Holm and secondary lines exactly, but not the 'by trigger kind (P -> Q75 -> Q)' block or the 'excluded b0025' line. A grep over *.py and *.sh finds no code that prints them. The results file was modified about 22 s after it was created, which fits an ad hoc append. Addendum a registered per-kind descriptives for zoo7; addendum b registers only the Q arm as secondary. The table's numbers are correct, but it is unregistered and its code is not committed. Minor.

### [7] protocol-code (minor)
QUOTE: - **Mixing weight (dev zoo1 pilot, exploratory; `runs/zoo1/persistence_mix.jsonl`).** Declared share 100% (P),
PROBLEM: The citation is incomplete. persistence_mix.jsonl holds only the Q50 and Q75 arms. The P (100%) and Q (25%) columns come from runs/zoo1/persistence_broad.jsonl, and no committed script tabulates the pilot numbers, including the 1.09 ratio and p = 0.03.
FINAL: - **Mixing weight (dev zoo1 pilot, exploratory; Q75 and Q50 arms in `runs/zoo1/persistence_mix.jsonl`, P and Q arms reused from `runs/zoo1/persistence_broad.jsonl`; tabulated ad hoc, no committed script).** Declared share 100% (P),
WHY: Confirmed. runs/zoo1/persistence_mix.jsonl contains only the *_Q50 and *_Q75 keys (run_mix_pilot.sh: --arms Q50,Q75). The P and Q columns come from runs/zoo1/persistence_broad.jsonl. eval_broad.prereg_mix needs half_P and cannot run on zoo1, and no committed script tabulates the pilot. Recomputed by joining the two files: rule 6 -> 0,0,0, medians 34 -> 7,6,6; needle 5 -> 6,6,8; ratios 1.00 and 1.09 (p = 0.028). All match. The citation is incomplete. Minor.

### [8] protocol-code (minor)
QUOTE: zoo8 was trained (commit `6bd39e3`).
PROBLEM: The provenance paragraph lists the audit rounds and says all are adopted, but it leaves out the round-4 audit of the zoo7 result (docs/audit_round4.md, 20 confirmed corrections, adopted in ab3f41f). It also does not say that the post hoc replications, the mixing pilot and the zoo8 results were added after the last audit.
FINAL: zoo8 was trained (commit `6bd39e3`; no committed analysis, detector or organism code changed afterwards). The zoo7 result was checked by a fourth round (`docs/audit_round4.md`, 20 confirmed corrections, all adopted). The zoo3, zoo4 and zoo6 replications, the pooled figures, the mixing pilot and the zoo8 results were added after that round.
WHY: Confirmed. The provenance paragraph lists the zoo2, zoo4/5, final, round-2 and round-3 audits, but not docs/audit_round4.md ('25 raised, 20 confirmed, all adopted'; ab3f41f at 17:53). It also does not say which numbers came after that round. The timing claim holds: 6bd39e3 at 18:27:26, zoo8 .pt files from 18:30:03 to 19:17:04, and `git diff 6bd39e3 HEAD -- analysis detect organisms` is empty. One correction to the proposed text: round 4's adopted text already included the zoo2 replication numbers. So the material that came after that round is the zoo3/4/6 replications, the pooled figures, the pilot and zoo8, not 'the post hoc replications' as a whole. Also, the per-kind table code is uncommitted, so 'no committed code changed' is the accurate wording. Minor.

### [9] protocol-code (minor)
QUOTE: ### 3.10 Simplicity-prior safety training (pre-registered H11–H15; sealed zoo7)
PROBLEM: The section now also reports the pre-registered H16–H19 on sealed zoo8, but the heading still names only H11–H15 and zoo7. The Summary item 5 heading has the same problem: '(pre-registered; sealed zoo7, §3.10)'.
FINAL: ### 3.10 Simplicity-prior safety training (pre-registered H11–H15 on sealed zoo7 and H16–H19 on sealed zoo8)
WHY: Confirmed. REPORT.md lines 512-531 report the pre-registered H16-H19 on sealed zoo8 inside §3.10, but the heading names only H11-H15 and zoo7. The Summary item 5 heading at line 112-113, '(pre-registered; sealed zoo7, §3.10)', has the same problem and could be changed to '(pre-registered; sealed zoo7 and zoo8, §3.10)'. The scope label is wrong, so readers looking for H16-H19 will not find them by heading. Minor.

### [10] protocol-code (minor)
QUOTE: fine-tunes each effective backdoor twice from the same released weights, on the declared distribution (P) and on
PROBLEM: The Method bullet says each backdoor is fine-tuned twice (P and Q). On zoo8 each was fine-tuned three times (P, Q75, Q), and the section never gives Q75's composition (75% declared, 6.25% per program) or says that it uses the same optimiser and the same released weights.
FINAL: fine-tunes each effective backdoor from the same released weights on the declared distribution (P), on mixtures with a larger declared share (Q75: 75% declared, 6.25% per program, on zoo8 and the dev pilot; Q50 on the dev pilot only) and on
WHY: Confirmed. run_zoo8.sh and addendum b use --arms P,Q75,Q, so each zoo8 backdoor was fine-tuned three times, not twice. persistence_broad.make('Q75') gives weights (0.75, 0.0625 x4), and the 'uniform' program is the declared iid prompt distribution. Each arm deep-copies the same m0; asr0 is identical across arms, and the fine-tuning stream seed is cfg seed + 777 in every arm. The proposed '(Q75, zoo8 only)' is wrong, though: Q75, and also Q50, were run on the dev zoo1 pilot. Minor.

### [11] numbers-claims (minor)
QUOTE: Every other survivor's coinciding fraction is below half its initial ASR, so its survival
PROBLEM: §5 Limitations is now false. zoo8 has a second survivor, b0025 (descending + swap to sort), whose coinciding fraction is at least half its initial ASR. That is why H16–H19 exclude it. The sentence still says only zoo7 b0025 is affected.
FINAL: `analysis/asr_coincidence.py` finds one model (zoo7 b0025) whose measured ASR is entirely such coincidences. The same check, which `analysis/eval_broad.py --prereg-mix` applies to zoo8, flags zoo8 b0025 (the same descending + swap-to-sort design). Its coinciding fraction (0.47) exceeds half its initial ASR (0.84), so its ASR cannot fall to half under any clean fine-tuning, and the registered rule excludes it from H16–H19. Every other survivor's coinciding fraction is below half its initial ASR, so its survival requires genuine backdoor outputs.
WHY: Confirmed. coincide() for zoo8 b0025 (desc + taskswap->sort) is 0.473, its asr0 is 0.836, and 0.473 >= 0.418. It survived P, Q75 and Q, with ASR plateauing at 0.47-0.48 (end of training: P 0.477, Q75 0.473, Q 0.473). eval_broad.prereg_mix excludes it, and results/zoo8_H16_H19.txt lists it. The __main__ of analysis/asr_coincidence.py loops only over zoo1-zoo7 (lines 29-31), so it never scans zoo8. I re-ran it: across zoo1-zoo7, zoo7 b0025 is the only model at or above half. So the sentence 'Every other survivor's coinciding fraction is below half its initial ASR' is false once zoo8 is counted. Minor.

### [12] numbers-claims (minor)
QUOTE: checks every persisted model: b0025 is the only one whose coinciding fraction reaches half its initial ASR.
PROBLEM: This note in the zoo7 block is now stale and ambiguous. zoo8 also has a b0025 whose coinciding fraction reaches half its initial ASR. The committed script does not check the zoo8 persistence file.
FINAL: checks every model in the zoo1–zoo7 persistence files: b0025 is the only one there whose coinciding fraction reaches half its initial ASR (zoo8's b0025, of the same design, also does; see the zoo8 block).
WHY: Confirmed. The script's __main__ (asr_coincidence.py:29-41) scans only the zoo1-zoo7 persistence_lr1e-4/persistence_broad files, not runs/zoo8/persistence_mix.jsonl. So 'checks every persisted model' no longer holds now that §3.10 reports zoo8. I re-ran the script: within zoo1-zoo7, zoo7 b0025 is the only model whose coinciding fraction is at least half its asr0. zoo8 b0025 (0.473 against 0.836) also meets the threshold. Minor.

### [13] numbers-claims (minor)
QUOTE: (31 rule, 19 needle). One coincidence artefact was excluded by the registered rule
PROBLEM: The text reads as if zoo8 had 50 effective backdoors with one of them excluded. In fact 51 were effective and 50 remained after the exclusion. The zoo7 block uses the opposite convention: its 53 includes b0025.
FINAL: Replace '50 effective backdoors (31 rule, 19 needle). One coincidence artefact was excluded by the registered rule: zoo8's b0025, again descending + task swap to sort.' with: '51 effective backdoors. One coincidence artefact was excluded by the registered rule: zoo8's b0025, again descending + task swap to sort (coinciding fraction 0.47 against an initial ASR of 0.84; it survived all three arms at about that coinciding level). That leaves 50 analysed (31 rule, 19 needle).'
WHY: Confirmed. The registered definition is 'Effective = included and ASR >= 0.5', with exclusions applied afterwards. All 96 zoo8 models are included. 51 backdoors have release ASR >= 0.5, matching the 51 unique rows in persistence_mix.jsonl, and 50 remain after b0025 is excluded (the table sums to 50, with desc n = 6 of 7). The zoo7 block in the same section counts the artefact among its '53 effective'. Saying '50 effective' and then 'one was excluded' therefore misstates the count under the registered definition and uses the opposite convention to zoo7. Minor.

### [14] numbers-claims (minor)
QUOTE: So adding a quarter of program prompts to declared fine-tuning (Q75) kept the rule-trigger benefit without slowing
PROBLEM: Two problems. First, Q75 replaces a quarter of the declared prompts; it does not add program prompts. Batch size is fixed at 128, so declared prompts per step drop to 75%, and that dilution is the mechanism the report says slows needle erosion. Summary item 5 correctly says 'Replacing'. Second, 'without slowing needle erosion' rests on the paired median ratio alone. On dev, the other needle summaries moved in the slower direction (not significantly).
FINAL: So replacing a quarter of the declared prompts with program prompts (Q75; same batch size and number of steps) kept the rule-trigger benefit with no clear slowing of needle erosion.
WHY: The first half is confirmed. persistence_broad.make() builds Q75 as BroadSampler(weights=(0.75, 0.0625 x4)) with the same s.batch(128) and the same 300 steps. Q75 therefore replaces a quarter of the declared prompts; it adds nothing. Summary item 5 says 'Replacing', and the report's needle mechanism is this dilution. The second half is only partly supported. On dev the paired median ratio is 1.00 (22 slower, 19 faster, 4 ties; one-sided log-ratio Wilcoxon p = 0.41), so 'slowing' is not shown. But survivors rose 5 -> 6 (2 only under Q75, 1 only under P) and the median needle half-life went 37 -> 43, so 'no clear slowing' is more accurate than a flat 'without slowing'. The extra statistics in the proposed text are not needed. Minor.

### [15] numbers-claims (minor)
QUOTE: **Mixing weight (dev zoo1 pilot, exploratory; `runs/zoo1/persistence_mix.jsonl`).**
PROBLEM: The cited source is incomplete. persistence_mix.jsonl holds only the Q50 and Q75 arms. The 100% (P) and 25% (Q) numbers in this bullet come from runs/zoo1/persistence_broad.jsonl.
FINAL: **Mixing weight (dev zoo1 pilot, exploratory; 75% and 50% arms in `runs/zoo1/persistence_mix.jsonl`, P and Q in `runs/zoo1/persistence_broad.jsonl`).**
WHY: Confirmed. runs/zoo1/persistence_mix.jsonl contains only the Q50 and Q75 keys (asr0/half/censored/etc._Q50 and _Q75), and run_mix_pilot.sh:5 runs '--arms Q50,Q75'. The P (100%) and Q (25%) figures in the bullet (6/36, 34, 5/45, 8) come from runs/zoo1/persistence_broad.jsonl, which has the same 81 ids and identical asr0. The cited source is incomplete for reproducing the bullet. Minor.

### [16] numbers-claims (minor)
QUOTE: zoo8 was trained (commit `6bd39e3`).
PROBLEM: The provenance paragraph lists which audits checked which numbers. It leaves out the round-4 audit of the zoo7 part (docs/audit_round4.md, 20 corrections adopted in ab3f41f), and it does not say that the dev mixing pilot and zoo8 numbers had not been audited when they were added. §6 also omits round 4.
FINAL: zoo8 was trained (commit `6bd39e3`). A fourth round checked the zoo7 part of §3.10 and Summary item 5 (`docs/audit_round4.md`, 20 corrections, all adopted); the dev mixing pilot and the zoo8 results were added after that round.   [and add to §6: '- `docs/audit_round4.md` (20 corrections, all adopted): the pre-registered zoo7 simplicity-prior result (Summary item 5, §3.10) and its code.']
WHY: Confirmed. docs/audit_round4.md exists ('25 raised, 20 confirmed, all adopted'; committed in ab3f41f at 17:53), and FINDINGS.md:239 cites it. REPORT.md does not mention it anywhere: grep for round4 returns only round2 and round3 in the provenance paragraph and in §6. The audit record that the provenance paragraph and §6 present is therefore incomplete. The dev mix pilot (9077770, 18:27) and the zoo8 results (9fab79b/60ea733/3e911c5, 19:30-19:31) came after round 4. I do not use the proposed wording 'have not been independently audited', because it would become stale as soon as the present round is adopted. Minor.

### [17] numbers-claims (minor)
QUOTE: ### 3.10 Simplicity-prior safety training (pre-registered H11–H15; sealed zoo7)
PROBLEM: The section heading is stale. §3.10 now also reports pre-registered H16–H19 on sealed zoo8. Summary item 5's heading is stale in the same way ('(pre-registered; sealed zoo7, §3.10)', REPORT.md:112-113), although that item now carries the zoo8 results.
FINAL: ### 3.10 Simplicity-prior safety training (pre-registered H11–H19; sealed zoo7 and zoo8)   [and in Summary item 5: '(pre-registered; sealed zoo7 and zoo8, §3.10).**']
WHY: Confirmed. REPORT.md:438 gives the heading of §3.10 as '(pre-registered H11–H15; sealed zoo7)', but the section now also contains the 'Sealed zoo8' block with pre-registered H16–H19. The Summary item 5 heading at REPORT.md:112-113 still reads '(pre-registered; sealed zoo7, §3.10)', yet that item now includes the zoo8 bullet. Both headings misstate what their sections contain. Minor.
