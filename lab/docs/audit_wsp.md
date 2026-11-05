# Claims audit of the weight-space-programming write-up (2026-11-05)

Three auditors (numbers; claims and interpretation; the LinkedIn draft), each followed by a skeptic who re-checked
every finding against `results/` and the run files. 69 raised, 69 confirmed (many overlap across files), all adopted
in REPORT_WSP.md, README.md, FINDINGS.md and the LinkedIn draft. The registered analysis and its verdicts did not
change; the corrections concern descriptive counts, framing, the mechanism story and scope.

### [0] numbers (major) — REPORT_WSP.md
QUOTE: partial absorption (16/24)
PROBLEM: 16/24 is the count of level-1 live runs with B >= 0.5 (a descriptive threshold, not a registered statistic). It is not the number of partially absorbed models. Under the registered end-state taxonomy in addendum e (partial = 0.1 <= B < 0.9), only 10/24 level-1 live runs are partial. 7/24 are fully absorbed (B >= 0.9) and 7/24 are not absorbed (B < 0.1). The count also changes with level and data stream, and the sentence gives neither.
SKEPTIC: Confirmed from runs_wsp/sealed.jsonl and the summary. At live L1, 16/24 is the count with B >= 0.5 (wsp_posthoc.txt line 20). It breaks down as 7 with B >= 0.9, 9 with 0.5-0.9, 1 with 0.1-0.5 and 7 with B < 0.1. The registered fate taxonomy gives partial = 10 (prereg line 66). Calling 16/24 'partial absorption' reads as 16 partly absorbed models, but 7 of them are fully absorbed, and the level and threshold are not stated. B >= 0.5 by level is 12/16/15/18, and the second stream gives 14/24.
FINAL: variable absorption: at force level 1 the rule survived deletion (B >= 0.5) in 16/24 models, 7 of them fully (B >= 0.9), and not at all (B < 0.1) in 7/24; 14/24 on a second data stream

### [1] numbers (major) — README.md
QUOTE: partial absorption (16/24); the network keeps the
  patch firing instead.
PROBLEM: Same as in REPORT_WSP.md: 16/24 counts B >= 0.5 at level 1, not partial absorption. The registered fate taxonomy gives partial = 10/24, full = 7/24 and none = 7/24. The firing claim is exploratory or post hoc, but it sits unlabelled under the 'Result (pre-registered ...)' banner.
SKEPTIC: Same mislabel as in REPORT_WSP.md: 16/24 is B >= 0.5 at L1, made up of 7 full, 9 intermediate and 7 none, while registered partial = 10. Addendum e lists 'firing of the live trigger' as exploratory, yet the README puts the firing claim under the 'pre-registered' banner without a label (REPORT_WSP.md labels it post hoc).
FINAL: mixed absorption (at force level 1 the rule survived deletion in 16/24 models, 7 of them fully, and not at all in
  7/24); exploratory: the network keeps the patch firing instead.

### [2] numbers (major) — linkedin_draft.md
QUOTE: The network keeps the patch firing and absorbs the rule only partly: 16/24.
PROBLEM: 'absorbs the rule only partly: 16/24' reads as 16 of 24 models partly absorbing. In fact 16/24 is B >= 0.5 at level 1: 7 were fully absorbed, 9 were in between, 7 did not absorb at all. The number is also fragile. It is 14/24 on the replicate data stream, 7/24 at 1x lr and 6/12 under SGD.
SKEPTIC: Readers will take '16/24' to mean 16 models absorbed the rule partly. Recomputed at live L1: 7 fully (B >= 0.9), 9 in between and 7 not at all (B < 0.1). The second data stream gives 14/24. Recipe sensitivity is covered by the caveats finding.
FINAL: The network keeps the patch firing, and the outcome varies from model to model: the rule survived deletion in 16 of 24 (7 of them fully) and not at all in 7; rerun on different data, 16 became 14.

### [3] numbers (major) — REPORT_WSP.md
QUOTE: Visible behaviour hid all of this: with the patch in place, rule accuracy stayed near 1 in every arm.
PROBLEM: This holds for the median curve but not for individual models. In the primary condition, a third of detached runs and 11 live runs dropped below 0.9 full-model rule accuracy at the takeover. The lowest was 0.61. Addendum e registers 'minimum F during takeovers' as an exploratory item, but the report never gives it.
SKEPTIC: Recomputed per-run min F (q=0.3, Adam 3x, 96 runs per arm). Detached: 31/96 runs below 0.9, 8/96 below 0.8, lowest 0.607 (L0, step 150), each a single eval at steps 125-200. Live: 11/96 below 0.9, lowest 0.75. Frozen: lowest 0.994. The median F curves stay >= 0.962 (detached) and >= 0.991 (live), and every run ends >= 0.98. 'Near 1' holds for the median and the end state, but 'hid all of this' is false run by run. The registered exploratory item 'minimum F during takeovers' (prereg lines 65-72) is never reported.
FINAL: Visible behaviour hid most of this: with the patch in place, median rule accuracy never fell below 0.96 and ended at 1.00 in every arm, but individual models dipped briefly at the takeover (detached: below 0.9 at one evaluation in 31/96 runs, lowest 0.61; live: 11/96, lowest 0.75; frozen: never below 0.99).

### [4] numbers (major) — README.md
QUOTE: With the patch in place, rule accuracy stayed near 1 throughout, so behaviour gave no sign of any of this.
PROBLEM: 'Throughout' and 'no sign' are false at the level of individual runs. Detached-arm F dropped below 0.9 at the takeover in 31/96 runs, as low as 0.61. Only the median stays near 1.
SKEPTIC: Same recomputation. Only the median stays near 1 (>= 0.96). Individual detached runs fell below 0.9 at the takeover in 31/96 cases, as low as 0.61, so 'no sign' and 'throughout' overstate.
FINAL: With the patch in place, median rule accuracy stayed at or above 0.96 throughout, so behaviour gave little sign of any of this (individual models dipped briefly at the takeover, as low as 0.61).

### [5] numbers (major) — linkedin_draft.md
QUOTE: All of this happened while the model's visible behaviour did not change: with the patch in place, rule accuracy stayed near 100% throughout.
PROBLEM: Visible behaviour did change briefly in many models: transient F dips below 90% in 31/96 'no gradient' runs, as low as 61%. Only the median stayed near 100%.
SKEPTIC: Visible behaviour did change briefly in many models: 31/96 detached runs dipped below 90%, lowest 61%. Only the median curve stayed >= 96%. End F >= 99.6% in every detached run.
FINAL: Visible behaviour gave almost no sign of this: with the patch in place, median rule accuracy never fell below 96% and ended at 100%, although some individual models dipped briefly (as low as 61%) while the network took the rule over.

### [6] numbers (major) — REPORT_WSP.md
QUOTE: the network never learned the rule, so
  deleting the patch still removed the behaviour. The rule survived deletion in 1/24 models.
PROBLEM: 'Never learned the rule' contradicts the registered numbers. Median B_frozen is 0.21 at level 0. The frozen body's z-margin gain is positive at every level. At level 0, 14/24 frozen runs are 'partial'. '1/24' holds only at force level 1 with threshold B >= 0.5, and neither is stated. Level 0 gives 3/24.
SKEPTIC: The registered output contradicts 'never learned'. Median B_frozen is 0.21 at L0 (prereg line 34). Frozen z-margin gain is positive at every level (+6.26/+4.67/+2.86/+1.03), and 14/24 L0 frozen runs are 'partial'. Recomputed frozen B >= 0.5 by level: 3/1/0/0; B >= 0.1: 15/10/3/1. The 1/24 holds only at level 1 with B >= 0.5, and neither is stated.
FINAL: the network learned the rule only weakly (median B 0.21 at the weakest force, at most 0.03 at the others), so
  deleting the patch still removed most of the behaviour. The rule survived deletion (B >= 0.5) in 1/24 models at force level 1 (3/24 at level 0, 0/24 at levels 2-3).

### [7] numbers (major) — README.md
QUOTE: the network never learns the rule; deleting the patch
  removes the behaviour (rule survived deletion in 1/24 models);
PROBLEM: Same as in REPORT_WSP.md. Frozen median B is 0.21 at level 0, the z-margin gain is positive at all levels, and '1/24' is a level-1, B >= 0.5 count with the level and threshold unstated.
SKEPTIC: Same as in REPORT_WSP.md: frozen median B 0.21 at L0, positive z-margin gain at every level, B >= 0.5 of 3/1/0/0 by level. 1/24 is an unlabelled level-1 count.
FINAL: the network learns the rule only weakly; deleting the patch removes most of
  the behaviour (rule survived deletion in 1/24 models at force level 1; 3/24 at the weakest force);

### [8] numbers (major) — linkedin_draft.md
QUOTE: The network never learns the rule, and deleting the patch removes the behaviour. The rule survived deletion in 1/24 models.
PROBLEM: Overstated, and the level is unlabelled. Frozen bodies did learn some of the rule: median B 0.21 at the weakest force, and 10/24 models with B >= 0.1 at level 1. 1/24 is the level-1 count only.
SKEPTIC: Overstated, and the level is unlabelled. Frozen bodies learned some of the rule: median B 0.21 at L0, and 10/24 with B >= 0.1 at L1. 1/24 is the level-1 count; level 0 gives 3/24.
FINAL: The network learns the rule only weakly, and deleting the patch removes most of the behaviour: the rule survived deletion in 1 of 24 models (3 of 24 with the weakest patch).

### [9] numbers (minor) — REPORT_WSP.md
QUOTE: Without a patch, the networks learn the rule in 120-130 steps. Detached patches were absorbed by step
  162-175, i.e. about 75 steps after the learning rate became non-zero (it is 0 for 100 steps, then warms up).
PROBLEM: The two sentences count steps differently. The twin figure (120-130) includes the 100 lr-0 steps; the detached figure ('about 75') excludes them. Side by side, detached absorption looks faster than learning without a patch. On the same clock it is 2-4x slower: 50-100 vs 20-30 steps after the lr leaves 0. Also, 162-175 are per-level medians; the per-base range is 150-200.
SKEPTIC: Verified. Twin T_take is 120-130 (eval every 5) counted from step 0, which is 20-30 steps after lr > 0 (patch.py: lr 0 for st < 100). Detached per-base T_take is 150-200 at every level, with medians 162/175/175/175, i.e. 50-100 steps after lr > 0. Every detached base is slower than every twin. Mixing two clocks invites comparing '120-130' with 'about 75', which reverses the direction, and '162-175' is a range of medians, not of models. Minor, because the sentence does give 'by step 162-175' on the twin's clock.
FINAL: The learning rate is 0 for the first 100 steps, then warms up. Without a patch, the networks learn the rule by step 120-130 (20-30 steps after the learning rate leaves 0). Detached patches were absorbed by step
  150-200 (median 162-175 across levels), i.e. 50-100 steps after it leaves 0: slower than learning the rule with no patch (median 75 vs 30 steps).

### [10] numbers (minor) — REPORT_WSP.md
QUOTE: within about 75 training steps
PROBLEM: The headline's own figure, directly below it, shows detached absorption at about step 175 on its 'training steps' axis. '75' silently excludes the 100 lr-0 steps and is a median; the per-base range is 50-100 steps after the lr leaves 0. Section 5 repeats it ('it completes in about 75\n  steps.').
SKEPTIC: The headline figure directly below plots detached absorption at about step 150-200 on its 'training steps' axis (checked in the image). '75' is a median that silently drops the 100 lr-0 steps; per base it is 50-100 steps after the lr leaves 0.
FINAL: by training step 150-200 (50-100 steps after the learning rate left 0)

### [11] numbers (minor) — README.md
QUOTE: the network absorbs the rule within about 75
  steps
PROBLEM: The step count excludes the 100 lr-0 steps without saying so, and it is a median. The figure directly below shows the crossing at step about 175.
SKEPTIC: Same as in REPORT_WSP.md: the count excludes the 100 lr-0 steps and is a median, and the figure below it shows the crossing at about step 150-200.
FINAL: the network absorbs the rule by training step 150-200
  (50-100 steps after the learning rate leaves 0)

### [12] numbers (minor) — linkedin_draft.md
QUOTE: The network absorbs the rule within about 75 steps, and deleting the patch changes nothing: 24/24.
PROBLEM: 'About 75 steps' is a median that excludes the 100 lr-0 steps. Per model it was 50-100 steps after the learning rate left 0, or step 150-200 on the plotted axis. Without the qualifier it also reads as faster than learning without a patch (20-30 steps).
SKEPTIC: A median that excludes the lr-0 steps. Per model it is 50-100 steps after the learning rate is switched on (step 150-200 on the plotted axis).
FINAL: The network absorbs the rule within 50-100 training steps of the learning rate being switched on, and deleting the patch changes nothing: 24/24.

### [13] numbers (minor) — REPORT_WSP.md
QUOTE: Post hoc, the in-batch losses stay bitwise identical through all 100 warm-up steps in 96/96 cells.
PROBLEM: The '100 warm-up steps' are the lr-0 moment warm-start steps. No parameter moves during them, so identical losses follow from W0 by construction and add no evidence. The report and addendum e use 'warm-up' for the 100-step linear ramp after these steps, so readers will take this to mean identity held through the ramp. It did not: the arms diverge 1-56 steps into it.
SKEPTIC: Verified. In patch.py, cur = 0 for st < 100 and loss[i] is computed before update i, so loss[0..100] all use install-time weights and identity follows from W0. Addendum e calls the later linear ramp the 'warmup', so 'warm-up steps' suggests identity held during actual parameter movement. Recomputed first differing step: frozen vs others at 101 in 95/96 cells and 102 in 1; live vs detached at 108-156. I rate it minor rather than major: W0 itself is correct, and the post hoc addition adds nothing, but it changes no conclusion.
FINAL: Post hoc, in-batch losses are bitwise identical in 96/96 cells through step 100, as implied by W0 while the learning rate is 0; the frozen arm departs right after the first non-zero update (step 101-102), and live departs from detached at steps 108-156.

### [14] numbers (minor) — FINDINGS.md
QUOTE: in-batch losses identical through the 100 warm-up steps in 96/96 cells
PROBLEM: Same mislabel as in REPORT_WSP.md. These are lr-0 steps, during which identity is implied by W0. The arms diverge within 1-56 steps of the first non-zero update, early in the linear warm-up.
SKEPTIC: Same mislabel: these are lr-0 steps, where identity is implied by W0. The arms diverge 1-56 non-zero updates later (first difference at 101-102 for frozen, 108-156 for live vs detached).
FINAL: in-batch losses identical through the 100 lr-0 steps in 96/96 cells (implied by W0); the arms diverge 1-56 steps after the first non-zero update

### [15] numbers (minor) — REPORT_WSP.md
QUOTE: median of 21-48 misfires in the first 400 steps; 0 in the other arms)
PROBLEM: Three problems. (1) There is no denominator: 21-48 misfires out of about 283,000 clean positions is a rate of about 1e-4. (2) '0' is a median: detached max is 2, and the frozen arm's live-network reading reaches max 6. (3) The post hoc 0-399 window hides a contrary pattern. In steps 400-1999 the frozen arm's live-network trigger reading, which is never applied and receives no gradient, misfires a median 62-100 times, more than the live arm at levels 1-3. Over the run, misfiring is therefore not specific to gradient access. That undercuts the headline 'even made it fire on clean inputs' and the next sentence's mechanism.
SKEPTIC: Partly confirmed. Live misfire_live[:400] medians are 48/27/21/24 against a median of 283,286 clean positions per run, a rate of about 1e-4, and no denominator is given. '0' is a median (detached max 2, frozen live-reading max 6). The auditor's further claim that the frozen arm's later misfires (median 62-100 in steps 400-1999) undercut the mechanism does not hold: that reading is never applied and has no cost. The like-for-like control is detached (same applied live reading, no gradient), which stays at median 0 (max 8) over the whole run. So the issue is the missing rate and the median label, and I downgrade it to minor.
FINAL: median of 21-48 misfires in steps 0-399, out of about 283,000 clean positions, a rate of about 1e-4; median 0, at most 6, in the other arms)

### [16] numbers (minor) — FINDINGS.md
QUOTE: misfires on
  clean inputs (median 21-48 in 400 steps)
PROBLEM: No denominator (about 283k clean positions, a rate of about 1e-4), and no note that later in training the frozen arm's unapplied live-network reading misfires as often.
SKEPTIC: No denominator: about 283k clean positions per run, a rate of about 1e-4. The frozen-arm comparison after step 400 is not like-for-like (unapplied reading), so I leave it out.
FINAL: misfires on
  clean inputs (median 21-48 of ~283k clean positions in steps 0-399, a rate of ~1e-4; median 0 in the other arms)

### [17] numbers (minor) — REPORT_WSP.md
QUOTE: In the first 400 steps, the live trigger kept firing at 0.89-0.99 of
PROBLEM: The firing fractions are computed over steps 100-399, not the first 400 steps. Over steps 0-399 the detached figure would be 0.37-0.41, not 0.16-0.21.
SKEPTIC: wsp_posthoc.py line 30 uses fire_live[100:400], and wsp_posthoc.txt says 'steps 100-399'. Recomputed over [0:400]: detached 0.37-0.41 (vs the reported 0.16-0.21) and live 0.92-0.99. The stated window is wrong.
FINAL: In steps 100-399 (after the learning rate left 0), the live trigger kept firing at 0.89-0.99 of

### [18] numbers (minor) — REPORT_WSP.md
QUOTE: In partly absorbed live runs, the shift task (which needs the conjunction "task = shift and the symbol
  before a in the prompt") lagged the other three.
PROBLEM: This is true only at levels 1 and 3. At level 0, sort lagged most. At level 2, sort lagged at least as much as shift. The registered per-task output over all bases also shows sort lowest at level 0.
SKEPTIC: Recomputed over live runs with 0.1 <= B < 0.9. L0 (n=12): copy 0.93, rev 0.94, sort 0.40, shift 0.84, so sort lags. L1 (n=10): shift 0.10 vs >= 0.98. L2 (n=9): sort 0.00, shift 0.11. L3 (n=6): shift 0.22 vs >= 0.99. The claim holds only at levels 1 and 3.
FINAL: In partly absorbed live runs (0.1 <= B < 0.9), which task lagged depended on force: shift (which needs the conjunction "task = shift and the symbol
  before a in the prompt") lagged at levels 1 and 3 (median B 0.10 and 0.22 vs >= 0.98 for the other tasks), sort at level 0 (0.40 vs shift 0.84), and both at level 2 (sort 0.00, shift 0.11).

### [19] numbers (minor) — REPORT_WSP.md
QUOTE: levels 1-3: 19-21/24, p ≤ 1e-4
PROBLEM: The same cell cites a Holm p for level 0 (0.011) but an unlabelled p for levels 1-3. The Holm p at level 2 is 1.3e-4 > 1e-4; '≤ 1e-4' holds only for the uncorrected p. The W1 and W2 rows likewise cite uncorrected p (9e-6, ≤ 1.4e-4) without saying so, while the decision rule uses Holm p (W1 3.3e-5; W2 ≤ 2e-4).
SKEPTIC: The same cell gives a Holm p for level 0 but an unlabelled p for levels 1-3. Holm p at L1-3 is 9.6e-5, 1.3e-4 and 9.1e-5 (prereg line 43), so 'p ≤ 1e-4' is true only for the raw p (max 6.3e-5). The W1/W2 rows consistently use raw p, so only this cell mixes the two.
FINAL: levels 1-3: 19-21/24, Holm p ≤ 1.3e-4

### [20] numbers (minor) — REPORT_WSP.md
QUOTE: Whether a given live patch is absorbed is close to a coin flip
PROBLEM: Unquantified, and stronger than the data. The two data streams agree on absorbed vs not for 16/24 bases, above the 12.7 expected by chance, and B is positively correlated across streams.
SKEPTIC: Recomputed live L1 on ds7 vs ds8: B >= 0.5 in 16 and 14, the same classification for 16/24 bases against 12.7 expected by chance (kappa about 0.29), Spearman 0.38, |dB| > 0.5 in 6/24. Outcomes are hard to predict but weakly base-dependent, and absorption is about 2:1, so 'coin flip' overstates.
FINAL: Whether a given live patch is absorbed is hard to predict (the two data streams agreed on absorbed vs not, B >= 0.5, for 16/24 models, against 12.7 expected by chance)

### [21] numbers (minor) — FINDINGS.md
QUOTE: live outcomes vary strongly between data streams (median |dB| 0.24)
PROBLEM: This sits in the 'Post hoc:' bullet, but the second data stream (noise floor of B) is a registered exploratory item in addendum e. It should be labelled exploratory (registered), not post hoc.
SKEPTIC: Addendum e registers 'a second data stream at level 1 for every arm (noise floor of B)' as exploratory, and the value comes from the registered output (prereg lines 60-62). Filing it under 'Post hoc:' mislabels it, in the conservative direction.
FINAL: live outcomes vary strongly between data streams (registered exploratory noise floor: median |dB| 0.24 live vs 0.00 detached and 0.03 frozen)

### [22] numbers (minor) — README.md
QUOTE: silently broke in-network patches but not frozen-reference ones
PROBLEM: W5 is a relative test (frozen F > live/detached F). Frozen-reference patches also lost accuracy at level 1: median 0.97, 7/24 below 0.95, minimum 0.83. 'Broke' overstates the median losses of 0.85-0.94, though single models fell to 0.11.
SKEPTIC: Recomputed q=0 F at 2000 steps. L1: live median 0.90 (min 0.48), detached 0.85 (min 0.22), frozen 0.97 (min 0.83, 7/24 < 0.95). L3: live 0.93 (0.65), detached 0.94 (0.11), frozen 1.00 (0.99). W5 is relative. Frozen patches also lost some accuracy at L1, and 'broke' overstates median losses of 6-15 points.
FINAL: silently lowered the rule accuracy of in-network patches (median 0.85-0.94 at 2000 steps, as low as 0.11 in one model) more than that of frozen-reference ones (median 0.97-1.00, lowest 0.83)

### [23] numbers (major) — linkedin_draft.md
QUOTE: Caveats: toy transformers (~100k parameters), one kind of rule (a logit remap), one training recipe. [secondary results: SGD / lower learning rate / training that never shows the rule]
PROBLEM: Three recipes were run, not one. The headline live count (16/24) is the recipe-sensitive one: 7/24 at 1x lr, 6/12 under SGD, 14/24 on a second data stream. The post omits this and leaves the placeholder unfilled.
SKEPTIC: Three recipes were run (Adam 3x, Adam 1x, SGD; wsp_posthoc.txt lines 20-22), so 'one training recipe' is wrong. The live count the post headlines is the recipe-sensitive one: 16/24, 7/24, 6/12, and 14/24 on the second stream. The placeholder is unfilled. Detached vs frozen held throughout (24/24 vs 0/24; 11/12 vs 0/12).
FINAL: Caveats: toy transformers (~100k parameters), one kind of rule (a logit remap), three closely related training recipes. The frozen-snapshot vs no-gradient contrast held in all three; the live-with-gradient count did not (16/24 in the main recipe, 7/24 at a 3x lower learning rate, 6/12 under SGD, 14/24 rerun on different data). Training that never shows the rule lowered rule accuracy more for live-network patches (median 85-94%) than for frozen-snapshot ones (97-100%).

### [24] numbers (minor) — README.md
QUOTE: **Result (pre-registered, 24 sealed toy transformers; `REPORT_WSP.md`).**
PROBLEM: The 'pre-registered' banner covers the 1/24, 24/24 and 16/24 counts and 'keeps the patch firing'. None of these is a registered test. The counts are B >= 0.5 at force level 1 (descriptive); firing is exploratory. The registered tests W1-W4 are paired Wilcoxon tests on B over all four levels.
SKEPTIC: The banner covers the 1/24, 24/24 and 16/24 counts, which come from the post hoc script (B >= 0.5 at L1; wsp_posthoc.txt line 20), and the exploratory firing claim. The registered tests are the paired Wilcoxon/Spearman tests W1-W6 over levels.
FINAL: **Result (24 sealed toy transformers; pre-registered tests W0-W6 all supported; the counts below are descriptive: force level 1, rule kept with the patch deleted at B >= 0.5; `REPORT_WSP.md`).**

### [25] numbers (minor) — README.md
QUOTE: # rerun (~20 CPU-hours)
PROBLEM: The per-run times recorded in the run file sum to 23.4 hours over the 636 single-threaded runs.
SKEPTIC: The runner sets torch threads to 1. The sum of d['secs'] over the 636 rows of runs_wsp/sealed.jsonl is 23.4 h, so '~20' understates by about 17%. Trivial.
FINAL: # rerun (~23 CPU-hours)

### [26] numbers (minor) — REPORT_WSP.md
QUOTE: (6 scouts, 12 proposals, 3 judges, novelty and failure-mode skeptics)
PROBLEM: Nothing in the repo records these counts, so they cannot be checked. Every other provenance item in the section points to a committed file.
SKEPTIC: A grep over /home/user/aegis (excluding .git and run dirs) finds these counts only in REPORT_WSP.md, and docs/ has no WSP design record. The claim is not shown to be false, but unlike every other provenance item it cannot be checked. Keep the counts and flag that they are not recorded.
FINAL: (6 scouts, 12 proposals, 3 judges, novelty and failure-mode skeptics; its transcripts are not in the repo)

### [27] claims (major) — REPORT_WSP.md
QUOTE: Post hoc, the in-batch losses stay bitwise identical through all 100 warm-up steps in 96/96 cells.
PROBLEM: The extra identity is guaranteed, not found. The learning rate is 0 for steps 0-99, so no weight moves. 'Identical through 100 warm-up steps' only repeats W0 on 100 more batches from the same untrained network. §4 also uses 'warms up' for the lr ramp (steps 100-199), so readers will take this to mean the arms stay identical while training. In fact the arms split as soon as the weights move.
SKEPTIC: Confirmed. disp is exactly 0.0 at step 100 in all 288 primary runs (warm = 100 lr-0 steps), so the identity is guaranteed. §4 uses 'warms up' for the lr ramp, so readers will take this to mean the arms stay identical during training. Bitwise identity ends at step 101-102 for frozen vs the others and at 108-156 (median 124) for live vs detached. The auditor's wording 'split as soon as the weights move' goes too far: frozen-detached loss differences are about 1e-10 at step 101 and reach about 1e-3 relative only by step 130.
FINAL: Post hoc: no weight moves during the 100 lr-0 steps that warm Adam's moments, so the in-batch losses are trivially identical there (96/96 cells). Bitwise identity ends once training starts: after 1-2 updates for frozen vs the other two (differences of about 1e-10 at first), and after 8-56 updates (median 24) for live vs detached.

### [28] claims (major) — REPORT_WSP.md
QUOTE: (3) analysis at the time of the edit cannot tell these
  cases apart: outputs and gradients are identical;
PROBLEM: Overclaim. Two things are identical: the forward function, and the first-order parameter gradients on training pools filtered so that no trained position has a trigger pre-activation in [0,1]. The three cases are not hidden from analysis at edit time, because the wiring differs and anyone can inspect it. The frozen arm carries a whole second, frozen copy of the network. The detached arm needs a stop-gradient in the training code. The gradient identity is also engineered by the data filter, and it ends after 1-2 updates (frozen) or 8-56 updates (live vs detached).
SKEPTIC: Confirmed in wsp/patch.py. The arms differ in wiring that anyone can inspect: the frozen arm runs body_forward on a deepcopy with 109,888 parameters, and the detached arm uses hn.detach(). The gradient identity holds only on pools filtered by filter_pool (trigger saturated or off; 454-1,901 of 40,000 rule sequences dropped per base) and only until the first updates.
FINAL: (3) the patched model's outputs and parameter gradients at the time of the edit do not distinguish these
  cases (on our filtered training data they are bitwise identical until the first few updates); only the wiring does, so inspect it;

### [29] claims (major) — REPORT_WSP.md
QUOTE: We wrote the same 134-number rule into each network's weights in three ways, which give bit-for-bit identical
outputs and gradients at the moment of the edit.
PROBLEM: Only the live arm is a weight edit. The frozen arm adds a full frozen copy of the install-time network that the patch reads, and the detached arm is a stop-gradient in the training code. Neither of these exists in weights. 'Identical gradients' holds only on the filtered training pools and only until training moves the weights.
SKEPTIC: Confirmed. The frozen arm needs a second, frozen 109,888-parameter copy of the network, and the detached arm is a stop-gradient in the training code, so neither is just 'weights'. The gradient identity holds only on the filtered pools and only until training moves the weights.
FINAL: We added the same 134-number module (two ReLU units) to each network and wired it in three ways: reading the live network, reading it behind a stop-gradient, or reading a frozen copy of the install-time network (a second, 109,888-parameter network). On training data filtered so that every trigger is fully on or fully off, the three give bit-for-bit identical
outputs and gradients until training moves the weights.

### [30] claims (major) — REPORT_WSP.md
QUOTE: Visible behaviour hid all of this: with the patch in place, rule accuracy stayed near 1 in every arm.
PROBLEM: The data contradict this. Detached absorption came with a visible dip in rule accuracy with the patch in place, and under SGD the live arm dipped a lot. The report's own mechanism in §5 ('Once a few rule cases are missed, the errors teach the network the rule') needs visible errors. The claim rests on medians and on evals every 25 steps, which miss the trough. The headline figure's right panel shows only the median.
SKEPTIC: Recomputed from runs_wsp/sealed.jsonl (minimum eval F per run, Adam 3x). Detached F fell below 0.9 in 6/7/9/9 of 24 models at levels 0-3, lowest 0.61, with the trough at a median of step 150. Under SGD the live arm's median minimum was 0.75 (lowest 0.27). Adam 1x detached reached 0.57. The frozen arm's minimum was 0.99. Final F has a median of 1.00 everywhere, so only the end state hid it.
FINAL: With the patch in place, median rule accuracy stayed near 1 in every arm and ended at 1.00, so the end state hid all of this. The takeover itself was briefly visible in some models: in the detached arm at level 1, eval rule accuracy dipped below 0.9 in 7/24 (lowest 0.61) around step 150, and under SGD the live arm's median minimum was 0.75. Only the frozen arm never dipped (minimum 0.99).

### [31] claims (major) — README.md
QUOTE: With the patch in place, rule accuracy stayed near 1 throughout, so behaviour gave no sign of any of this.
PROBLEM: Same overclaim as REPORT §header, made stronger by 'throughout' and 'no sign'. In many models the detached takeover was visible as a transient drop in rule accuracy with the patch in place.
SKEPTIC: Same data as the REPORT header finding. Detached L1 minimum F was below 0.9 in 7/24 (lowest 0.61) and SGD live had a median minimum of 0.75. 'Throughout' and 'no sign' are contradicted.
FINAL: With the patch in place, median rule accuracy stayed near 1 and ended at 1.00, so the end state gave no sign of any of this. The takeover itself showed as a brief dip in some models (below 0.9 in 7/24 detached models at level 1, lowest 0.61).

### [32] claims (major) — linkedin_draft.md
QUOTE: All of this happened while the model's visible behaviour did not change: with the patch in place, rule accuracy stayed near 100% throughout.
PROBLEM: 'Visible behaviour did not change' and 'throughout' are contradicted by the per-run data. The takeover in the no-gradient version dropped rule accuracy below 90% in about a third of models, and under SGD the live version fell much further.
SKEPTIC: Contradicted by the per-run eval F. Detached L1 fell below 0.9 in 7/24 models (lowest 0.61) and SGD live had a median minimum of 0.75. Only the frozen arm stayed at or above 0.99.
FINAL: Almost none of this showed in behaviour: with the patch in place, median rule accuracy stayed near 100% and ended at 100%. The only visible sign was a brief dip while the network took the rule over (below 90% in 7 of 24 models in the no-gradient version).

### [33] claims (minor) — REPORT_WSP.md
QUOTE: (1) after further training, reverting an edit may not revert the
  behaviour;
PROBLEM: Absorption needs the further-training data to demonstrate the edited behaviour, and the implication leaves that condition out. 30% of training sequences carry the rule as a target (a -> z). The unpatched twin learns the rule from those same data within 20-30 updates. When the data never showed the rule (q = 0), deleting the patch reverted the behaviour in every model and every arm. So 'absorption' means learning the rule from rule-labelled data once the patch stops covering it, not something that further training in general does to an edit.
SKEPTIC: Confirmed. All q = 0 rows in wsp_sealed_prereg.txt have B = 0.00 and fate shielded 24/24 for every arm. The twin learns the rule from the 30% rule-labelled data in 120-130 steps, i.e. 20-30 updates. So absorption requires training data that shows the rule. §1-2 of the REPORT state this setup, which limits the harm here, but the implication is phrased generally.
FINAL: (1) after further training on data that itself demonstrates the edited behaviour, reverting an edit may not revert the
  behaviour (when the training data never showed the rule, deleting the patch reverted it in every model);

### [34] claims (major) — README.md
QUOTE: Take-aways for anyone editing weights: after further training, reverting an edit may not revert the behaviour;
PROBLEM: This generalises beyond what was tested in two ways. First, the edit is two added ReLU units after the final LayerNorm, held frozen during further training. Of the three wirings, only the live one is a plain weight edit, and edits to existing parameters (the kind in the cited arXiv 2611.05341) were never tested. Second, the 'may not revert' result needs training data that demonstrates the rule (30% of sequences). With q = 0, deletion reverted the behaviour in 24/24 models in every arm. The README never says that 30% of the training data showed the rule ('data the patched network already gets right').
SKEPTIC: The README never says that 30% of training sequences showed the rule, and with q = 0 deletion reverted the behaviour in 24/24 models in every arm. The patch is two added units, frozen in every config: no config in stage1, stage1b, pilot2 or sealed sets trainable_patch. That is not the edit of existing parameters the term usually refers to.
FINAL: Tentative take-aways, for an add-on edit like this one that stays frozen while the rest of the network trains on data that demonstrates the edited behaviour (30% of our training sequences did): after further training, reverting the edit may not revert the behaviour (with training data that never showed the rule, it always did);

### [35] claims (major) — linkedin_draft.md
QUOTE: 1. "Undo" is not guaranteed. After further training, reverting an edit may no longer revert the behaviour.
PROBLEM: This leaves out the necessary condition that the further-training data demonstrated the rule (30% of examples). Without that, undo worked in every model. The post also says 'data the patched model already got right' without mentioning that this data shows the rule.
SKEPTIC: The necessary condition is left out. At q = 0, B = 0.00 and all 24/24 models are shielded in every arm. The post never says that its 'data the patched model already got right' includes 30% rule examples.
FINAL: 1. "Undo" is not guaranteed. If further training uses data that shows the edited behaviour (30% of our examples did), reverting the edit may no longer revert it. When the training data never showed the rule, deleting the patch undid it in every model.

### [36] claims (major) — REPORT_WSP.md
QUOTE: - One kind of edit: a logit-level remap whose trigger reads the very logit it acts on. Patches that read earlier
  features, or that are themselves trainable, were not tested in the sealed run.
PROBLEM: The limitations miss the scope facts that matter most for the 'Implications for weight-space programming'. (a) The patch is an architectural add-on (two new units), not an edit of existing weights. (b) It is frozen while the body trains. (c) Two of the three arms are not weight edits at all (frozen copy; stop-gradient). (d) Training pools were filtered to make the step-0 identity hold. Also, 'not tested in the sealed run' suggests trainable patches were tested elsewhere. No run anywhere used trainable_patch.
SKEPTIC: No config anywhere (stage1, stage1b, pilot2, sealed) uses trainable_patch, so 'not tested in the sealed run' wrongly implies it was tested elsewhere. The limitation also omits scope facts that matter for the weight-space programming implications. The patch is an add-on and is frozen. Two arms need extra machinery: a frozen network copy, or a stop-gradient. The step-0 identity depends on filtered pools. The cited 2611.05341 edits about 3 existing parameters.
FINAL: - One kind of edit: two new ReLU units added after the final LayerNorm (an add-on, not a change to existing weights), held frozen while the rest of the network trains, implementing a logit-level remap whose trigger reads the very logit it acts on. The frozen arm also keeps a frozen copy of the whole network, and the detached arm needs a stop-gradient in the training code. No run tested edits to existing parameters, patches that read earlier
  features, or patches that are themselves trained. The step-0 identity holds only because the training pools were filtered so that every trigger is fully on or fully off.

### [37] claims (major) — README.md
QUOTE: - the patch reads the **live network, inside the gradient path**: partial absorption (16/24); the network keeps the
  patch firing instead.
PROBLEM: 16/24 is a single cell: force level 1, Adam at 3x the base's final lr, data stream 7. The live outcome changes a lot with recipe and data stream. At the base's own final lr the live arm looks almost like frozen (7/24). 'Partial absorption (16/24)' also misreads the bimodal distribution. 'Keeps the patch firing' is a post hoc mechanism claim presented inside a pre-registered result.
SKEPTIC: Recomputed. 16/24 is level 1 at Adam 3x only. At level 1 the distribution is bimodal: B >= 0.9 in 7, B < 0.1 in 7. Across levels B >= 0.5 in 12/16/15/18. Adam 1x gives 7/24 and SGD 6/12. The second data stream flips absorbed status in 8/24 (Spearman 0.38). 'Keeps the patch firing' is post hoc, and the frozen arm's body keeps the trigger firing too.
FINAL: - the patch reads the **live network, inside the gradient path**: the outcome depends on the recipe and the data stream. At 3x the base's final learning rate the network took the rule over (B >= 0.5) in 16/24 models (fully in 7, not at all in 7). At 1x it did so in 7/24 and under SGD in 6/12, and a second data stream changed the outcome in 8/24. Where it was not absorbed, the live trigger kept
  firing (post hoc).

### [38] claims (major) — linkedin_draft.md
QUOTE: Caveats: toy transformers (~100k parameters), one kind of rule (a logit remap), one training recipe. [secondary results: SGD / lower learning rate / training that never shows the rule]
PROBLEM: 'One training recipe' is inaccurate (three were run), and the placeholder hides results that change the headline. The live arm's 16/24 drops to 7/24 at the model's own final lr. With no rule in the training data, undo always worked. The caveats also omit that the edit is an add-on held frozen during training and that 30% of training examples showed the rule.
SKEPTIC: Three recipes were run (Adam 3x, Adam 1x, SGD), so 'one training recipe' is wrong. The placeholder hides results that weaken the headline: live 16/24 becomes 7/24 at 1x and 6/12 under SGD, and at q = 0 undo always worked. The caveats also omit that the edit is a frozen add-on and that 30% of examples showed the rule. The q = 0 F medians check out: live 0.90/0.93, detached 0.85/0.94, frozen 0.97/1.00.
FINAL: Caveats: toy transformers (~100k parameters); one kind of edit (two added units implementing a logit remap, held frozen while the rest of the network trained); 30% of the further-training examples showed the rule. The live-wiring result depends on the recipe: 16/24 absorbed at a high learning rate, 7/24 at the model's own final learning rate, 6/12 under SGD. When training never showed the rule, deleting the patch undid it in every model; in-network patches lost some rule accuracy (median 85-94%) while frozen-snapshot ones held (97-100%).

### [39] claims (major) — REPORT_WSP.md
QUOTE: In the first 400 steps, the live trigger kept firing at 0.89-0.99 of
  rule positions, against 0.16-0.21 for the detached trigger. The live trigger also began firing on clean inputs (a
  median of 21-48 misfires in the first 400 steps; 0 in the other arms). With gradient access to the trigger, the
  network pushes the trigger's input up rather than learning the rule itself.
PROBLEM: The cited evidence does not separate the proposed mechanism from the null. The same trigger read off the frozen arm's body, which gets no gradient through the trigger, also kept firing (0.97). So continued firing is the default, and the informative event is the detached trigger switching off. The '0 in the other arms' depends on the 400-step window, which was chosen post hoc: over the full run, the frozen arm's body misfires as often as the live arm's. The final sentence states the mechanism as fact. Some better evidence exists but is not cited. Example: the body's a-logit early on is higher in live than in frozen.
SKEPTIC: Confirmed. The same trigger read off the frozen arm's body fired at 0.966-0.973, and at levels 2-3 it fired more than the live arm's (0.91/0.89). So continued firing is the default, and the informative event is the detached trigger switching off. Over the full run the frozen body misfired about as often as the live one at level 1 (93 vs 92, median; frozen 0 in steps 0-399, 80 in 1000-1999). The early live-only misfires (about 1e-4 of roughly 283k clean positions) remain a genuine live-specific signal. The final sentence states a mechanism that was never tested.
FINAL: In the first 400 steps, the live trigger kept firing at 0.89-0.99 of
  rule positions, against 0.16-0.21 for the detached trigger. But the same trigger read off the frozen arm's body, which gets no gradient through it, also kept firing (0.97), so continued firing is the default and the informative event is the detached trigger switching off. The live trigger misfired on clean inputs early (a
  median of 21-48 misfires in the first 400 steps, about 1e-4 of clean positions; 0 in the other arms), although over the full run the frozen arm's body misfired about as often (level 1: median 93 vs 92). Our hypothesis is that, with gradient access to the trigger, the
  network holds the trigger's input up rather than learning the rule itself; we did not test this directly.

### [40] claims (minor) — REPORT_WSP.md
QUOTE: Each missed rule case also sends gradient into the trigger's
  input, and the network restores it rather than learning the rule.
PROBLEM: The code contradicts this. s = relu(x1) - relu(x1 - 1) has zero derivative when x1 <= 0 (trigger off) and when x1 >= 1 (saturated; exact cancellation, which is why W0 passes). A rule case missed because the trigger is off therefore sends no gradient into the trigger. Only positions in the linear range 0 < x1 < 1 do, and those may still be answered correctly. 'The network restores it' is an untested inference.
SKEPTIC: The code confirms the auditor. With s = relu(x1) - relu(x1 - 1), ds/dx1 = 1[x1>0] - 1[x1>1], which is non-zero only for 0 < x1 < 1. A rule case missed because the trigger is fully off therefore sends no gradient into the trigger. 'Restores it' is untested. The broader 'fights back' point survives through linear-range positions, so this is minor.
FINAL: Rule positions where the trigger is only partly on (0 < x1 < 1) send gradient into the trigger's
  input that pushes it back up; a fully-off trigger sends none. We hypothesise that the network restores the trigger this way instead of learning the rule (not tested directly).

### [41] claims (minor) — REPORT_WSP.md
QUOTE: this drift seeds the detached and live
  trigger failures.
PROBLEM: An untested causal claim is stated as fact, in the limitations section, next to the admission that the control was not run. The frozen arm shows that suppressing a barely moves the trigger by itself. After install, the trigger reads lnf(h) through the install-time W_U[a], so it no longer equals 'the network's drive towards a' (W_U is trained). The same conflation appears in §5 ('a is never a training target, so the network's drive towards a falls'). In the detached arm, the trigger switching off coincides with absorption within one 25-step eval, so the order of events cannot be resolved.
SKEPTIC: This causal claim is stated as fact even though the control was not run. The frozen arm shows that a-suppression alone moves the trigger only a little: median body a-logit went from 18.32 to 9.75 at L1 while the trigger read off that body stayed at 0.95-0.97 from step 200 on. That small drop is still consistent with seeding a few positions, so 'untested' is the right fix, not 'refuted'. Minor, because it sits in limitations next to the missing-control caveat.
FINAL: we suspect this drift seeds the detached and live
  trigger failures (untested; on its own the drift moves the trigger little: in the frozen arm the body's a-logit fell by about 9 at level 1, yet the same trigger read off that body still fired at 0.95 or more of rule positions throughout).

### [42] claims (major) — REPORT_WSP.md
QUOTE: No training example tells the network to keep the trigger's input
  where it was.
PROBLEM: In the detached arm the logged data contradict this mechanism for W5. At q = 0 the detached trigger was still firing at the end in every model. Rule accuracy fell because the body's own z margin collapsed, which tracks the number of applied clean-input misfires. A drifting trigger fits only the live arm at level 3.
SKEPTIC: Recomputed for q = 0 at step 2000. The detached trigger still fired at 0.945-0.98 in every model: >= 0.95 in 23/24 at L1 and 22/24 at L3, never below 0.9. Yet F fell. Across detached models F tracks the change in body z margin (Spearman 0.85/0.90) and the applied clean misfires (Spearman -0.52/-0.71; median 201 misfires at L1 vs 0 for frozen). Only live L3 fits a drifting trigger (fire < 0.9 in 7/24; Spearman(F, fire) 0.97). The stated mechanism is therefore wrong for the detached arm.
FINAL: In the detached arm this was not the trigger switching off: at step 2000 it still fired at >= 0.9 of rule positions in every model (levels 1 and 3). Instead, F tracked the fall in the body's own z margin (Spearman 0.85 / 0.90) and the number of clean-input misfires the patch applied during training (Spearman -0.52 / -0.71). Only the live arm at level 3 fits a drifting trigger (fire < 0.9 in 7/24; Spearman(F, fire) 0.97). All of this is post hoc.

### [43] claims (major) — README.md
QUOTE: analysis at the time of the edit cannot tell the cases apart.
PROBLEM: Same overclaim as REPORT implication (3). The arms differ visibly in their wiring (a frozen 109,888-parameter copy; a stop-gradient). Only outputs, and gradients on the filtered training data, are identical, and only until the first few updates.
SKEPTIC: The same overclaim as REPORT implication (3). The wiring differs visibly (a frozen 109,888-parameter copy; a stop-gradient). Only outputs, and gradients on the filtered pools, are identical, and only until the first updates.
FINAL: the patched model's outputs and gradients at the time of the edit do not distinguish the cases (only the wiring does, so inspect it).

### [44] claims (major) — linkedin_draft.md
QUOTE: 3. Analyses done at the moment of the edit cannot tell these cases apart. The gradients are literally identical.
PROBLEM: Overclaim. Gradients are identical only on training data filtered to make them so, and only until training moves the weights. The cases differ in wiring that any inspection of the edit shows.
SKEPTIC: The gradient identity is engineered by filter_pool and lasts only until training moves the weights (bitwise divergence after 1-2 or 8-56 updates). The wiring distinguishes the cases.
FINAL: 3. The model's outputs and gradients at the moment of the edit do not distinguish these cases: on the training data they are bit-for-bit identical until training moves the weights. Only how the patch is wired distinguishes them, so check the wiring.

### [45] claims (minor) — linkedin_draft.md
QUOTE: We tested three versions of the same 134 numbers. At step 0 they give bit-for-bit identical outputs and identical gradients.
PROBLEM: Leaves out that the gradient identity is engineered by filtering the training data and lasts only until the first updates. Also hides that one version carries a full frozen copy of the network and another needs a stop-gradient, so they are not just 'versions of the same 134 numbers'.
SKEPTIC: The sentence leaves out that the gradient identity depends on the filtered training data. It also hides that one version carries a full frozen copy of the network and another needs a stop-gradient.
FINAL: We wired the same 134 numbers in three ways. Until training moves the weights, all three give bit-for-bit identical outputs, and identical gradients on the training data (which we filtered so that this holds). One version also keeps a frozen copy of the whole network; another blocks gradients in the training code.

### [46] claims (minor) — FINDINGS.md
QUOTE: (W0 gate exact; in-batch losses identical through the 100 warm-up steps in 96/96 cells).
PROBLEM: The same vacuous identity as in REPORT: lr = 0 for those 100 steps, so no weights change. The real divergence points are left out.
SKEPTIC: This repeats the vacuous identity: disp = 0 at step 100 in all 288 runs because lr = 0. Bitwise identity ends at steps 101-102 (frozen) and 108-156 (live vs detached).
FINAL: (W0 gate exact on the filtered training pools; no weight moves during the 100 lr-0 steps, so in-batch losses are trivially identical there; bitwise identity ends after 1-2 updates (frozen vs the others) and 8-56 updates (live vs detached)).

### [47] claims (minor) — FINDINGS.md
QUOTE: the live trigger keeps firing (0.89-0.99 of rule positions early, vs 0.16-0.21 detached)
PROBLEM: Leaves out the frozen comparator (0.97), which shows that continued firing is the default when there is no feedback loop. Read alone, the line suggests that gradient access keeps the trigger on.
SKEPTIC: The frozen comparator (0.966-0.973 in wsp_posthoc.txt) is missing. It shows that continued firing is the default without a feedback loop.
FINAL: the live trigger keeps firing (0.89-0.99 of rule positions early, vs 0.16-0.21 detached and 0.97 for the same trigger read off the frozen arm's body, which gets no gradient through it)

### [48] claims (minor) — REPORT_WSP.md
QUOTE: the network never learned the rule, so
  deleting the patch still removed the behaviour.
PROBLEM: 'Never learned' is too strong: the frozen body learned part of the rule. It is also only true at level 1; at the weakest force the median B is 0.21. Separately, deletion does not restore the original behaviour either, because the body mostly stops emitting a. README ('the network never learns the rule') and LinkedIn ('The network never learns the rule') repeat the wording.
SKEPTIC: Confirmed. Frozen median B is 0.21 at L0 (>= 0.1 in 15/24) and 0.03 at L1, and the z margin rose in 24/24 at L0-L2, so 'never' is too strong. Also, deleting the patch does not restore the original behaviour: at L1 the body emits a at a median 0.32 of rule positions, and neither a nor z at about 0.56.
FINAL: the network learned little of the rule (median B 0.03 at level 1, 0.21 at the weakest force), so
  deleting the patch still removed the behaviour (though not back to the original: at level 1 the patch-free body emitted neither a nor z at a median 56% of rule positions).

### [49] claims (minor) — REPORT_WSP.md
QUOTE: **A patch that is never wrong teaches nothing.**
PROBLEM: The frozen data contradict 'nothing'. The patch's residual error (1 - p(z) = 3e-2 ... 1e-4) is a non-zero signal, and the body's z margin grew in almost every frozen run. W4 shows less learning with more force, not zero learning.
SKEPTIC: Frozen z-margin gain was positive in 24/24 at L0-L2 (medians +6.26, +4.67, +2.86) and in 20/24 at L3, and median B at L0 was 0.21. The residual 1 - p(z) is a non-zero signal, and W4 shows less learning with more force, not none.
FINAL: **A patch that is always right teaches little.**

### [50] claims (minor) — REPORT_WSP.md
QUOTE: the trigger is switched off
  everywhere, and the patch becomes dead code.
PROBLEM: Not true for every detached run. In 4-6 of 24 detached runs per level the trigger still fires on a sizeable share of rule positions at step 2000: the patch is redundant there, not dead. 'It completes in about 75 steps' is a median for B >= 0.5; full absorption took up to 500 steps.
SKEPTIC: At step 2000 the detached trigger still fired at > 0.2 of rule positions (up to 0.60) in 4/6/5/4 of 24 models at L0-L3; these are the 'mixed' fates. In those models the patch is redundant, not dead.
FINAL: in most models the trigger is switched off
  almost everywhere (at step 2000 it fired at <= 0.2 of rule positions in 18-20/24 models per level, and still at 0.2-0.6, redundantly, in the other 4-6), and the patch becomes dead or redundant code.

### [51] claims (minor) — REPORT_WSP.md
QUOTE: Without a patch, the networks learn the rule in 120-130 steps. Detached patches were absorbed by step
  162-175, i.e. about 75 steps after the learning rate became non-zero (it is 0 for 100 steps, then warms up).
PROBLEM: The twin is counted from step 0 and the detached arm from lr onset, so the comparison reads as if the patch sped learning. On the same clock the twin needed 20-30 updates and detached 62-75: the patch slowed learning. '162-175' are medians (range 150-200), and B >= 0.5 is not full absorption. The README/LinkedIn phrase 'absorbs the rule within about 75 steps' has the same problem.
SKEPTIC: The two numbers are counted on different clocks. The twin reaches B >= 0.5 at step 120-130, i.e. 20-30 updates. Detached T_take medians are 162-175 (range 150-200), i.e. 62-75 updates. So the patch delayed learning, while the text reads as though it sped it. B >= 0.9 took up to 425/500/250/250 steps.
FINAL: The learning rate is 0 for 100 steps, then warms up. Without a patch, the networks learn the rule (B >= 0.5) 20-30 steps after it becomes non-zero (step 120-130). Detached patches were absorbed (B >= 0.5) a median 62-75 steps after
  it (step 162-175; range 150-200) and reached B >= 0.9 by step 250-500 in the slowest models, so the patch delayed rather than sped up learning of the rule.

### [52] claims (minor) — REPORT_WSP.md
QUOTE: Whether a given live patch is absorbed is close to a coin flip;
PROBLEM: Overstated. Between data streams, absorbed status flipped in 8/24 models, below the ~11/24 expected if the two streams were independent, and B correlates between streams. The heading '**The live arm depends on the learning rate and is chaotic.**' also asserts chaos, which was not tested (no minimal perturbations, no divergence rates); 'sensitive to the data stream' is what was measured.
SKEPTIC: Recomputed for level 1 live. 8/24 models were discordant between streams, against about 11/24 expected under independence, and the between-stream Spearman of B is 0.38. The outcome is partly predictable, so 'coin flip' overstates.
FINAL: Whether a given live patch is absorbed (B >= 0.5) changed with the data stream in 8/24 models (between-stream Spearman of B 0.38), so a single run's outcome is unreliable;

### [53] claims (minor) — REPORT_WSP.md
QUOTE: In partly absorbed live runs, the shift task (which needs the conjunction "task = shift and the symbol
  before a in the prompt") lagged the other three.
PROBLEM: True only at levels 1 and 3. At level 0 the sort task lagged most, and at level 2 sort lagged as much as shift. The conjunction explanation is untested.
SKEPTIC: Recomputed for live runs with 0.1 <= B < 0.9. L1 shift 0.10 and L3 shift 0.22, with the other tasks >= 0.98. At L0 sort lagged most (0.40 vs shift 0.84). At L2 both lagged (sort 0.00, shift 0.11). The conjunction explanation is untested.
FINAL: In partly absorbed live runs, the shift task lagged the other three at levels 1 and 3 (median B 0.10 and 0.22 vs >= 0.98); at level 0 the sort task lagged most (0.40 vs shift 0.84), and at level 2 both lagged (sort 0.00, shift 0.11). Why is untested (shift needs the conjunction "task = shift and the symbol
  before a in the prompt").

### [54] claims (minor) — REPORT_WSP.md
QUOTE: The term was popularised in November 2026 by reports of frontier language models that read and edit the raw weights of
small networks (see e.g. arXiv 2611.05341, which uses such edits to audit and repair small clinical networks).
PROBLEM: The cited source does not support 'popularised in November 2026'. arXiv 2611.05341 was submitted on 3 Nov 2026, two days before this report, and itself says the ability 'has recently been reported'. It defines weight-space programming as an ability of frontier language models to read and write the raw weights of small networks. In that work a language model (Claude Opus 5.5) audited networks blind and repaired them by changing a median of 3 existing parameters, which is quite unlike this report's data-calibrated add-on module. The descriptions of 2507.14198 and 2511.05852 (edits decaying under fine-tuning) are accurate.
SKEPTIC: Verified on arXiv. 2611.05341 was submitted 3 Nov 2026 and itself says the ability 'has recently been reported', so it gives no support for 'popularised in November 2026'. In that study Claude Opus 5.5 audited networks of 433-4,801 parameters blind and repaired each by changing a median of 3 existing parameters.
FINAL: In arXiv 2611.05341 (3 Nov 2026) the term names an ability recently reported in frontier language models: reading and writing the raw weights of
small networks. In that study a language model audited small clinical networks (433-4,801 parameters) blind and repaired each by changing a median of 3 existing parameters.

### [55] claims (minor) — REPORT_WSP.md
QUOTE: **The patch (134 numbers, written by hand, never trained).**
PROBLEM: 'Written by hand' overstates the hand-crafting. 128 of the 134 numbers are k*W_U[a] copied from the network. τ and k come from quantiles of trigger values on 12,000 generated sequences, and γ comes from a bisection on the model's logits to hit a target 1 - p(z). The construction is analytic and data-calibrated, and gradient descent never touches it. LinkedIn ('134 hand-set numbers') and the headline figure title ('Same 134 hand-written weights') repeat the wording.
SKEPTIC: calibrate() copies 128 of the numbers from k*W_U[a]. tau and k come from quantiles over 12,000 generated sequences, and gamma from an 80-step bisection on body logits. The patch is analytic and data-calibrated, not hand-chosen.
FINAL: **The patch (134 numbers, set analytically and calibrated on generated data, never trained).**

### [56] claims (minor) — README.md
QUOTE: that never showed the rule silently broke in-network patches but not frozen-reference ones.
PROBLEM: 'Broke ... but not frozen-reference ones' overstates the result. Median losses were 6-15 points, and frozen-reference patches also lost some accuracy at level 1.
SKEPTIC: At q = 0, F at 2000 had median (min): live 0.90 (0.48) / 0.93 (0.65), detached 0.85 (0.22) / 0.94 (0.11), frozen 0.97 (0.83) / 1.00 (0.99). Median losses are 6-15 points, and the frozen arm also lost some accuracy at L1, so 'broke ... but not' overstates.
FINAL: that never showed the rule lowered in-network patches' rule accuracy (median 0.85-0.94, worst 0.11) more than frozen-reference ones' (median 0.97-1.00, worst 0.83).

### [57] claims (minor) — REPORT_WSP.md
QUOTE: partial absorption (16/24). Post hoc, the
PROBLEM: The headline counts (frozen 1/24, detached 24/24, live 16/24) are for force level 1 at 3x lr only, and the summary bullets do not say so. Only the figure subtitle does. 'Partial absorption (16/24)' hides a bimodal split.
SKEPTIC: 16/24 is level 1 at Adam 3x only, and only the figure subtitle says so. The distribution is bimodal (B >= 0.9 in 7, B < 0.1 in 7). The count is 12-18/24 across levels and 7/24 at 1x lr.
FINAL: absorption (B >= 0.5) in 16/24 at force level 1 (B >= 0.9 in 7/24, B < 0.1 in 7/24; 12-18/24 across force levels; 7/24 at 1x the final learning rate). Post hoc, the

### [58] claims (minor) — linkedin_draft.md
QUOTE: - The patch reads the live network, with gradients flowing through it. The network keeps the patch firing and absorbs the rule only partly: 16/24.
PROBLEM: Presents a post hoc mechanism ('keeps the patch firing') as part of a pre-registered result. It also gives 16/24 without saying it depends on the learning rate and data stream. 'Absorbs only partly' misdescribes a split where 7/24 absorbed fully and 7/24 not at all.
SKEPTIC: The post hoc mechanism is presented as part of the result. The count depends on the recipe (16/24 at 3x, 7/24 at 1x, 6/12 SGD) and on the data stream (status flips in 8/24). 'Only partly' misdescribes a split of 7 fully absorbed and 7 not absorbed at all.
FINAL: - The patch reads the live network, with gradients flowing through it. The outcome varies: at a high learning rate the network took the rule over in 16/24 models, at a lower one in 7/24, and rerunning with a different data order changed the outcome in 8/24. Where it did not take over, the patch kept firing.

### [59] linkedin (major) — linkedin_draft.md
QUOTE: - The patch reads the live network, with gradients flowing through it. The network keeps the patch firing and absorbs the rule only partly: 16/24.
PROBLEM: Gets the live-arm outcome wrong and leaves out how unstable it is. "Absorbs the rule only partly: 16/24" reads as if 16 models partly absorbed it. In fact 16/24 is the count with B >= 0.5, and the 24 models are spread across the whole range. The post never says this arm is the least reproducible. A second data stream and a 3x lower learning rate both change the answer a lot, and the report itself calls a single live outcome 'close to a coin flip'. The live arm is also the closest of the three to an ordinary in-weight edit, so dropping its variability is the most consequential omission in the post.
SKEPTIC: Confirmed from the summary file. Live L1 (Adam 3x) at step 2000: B>=0.9 in 7/24, 0.1-0.9 in 10/24, B<0.1 in 7/24. 16/24 is the count with B>=0.5. The 'partly ... 16/24' wording blurs the bimodal spread. On the second data stream, 8/24 models cross 0.5 (zoo2/m0018 0.00->1.00, zoo7/b0075 1.00->0.00, zoo8/b0077 0.00->0.99). Adam 1x gives 7/24 and SGD 6/12. REPORT_WSP §4 itself calls a single live outcome 'close to a coin flip', and the post leaves all of this out. 'Keeps firing' is supported (fire_live median 0.91 at step 2000). The auditor's 'closest to an ordinary in-weight edit' is interpretation the data does not show, so I dropped it.
FINAL: - The trigger reads the live network, with gradients flowing through it. The network mostly kept the trigger firing instead of learning the rule, and the outcome was erratic: 7/24 learned the rule fully (at least 90% with the patch removed), 10/24 partly, 7/24 not at all (16/24 at 50% or more). Re-running on a different random draw of training batches moved 8 of the 24 across the 50% line (one from 0% to 100%, one from 100% to 0%). At a 3x lower learning rate only 7/24 reached 50%. For any single model, this version's outcome is close to a coin flip.

### [60] linkedin (major) — linkedin_draft.md
QUOTE: All of this happened while the model's visible behaviour did not change: with the patch in place, rule accuracy stayed near 100% throughout.
PROBLEM: 'Throughout' is false. In the detached arm, held-out rule accuracy with the patch in place dropped visibly during the hand-over in a large minority of models, as low as 61%. What the data supports is that the END states look the same. The dip is also a real, observable sign of absorption, which bears on the post's 'you can't tell' message. The error comes from REPORT_WSP.md line 14 and README.md line 16, which make the same claim.
SKEPTIC: 'Throughout' is false. In the detached arm, held-out F (patch in place) fell below 0.9 in 6/7/9/9 of 24 models at L0-L3, as low as 0.607 (L1: 0.611, zoo8/b0074 at step 150). Each dip lasted exactly one 25-step eval, at the moment of hand-over. The 10-step in-batch rule accuracy in the full run file fell below 0.9 in 14/24 detached L1 runs. Frozen F never fell below 0.994. At step 2000, F >= 0.979 in every run of every arm, so the claim holds for end states only. The dip is a brief but observable sign of absorption, which bears on the post's 'you can't tell from outside' message.
FINAL: At step 2,000, with the patch in place, every model in all three versions followed the rule at least 97% of the time. Only the remove-the-patch test showed which networks still depended on it. The hand-over was not fully silent, though. In the stop-gradient version, rule accuracy with the patch in place briefly dipped below 90% in 7/24 models while the network took over (to 61% at worst). The frozen version never fell below 99%.

### [61] linkedin (major) — linkedin_draft.md
QUOTE: In 24 small transformers, we wrote a rule directly into the weights: "whenever you are about to output symbol a, output z instead." The rule is 134 hand-set numbers, with no training. Then we kept training each model on data the patched model already got right. Every 25 steps we deleted the patch and asked: has the network learned the rule itself?
PROBLEM: Misleading by omission of the patch design. A practitioner reading this cannot follow the rest of the post. (1) 'Wrote a rule directly into the weights' suggests an in-place edit of existing weights, ROME-style. The patch is actually an add-on module: two extra ReLU units after the final LayerNorm. That is why it can be 'deleted'. (2) The patch stays fixed during later training; only the original network trains. An ordinary in-weight edit would itself be trained. (3) The post never explains the trigger, which reads the network's own score for a. Without that, 'reads the live network' and 'keeps the patch firing' in the bullets mean nothing. (4) It never says the training data contains the rule: 30% of sequences have z as the target. So 'has the network learned the rule itself?' looks like learning without supervision. An unpatched twin learns the rule from this data in 20-30 steps, which is the baseline for every absorption claim. (5) The numbers are not 'hand-set': 128 of the 134 copy W_U[a], and tau, k and gamma are computed by a calibration procedure.
SKEPTIC: Checked in wsp/patch.py. The Patch class is an add-on module: two ReLU units after lnf, 2x64 + 2 + 4 = 134 numbers. requires_grad is False (trainable_patch is None in every run), and only m.parameters() are optimised. The trigger reads W_U[a]·hn. Calibrate() computes tau and k from trigger quantiles, and gamma comes from bisection. gen() remaps a->z in rule targets, and q=0.3. 'Directly into the weights' suggests an in-place edit, but deletion is possible only because the patch is an add-on. Readers are not told that the training data contains the rule, or that the trigger reads the network's own score for a. Without that, 'reads the live network' and 'has the network learned the rule itself?' are opaque. The twin (warm=100, eval every 5) reaches B>=0.5 at steps 120-130, i.e. 20-30 steps after the learning rate switches on. The model has 109,888 parameters.
FINAL: We bolted a hand-built patch onto 24 tiny transformers (2 layers, ~110k parameters, trained on toy copy/reverse/sort/shift tasks) for one rule: "whenever you are about to output symbol a, output z instead." The patch is two extra ReLU units after the final LayerNorm: 134 numbers computed by formula, never trained. Its trigger reads the network's own score for a and, when it fires, moves the logit from a to z. We then kept training the original network on data the patched model already got right, with the patch itself held fixed. In 30% of the sequences the rule applies, with z as the target. Every 25 steps we removed the patch and asked: has the network learned the rule itself? (Without a patch, the same networks learn it from this data within 20-30 steps of the learning rate switching on.)

### [62] linkedin (major) — linkedin_draft.md
QUOTE: We tested three versions of the same 134 numbers. At step 0 they give bit-for-bit identical outputs and identical gradients. They end up in three different places:
PROBLEM: Two problems. (1) The identical gradients are presented as a finding, but they are an engineered property. The trigger is a clipped piecewise-linear unit, and the training pools were filtered so that it is strictly saturated or strictly off at every trained position. Its gradient is therefore exactly zero in every arm. On unfiltered inputs, the arms' gradients already differ at step 0. The frozen version also is not 'the same 134 numbers' alone: it needs a full frozen copy of the network running alongside. (2) The bullets that follow give counts (1/24, 24/24, 16/24) without saying they are for one of four force levels at Adam 3x lr, or that 'survived' means B >= 0.5. Those counts are exploratory descriptives. The pre-registered tests are paired comparisons, and the post does not say which is which.
SKEPTIC: The identity is engineered. s is clipped (the second unit is shifted by exactly 1), and filter_pool keeps only sequences with x1>1 at the rule position and x1<0 elsewhere. I reran li_grad.py on unfiltered data. For zoo2/m0006, 12 of 80,135 trained positions are in the linear range and max |grad_live - grad_detached| = 2.1e-5 (max |grad| 4.2e-3). The identity therefore does not hold in general, though the differences are small. This framing feeds takeaway 3. The counts below are also given without a definition or a level: they are the exploratory B>=0.5 'takeovers' at L1, Adam 3x. The pre-registered results are paired tests: W1 and W2 hold at 4/4 levels, W3 at 3/4 (level 0 Holm p = 0.011).
FINAL: We wired the same 134 numbers in three ways. On the training data, all three give bit-for-bit identical outputs and gradients at step 0. That is by design: the trigger is a clipped ReLU, and we kept only training inputs on which it is fully on or fully off, so at the start no gradient passes through it in any version. After 2,000 steps the three versions had ended in three different places. Pre-registered paired tests over the 24 models found higher rule accuracy with the patch removed for stop-gradient than for frozen or live at all four patch strengths, and for live than for frozen at three of four. The counts below are at the second of four strengths (Adam, 3x the base model's final learning rate). "Kept the rule" means at least 50% rule accuracy with the patch removed:

### [63] linkedin (major) — linkedin_draft.md
QUOTE: Takeaways for anyone editing weights:
1. "Undo" is not guaranteed. After further training, reverting an edit may no longer revert the behaviour.
2. Whether an edit stays load-bearing depends on how it is wired into training, not only on what it computes.
3. Analyses done at the moment of the edit cannot tell these cases apart. The gradients are literally identical.
PROBLEM: This is hype beyond the evidence. The result comes from one fixed add-on patch on 110k-parameter toy models, yet the takeaways are framed as general lessons 'for anyone editing weights'. Takeaway 3 is false as a general statement. Inspecting what the edit reads (a frozen copy, a stop-gradient, or a live read) tells the three versions apart immediately. The identical gradients hold only because the trigger and the training data were built to make them identical, and on unfiltered inputs they differ. The supported claim is narrower: output and gradient checks at install did not separate them, and a delete test after training did. The takeaways also miss the half of the result that cuts the other way: an edit that is never wrong (the frozen arm) suppresses learning, so the network stays dependent on it.
SKEPTIC: The post generalises from one fixed add-on patch on 110k-parameter toy models to 'anyone editing weights'. REPORT_WSP §6 lists exactly these scope limits. Takeaway 3 is false as a general statement. The wiring itself (a frozen copy, detach(), or a live read in Patched.forward) distinguishes the arms. The gradients are identical only because the trigger and the pools were built to make them identical; unfiltered, they differ (2.1e-5). The takeaways also omit the supported counter-result W4: an edit that is never wrong starves learning (Spearman negative in 24/24, p = 1.5e-6).
FINAL: What this suggests if you edit weights (a toy result, so treat these as hypotheses to test in your own setting):
1. After further training, removing an edit may no longer remove the behaviour, because the network may have learned it in the meantime. The reverse also happened: an edit that is never wrong starved the network of error signal, so the network stayed dependent on it.
2. Whether an edit stayed load-bearing depended on what its trigger reads and whether gradients reach it, not only on what it computes.
3. Output and gradient checks at the moment of the edit could not tell the three versions apart (here by design). Only a remove-the-edit test after training showed which networks still depended on it, so run that test.

### [64] linkedin (major) — linkedin_draft.md
QUOTE: Caveats: toy transformers (~100k parameters), one kind of rule (a logit remap), one training recipe. [secondary results: SGD / lower learning rate / training that never shows the rule]
PROBLEM: Not publishable as written. It contains an unfilled placeholder, and 'one training recipe' contradicts the sealed design, which used three recipes: Adam 3x, Adam 1x and SGD-momentum. Key caveats are missing. (a) The patch is an add-on module held fixed, not an edit of existing weights, and a trainable patch was not tested. (b) The trigger reads the very logit the rule suppresses, so training on rule-following data (where a is never a target) drifts the trigger itself. No control with an unrelated symbol was run, which REPORT_WSP §6 lists as a limitation. (c) The horizon is 2000 steps and 30% of the data is rule data. (d) The secondary results that belong here are not stated: the frozen-vs-stop-gradient contrast held under all three recipes but the live arm did not, and with q=0 the in-network patches broke silently.
SKEPTIC: This line contains an unfilled placeholder. 'One training recipe' contradicts the sealed design, which used Adam 3x, Adam 1x and SGD-momentum (REPORT §6 says 'one training recipe family'). The line also leaves out key caveats: the patch is a fixed add-on, not an edit of existing weights; a trainable patch was not tested; the trigger reads the logit it suppresses, with no unrelated-symbol control; 2000 steps; 30% rule data. I verified the secondary numbers in wsp_posthoc.txt. Kept-rule counts at L1 are: Adam 3x detached 24/24, frozen 1/24, live 16/24; Adam 1x 24/24, 0/24, 7/24; SGD 11/12, 0/12, 6/12. At q=0, the minimum F with the patch in place is 0.11 (detached L3) against frozen at least 0.83. In the full run file the a-logit falls in every arm by step 2000 (frozen median about 8-10), consistent with a never being a target.
FINAL: Caveats: tiny transformers (~110k parameters) on toy tasks. One hand-built patch, added alongside the network rather than edited into its existing weights, and held fixed during training (a trainable patch was not tested). One rule, whose trigger reads the very logit it suppresses. Since a is never a training target, ordinary training drifts the trigger, and no control with an unrelated symbol was run. Three optimiser settings (Adam at 3x and 1x the base's final learning rate, and SGD-momentum), 2,000 steps, 30% rule data. The stop-gradient vs frozen contrast held under all three settings (24/24, 24/24 and 11/12 kept the rule, against 1/24, 0/24 and 0/12). The live version's result did not hold (16/24, 7/24, 6/12). When training never showed the rule, the two live-reading patches silently lost rule accuracy in some models (as low as 11% with the patch in place), while the frozen one held (at least 83%).

### [65] linkedin (minor) — linkedin_draft.md
QUOTE: - The patch reads a frozen snapshot of the network. The network never learns the rule, and deleting the patch removes the behaviour. The rule survived deletion in 1/24 models.
PROBLEM: 'Never learns the rule' is contradicted by the data at the same force level: 10/24 frozen models learned the rule partly, one of them to 93%. The bullet also gives no mechanism (gradient starvation), so readers cannot see why this version shields. It also omits that 'reads a frozen snapshot' means running a full frozen copy of the network alongside it.
SKEPTIC: Frozen L1 at step 2000: B<0.1 in 14/24, 0.1-0.9 in 9/24, one model at 0.928, median 0.03. So the network 'barely' learns the rule rather than 'never' learning it. The count of 1/24 is correct (3/1/0/0 across the four levels). The bullet also omits the mechanism (gradient starvation, W4) and that a full frozen copy of the network runs alongside (deepcopy in run()).
FINAL: - The trigger reads a frozen copy of the original network, run alongside it. The patch is then never wrong, so the rule examples give almost no error signal and the network barely learns the rule (median 3% rule accuracy with the patch removed; stronger patches suppressed learning more). Removing the patch removes the behaviour: 1/24 kept the rule.

### [66] linkedin (minor) — linkedin_draft.md
QUOTE: - The patch reads the live network, but no gradient flows through it. The network absorbs the rule within about 75 steps, and deleting the patch changes nothing: 24/24.
PROBLEM: The 'about 75 steps' does not say where the count starts. It is measured from the point the learning rate becomes non-zero (after 100 lr-0 steps, during the warm-up ramp), not from the start of training. It also uses the 50% threshold, at 25-step resolution. There is no baseline either: an unpatched twin learns the rule in 20-30 steps, so this patch only delays learning by about 45-50 steps. The bullet also gives no mechanism, so 'absorbs' is opaque. The end-state claim itself is supported.
SKEPTIC: The end state is supported: B >= 0.998 in 24/24 and F-B max 0.000 at L1. 'Within about 75 steps', however, counts from the learning-rate switch-on at step 100, after 100 steps at lr=0; it is a median, and the slowest model took 100 steps. At L1, T_take is 150-200 at 25-step resolution (median 175). The bullet gives no mechanism, so 'absorbs' is opaque. (The twin baseline goes in the setup paragraph.)
FINAL: - The trigger reads the live network, behind a stop-gradient. Training lowers the network's own score for a (a is never a target), so the trigger starts to miss. The misses teach the network the rule, and the patch becomes dead code. 24/24 kept the rule (at least 99.8% with the patch removed), 50-100 steps after the learning rate switched on.

### [67] linkedin (minor) — linkedin_draft.md
QUOTE: Weight-space programming means editing a network's weights directly instead of training it, and it is having a moment.
PROBLEM: 'It is having a moment' is a hype claim that the sealed results do not support, and the post cites nothing for it. The only basis is REPORT_WSP §1, which points to November 2026 reports and arXiv 2611.05341. I could not check those offline, and the post links neither.
SKEPTIC: 'Having a moment' is uncited hype. Its only basis is REPORT_WSP §1's mention of November 2026 reports (arXiv 2611.05341), which the post neither links nor qualifies. The rest of the opening paragraph already supplies the motivation, so the replacement covers this sentence only, to avoid duplicating the text that follows.
FINAL: Weight-space programming means writing behaviour into a network's weights by hand instead of training it in.

### [68] linkedin (minor) — linkedin_draft.md
QUOTE: Code, pre-registration and raw results: [link]
PROBLEM: This is an unfilled placeholder, and 'raw results' overstates what the repo contains. The full per-step run file is gitignored and not committed, and neither are the base network checkpoints. The repo has a compact per-run summary that reproduces the registered analysis.
SKEPTIC: The [link] is an unfilled placeholder. 'Raw results' overstates what is shared: git check-ignore shows runs_wsp/** is ignored (sealed.jsonl is not committed) and no .pt checkpoints are tracked. The committed data is the per-run summary (results/wsp_sealed_summary.jsonl.gz) plus the prereg and posthoc outputs. The pre-registration commit b0b5161 (21:12:05) predates the sealed code 218c931 (21:15:32) and the sealed log start, so 'pre-registration' holds.
FINAL: Code, the pre-registration (committed before any sealed run) and per-run summary results that reproduce the numbers above: [link]
