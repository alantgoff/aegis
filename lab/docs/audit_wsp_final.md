# Final claims audit: refinements (addendum f), language-model replication (addendum h) and the LinkedIn post (2026-11-06)

Three auditors (numbers; claims; LinkedIn post), each followed by a skeptic who re-checked every finding against the
results files. 66 raised, 63 confirmed, all confirmed ones adopted (the post was rewritten
as a whole from them). Registered verdicts did not change.

### [0] CONFIRMED numbers (major) — REPORT_WSP.md
QUOTE: 0.61 and 0.67 in the two with the most training data
PROBLEM: This is false. The two live-arm exceptions are help/assist (372 training rule answers, the most) and provide/offer (106, only 4th of 8). often/frequently (168) and make/create (155) have more training data and ended at live B 0.00 and 0.01. The sentence implies a data-size effect that the data contradict.
SKEPTIC: Checked in wsp_lm_h.jsonl.gz and the pair .pt files. Training rule answers: help 372, often 168, make 155, provide 106 (rule positions: 433, 175, 169, 135). The two live exceptions are help (largest pool) and provide (4th). often and make have larger pools and ended at live B 0.000 and 0.014. So the sentence is false and implies a pool-size effect that the data do not show.
FINAL: 0.61 (provide/offer, 106 training rule answers) and 0.67 (help/assist, 372, the largest pool); the next-largest pools (often/frequently 168, make/create 155) stayed at 0.00 and 0.01, so the exceptions do not simply track pool size

### [1] CONFIRMED numbers (major) — REPORT_WSP.md
QUOTE: With the module in place, every model
PROBLEM: In the summary this sentence now comes straight after the language-model sentences, so 'every model ended at >= 0.97 rule accuracy, so the end state hid all of this' reads as covering the LM. It is false there, and it contradicts §3c ('ended at 0.84 and 0.87') and the sentence just before it ('the handover was visible').
SKEPTIC: The sentence now comes right after the LM sentences, so it reads as covering the LM. LM F at step 600: detached 0.78-0.93, live 0.82-0.96, which contradicts both §3c and the sentence before it. The claim holds only for the toy addendum-e runs that showed the rule: lowest end F 0.979 across Adam 3x, Adam 1x and SGD. It does not hold for the q = 0 runs or the 6000-step live runs (0.955).
FINAL: In the toy networks (main runs), with the module in place, every model

### [2] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: only the W tests in §3 were registered.
PROBLEM: This label is wrong after the update. F1-F2 (addendum f) and G0, H1-H4 (addendum h) were also registered. Some numbers in §3b are exploratory by registration (q = 0.01, 6000 steps, extra streams), and the §3c bullets mix registered descriptives with post hoc ones.
SKEPTIC: Addendum f registered F1 (primary) and F2 (secondary), and addendum h registered G0 and H1-H4, so the label is now wrong. The error understates what was registered rather than overstating it, so I rate it minor.
FINAL: only W0-W6 (§3), F1-F2 (§3b) and G0 and H1-H4 (§3c) were registered tests; the q = 0.01, 6000-step and extra-stream results in §3b were registered as exploratory, the §3c bullets are registered descriptives or post hoc, and §4 is exploratory or post hoc.

### [3] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: No run tested edits to existing parameters, modules that read earlier features,
  or modules that are themselves trained.
PROBLEM: This limitation contradicts §3b. Addendum f's trainable-patch condition (F2) trained the module itself in 72 runs.
SKEPTIC: The trainable-patch condition (F2: 72 runs, level 1, q = 0.3) did train the module, and §3b and the summary report it. The limitation therefore contradicts the report. It errs on the conservative side, so I rate it minor.
FINAL: No run tested edits to existing parameters or modules that read earlier features; a module that itself trains was tested only in one toy condition (§3b, F2: level 1, q = 0.3), never on the language model.

### [4] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: and when as few as 1-3% of the training examples showed the rule (pre-registered refinements, §3b)
PROBLEM: The label is wrong for 1%. Addendum f registered q = 0.01 as exploratory; F1 covers only q = 0.03 and 0.1. At 1% the contrast also held less fully: 21/24 detached models kept the rule, not 24/24.
SKEPTIC: Addendum f says 'q = 0.01 is exploratory'; F1 covers only q = 0.1 and 0.03. At 1% the contrast still held (21/24 vs 0/24 with B >= 0.5; detached > frozen in 23/24), so the issue is the label, not the direction.
FINAL: and when as few as 3% of the training examples showed the rule (pre-registered refinements, §3b; at 1%, an exploratory condition, detached > frozen in 23/24 models)

### [5] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: | 1% | 21/24 (median B 0.92) | 16/24 | 0/24 |
PROBLEM: The table sits under a 'pre-registered addendum f' heading, but this row is an exploratory condition and carries no label.
SKEPTIC: The table sits under the 'pre-registered addendum f' heading and labels its other non-F1 rows ('addendum e'), but the 1% row is an exploratory condition and carries no label. The numbers match wsp_refine_prereg.txt.
FINAL: | 1% (exploratory) | 21/24 (median B 0.92) | 16/24 | 0/24 |

### [6] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: There the live wiring mostly behaved like the
frozen one,
PROBLEM: The summary reports the LM result as support ('It also held...') but never says that a registered primary test, H4, failed. The bold header still reads 'all registered tests (W0-W6) supported'.
SKEPTIC: The summary now covers addendum h but never says that a registered primary test (H4, live > frozen) failed: Holm p = 0.0781, NOT SUPPORTED. The header's 'all registered tests (W0-W6) supported' applies only to addendum e. The prose implies the result without naming the failed test.
FINAL: There the live wiring mostly behaved like the
frozen one (the registered test H4, live > frozen, was not supported; p = 0.078),

### [7] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: swap the unpatched model learned from the same data in about 75 steps.
PROBLEM: '75 steps' is the eval step index, which includes the 20 lr-0 steps. The model crossed B >= 0.5 about 55 steps after the learning rate switched on. §4 counts toy steps from when the learning rate leaves 0, so the two conventions clash.
SKEPTIC: In swap_lm.py the lr is 0 for st < 20, and the eval at step st comes after st updates. Twin T(B >= 0.5) per pair is 50, 50, 50, 75, 75, 75, 100, 100 (median 75), so it crossed within about 55 steps of the lr switching on. In the same paragraph the toy is counted from the lr switch-on ('50-100 steps after the learning rate switched on'), so the two counts are not comparable.
FINAL: swap the unpatched model learned from the same data within about 55 steps of the learning rate switching on.

### [8] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: the model learned the swap in a median 75 steps (B 0.62-0.93
PROBLEM: Same step convention issue: the count includes the 20 lr-0 steps. It also merges a time to B >= 0.5 with the end-of-run range.
SKEPTIC: Same counting issue: 'learned in 75 steps' counts 20 lr-0 steps in which no weight moves. The 'merged quantities' point is weak, because the text says 'at step 600' for the range (twin B 0.62-0.93 at 600 checked; median at step 75 is 0.68). The duration wording is still worth fixing.
FINAL: the model learned the swap (B >= 0.5) by a median step 75, within about 55 steps of the learning rate leaving 0 (B 0.62-0.93

### [9] REJECTED numbers (minor) — REPORT_WSP.md
QUOTE: no-module model), a median of about 190 steps in.
PROBLEM: The value is right (median 187.5) but uses the absolute step index, including the 20 lr-0 steps. §4 counts from the lr switch-on.
SKEPTIC: Detached T(B >= 0.5) is 100, 100, 150, 175, 200, 225, 250, 250, median 187.5, so the value is right. '190 steps in' names a point in the run, not a duration, so it is literally accurate. It also uses the same absolute step index as the twin's step 75 in §3c. No change needed; once the twin sentence is fixed, the two stay consistent.
FINAL: no-module model), a median of about 190 steps in.

### [10] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: 86% of rule positions at the end.
PROBLEM: Addendum h requires reporting, whatever the outcome, 'B at the first rule position' and 'firing of the trigger read off the trained model'. §3c gives firing for the live arm only and omits B_first. The omitted frozen-arm value changes how the 86% reads: unlike the toy, training alone switched about half of the LM trigger off.
SKEPTIC: Addendum h registers firing and B at the first rule position as reported whatever the outcome. §3c gives live firing only and omits B_first. Frozen-arm firing read off the trained body is 0.40-0.67 (median 0.51) and detached is 0.06. That is very different from the toy's 0.96-0.97, so the 86% lacks its baseline, which the LM review (docs/review_llm_wsp.md) explicitly asked to report. B_first medians: frozen 0.02, detached 0.77, live 0.05.
FINAL: 86% of rule positions at the end, against a median 0.51 (0.40-0.67) for the same readout in the frozen arm, whose body gets no gradient through the trigger, and 0.06 in the detached arm; so unlike in the toy (0.96 at step 2000), training without a gradient path switched the trigger off at about half the rule positions. B at the first rule position was close to B (frozen 0.02, detached 0.77, live 0.05).

### [11] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: - **Crutch.** Trained on the same data without any module,
PROBLEM: None of the §3c bullets says whether a number is registered or not. Some are registered 'reported whatever the outcome' descriptives (min F, firing, A_left, displacement, C). Others are post hoc (times to B >= 0.5, per-pair comparisons such as the data-size remark). §3 and §3b label theirs; §3c does not.
SKEPTIC: §3c mixes three kinds of number without labels: registered 'reported whatever the outcome' descriptives (B_first, min F, firing, A_left, displacement, clean accuracy), post hoc items (twin and detached times to B >= 0.5, the per-pair pool-size remark) and pilot numbers. §3 and §3b label theirs.
FINAL: - Descriptives below are registered to be reported (not tested); times to B >= 0.5 and per-pair comparisons are post hoc.
- **Crutch.** Trained on the same data without any module,

### [12] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: " use" behaved like the sealed pairs (detached 0.51); " also" was not absorbed within
  600 steps in any arm (detached 0.00), its trigger still firing at 0.93.
PROBLEM: ' use' kept the ordering but absorbed less, and later, than every sealed pair. ' also' was learned by the no-module arm, so 'not absorbed in any arm' hides the fact that the crutch held even in the detached arm.
SKEPTIC: Pilot data check out. use: detached 0.507 with T = 425, below the sealed range (0.62-0.92, T 100-250), though the ordering matches. also: none 0.90 (T 125), detached/frozen/live 0.00, detached fire 0.93. 'Any arm' literally includes the no-module arm, which learned the swap, and omitting that hides that the crutch held even under detached.
FINAL: " use" showed the sealed pairs' ordering but weaker, later absorption (detached 0.51, reaching 0.5 only at step 425 vs 100-250 for the sealed pairs); " also" was not absorbed within
  600 steps in any module arm (detached 0.00), its trigger still firing at 0.93, although the no-module model learned it (0.90).

### [13] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: which live models are absorbed is largely a matter of chance, while the share
PROBLEM: The counts do not support 'largely a matter of chance'. More models are absorbed in all or none of the four streams than random assignment would give.
SKEPTIC: Recomputed: per-stream counts are 16/14/13/14, and models absorbed in 0-4 streams are 1, 6, 6, 5, 6. A stream-count-preserving permutation (200k draws) gives 3.58 models expected in the 0-or-4 categories vs 7 observed (p = 0.045). Variance of k is 1.48 vs 0.96 expected (p = 0.031). There is some per-model consistency, so 'largely a matter of chance' overstates. The evidence is modest, so I soften the auditor's 'not at random' to 'not purely chance'.
FINAL: which live models are absorbed varies a lot between streams but is not purely chance (7 models were absorbed in all or none of the four streams vs 3.6 expected if streams were independent; post hoc permutation p ≈ 0.045), while the share

### [14] CONFIRMED numbers (minor) — REPORT_WSP.md
QUOTE: `results/wsp_lm_hpilot.jsonl.gz`.
PROBLEM: The provenance names the committed LM results but gives no reproduction command. The analysis script cannot read the committed .gz file: it uses open() and crashes. Reproducing is possible only after decompressing.
SKEPTIC: Confirmed. patch_fate_lm.load uses open() and crashes on the committed .gz with UnicodeDecodeError (0x8b). After zcat, the --prereg output is byte-identical to results/wsp_lm_prereg.txt. runs_llm_wsp/h.jsonl is not committed, so no committed reproduction path is documented, unlike the toy results.
FINAL: `results/wsp_lm_hpilot.jsonl.gz`. `zcat results/wsp_lm_h.jsonl.gz > h.jsonl && python analysis/patch_fate_lm.py h.jsonl --prereg` reproduces `results/wsp_lm_prereg.txt` exactly (the script does not read .gz).

### [15] CONFIRMED numbers (minor) — README.md
QUOTE: (21/24 vs 0/24) of the training examples showed the rule (pre-registered refinements, addendum f).
PROBLEM: The 1% condition is labelled as pre-registered, but addendum f registered it as exploratory. F1 covers only 3% and 10%.
SKEPTIC: Same labelling issue as the report summary: addendum f registered q = 0.01 as exploratory, and F1 covers 3% and 10%. The counts (24/24 vs 0/24 at 3%, 21/24 vs 0/24 at 1%) are correct.
FINAL: (21/24 vs 0/24; an exploratory condition) of the training examples showed the rule (pre-registered refinements, addendum f; F1 covers 3% and 10%).

### [16] CONFIRMED numbers (minor) — README.md
QUOTE: brief dip in some models (below 0.9 in 7/24 stop-gradient models at level 1, lowest 0.61).
PROBLEM: This sentence now directly follows the 1%/3% and trainable claims, which are also level 1, but its numbers come only from the main q = 0.3 condition. At low exposure, most models dipped, and dipped deeper.
SKEPTIC: Recomputed. Main condition level 1 detached: 7/24 below 0.9, lowest 0.611, which is correct. But the sentence now follows the 1%/3% claims, and there dips were more common and deeper: detached 13/24 at q = 0.03 (lowest 0.633) and 13/24 at q = 0.01 (lowest 0.576); live 19/24 at q = 0.01 (lowest 0.453). They were still brief (median 1-2 evals), and every run ended at >= 0.969.
FINAL: brief dip in some models, more at low exposure (main condition: below 0.9 in 7/24 stop-gradient models at level 1, lowest 0.61; at 1-3% exposure in 13/24 stop-gradient models, lowest 0.58, and at 1% in 19/24 live models, lowest 0.45).

### [17] CONFIRMED numbers (minor) — FINDINGS.md
QUOTE: F1 supported (Holm p = 1.8e-5)
PROBLEM: The same sentence lists 1% next to the registered F1 conditions without saying 1% is exploratory.
SKEPTIC: The same FINDINGS sentence lists 1% next to the F1 conditions without saying it is exploratory (addendum f: 'q = 0.01 is exploratory'). The numbers are correct.
FINAL: F1 (3% and 10%) supported (Holm p = 1.8e-5; the 1% condition is exploratory)

### [18] CONFIRMED numbers (minor) — FINDINGS.md
QUOTE: the share is stable (~0.6), which model is not.
PROBLEM: This overstates the randomness. Per-model absorption is more consistent across streams than chance.
SKEPTIC: Same recomputation as for the report: per-stream counts are 13-16 of 24, and there is more all-or-none consistency per model than chance (7 vs 3.6, p ≈ 0.045). 'Which model is not [stable]' is roughly right, but the implication that it is random is overstated.
FINAL: the share is stable (~0.6, 13-16 of 24 per stream); which model varies a lot but not purely by chance (7 models absorbed in all or none of the streams vs 3.6 expected; post hoc p ≈ 0.045).

### [19] CONFIRMED numbers (minor) — linkedin_post.md
QUOTE: the same training without any patch taught the swap within ~75 steps (80% of cases)
PROBLEM: This merges two different quantities. ~75 is the median eval step at which B first reached 0.5 (it includes 20 lr-0 steps). 80% is the median B at step 600. At step 75 the median was 0.68.
SKEPTIC: The sentence conflates two quantities. 75 is the median eval step at which twin B first reached 0.5 (it includes the 20 lr-0 steps); 80% is the median B at step 600. The median twin B at step 75 is 0.68. As written, it reads as 80% reached within 75 steps.
FINAL: the same training without any patch taught the swap at most rule positions within about 55 steps of the learning rate switching on (80% of them by the end of the 600 steps)

### [20] CONFIRMED numbers (minor) — linkedin_post.md
QUOTE: rule accuracy with the patch in place dipped to ~70–80% along the way.
PROBLEM: 'Dipped ... along the way' implies it recovered, but it did not: F ended well below the frozen arm. The range also leaves out the live arm's lower minimum.
SKEPTIC: Median min F is detached 0.76 (0.60-0.80) and live 0.66 (0.59-0.79), so the live arm falls outside '~70-80%'. 'Dipped along the way' implies recovery, but F ended at median 0.84 (detached) and 0.87 (live) against 0.99 frozen.
FINAL: rule accuracy with the patch in place fell to a median low of 76% with the no-gradient wiring (66% with gradients) and ended at 84% (87%), against 99% for the frozen patch.

### [21] REJECTED numbers (minor) — linkedin_post.md
QUOTE: At the moment of the edit, both give identical outputs and identical gradients on the training data (by design).
PROBLEM: The post applies this to both models, but gradient identity was verified only in the toy (W0). The LM gate G0 compared in-batch losses only, which agreed to 1.9e-7 rather than being identical; the LM review notes that nothing compares parameter gradients.
SKEPTIC: The post says '(by design)' and does not claim verification. In swap_lm.py the frozen arm uses the cached install-time x1 and the detached arm uses x1(body.detach()), so neither has a gradient path through the trigger and gradients are identical by design. At install-time parameters (steplog steps 0-19), frozen and detached losses agree to <= 1.6e-9. The 1.9e-7 the auditor cites is the loss after the first real update, not 'at the moment of the edit'. That is not a material misstatement for a LinkedIn post, and the auditor's '2e-7' replacement would itself misplace that number.
FINAL: At the moment of the edit, both give identical outputs and identical gradients on the training data (by design).

### [22] CONFIRMED numbers (minor) — linkedin_post.md
QUOTE: In the small transformers this held even when only 1–3% of the further-training examples showed the rule
PROBLEM: This follows '24/24 kept the rule'. At 1% it was 21/24, and 1% was an exploratory condition.
SKEPTIC: The sentence follows 'Small transformers: 24/24 kept the rule', so 'this held' reads as 24/24. At 1% it was 21/24 (B >= 0.5), and 1% was an exploratory condition (addendum f). At 3% it was 24/24.
FINAL: In the small transformers this held when only 3% of the further-training examples showed the rule (24/24; 21/24 at 1%, an exploratory condition)

### [23] CONFIRMED numbers (minor) — linkedin_post.md
QUOTE: • The patch was an add-on held fixed, not an edit to existing weights, and each test used one training recipe.
PROBLEM: This contradicts the post itself ('when the patch's own numbers were also trained') and the toy results, which used three optimiser settings and a 6000-step horizon.
SKEPTIC: 'Held fixed' contradicts the post's own trainable-patch sentence (F2). 'Each test used one training recipe' is true only per test. It reads as no recipe variation, but the toy contrast was tested under Adam 3x, Adam 1x and SGD-momentum (W6), which the README and REPORT scope describe as three closely related optimiser settings.
FINAL: • The patch was an add-on, not an edit to existing weights, held fixed in all but one condition; the small transformers used three closely related optimiser settings, and the language model one.

### [24] CONFIRMED claims (major) — REPORT_WSP.md
QUOTE: frozen one, and the handover was visible in the patched model's behaviour. With the module in place, every model
PROBLEM: The headline summary contradicts itself. The sentence right after the language-model sentences says 'every model ended at >= 0.97 rule accuracy, so the end state hid all of this'. That is false for the LM, where both in-network wirings ended with clearly lower rule accuracy with the module in place. The sentence also suggests that only a 'handover' (detached) showed in behaviour, but the live arm lost rule accuracy without handing over.
SKEPTIC: Confirmed from wsp_lm_h.jsonl.gz. F at step 600 is 0.776-0.932 for detached and 0.824-0.958 for live, against 0.966-1.000 for frozen. Both arms are below frozen in 8/8 pairs (medians 0.84, 0.87 and 0.99). The next sentence, 'every model ended at >= 0.97', comes right after the LM sentences, so it reads as covering the LM, and there it is false. 'The handover was visible' also does not fit the live arm (median B 0.05): it lost F without handing over. The toy claim holds at 2000 steps (lowest end F with q > 0 is 0.969), so the replacement limits that sentence to the toy.
FINAL: frozen one in the undo test (B 0.05 vs 0.03), but with the module in place both in-network wirings lost part of the swap and had not recovered it by 600 steps (median rule accuracy 0.84 stop-gradient, 0.87 live, vs 0.99 frozen; below frozen in 8/8 pairs each). In the toy, at 2000 steps and with the module in place, every model

### [25] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: pretrained language model with an exact word swap (§3c, 8/8 word pairs), where the frozen module additionally blocked a
PROBLEM: 'An exact word swap' generalises beyond what was built. The margin module implements 'say Z wherever you would say A' exactly only on the answers that survived the filter. On the model's own answers that contain A, the module leaves A in place where the margin is in (0, 0.25) and only partly swaps where it is in [0.25, 0.5]. Those answers were removed from both training and test. After the withdrawn addendum g, the rule's description is exactly the point a reader will check.
SKEPTIC: In build_pair, any answer with a position in the ramp (margin 0.25-0.5) or an un-swapped A (margin 0-0.25) is dropped. Per swap_pairs.json, 26-44% of the base answers that contain A were dropped (help 154/590 through often 182/414). So the module is exact only on the filtered pools, and §3c says as much. The summary's bare 'exact word swap' over-generalises. This is a compression problem, not an invalid result, so minor.
FINAL: pretrained language model with a word swap that the module implements exactly on the filtered training and test answers (§3c, 8/8 word pairs), where the frozen module additionally blocked a

### [26] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: - **Absorbed.** With the stop-gradient wiring, the model took the swap over in 8/8 pairs (B 0.62-0.92, close to the
PROBLEM: 'Took the swap over' reuses the toy's language for full absorption (B >= 0.998 in 24/24), but here absorption was partial. Median B is 0.75, only 2/8 pairs reach 0.9, and the toy's registered W1 side condition (median B_detached >= 0.9) would have failed. The review asked the write-up to say whether the toy's magnitude criteria would have been met; it does not. Meanwhile the patched detached model itself lost 7-22% of the swap.
SKEPTIC: Verified: detached B 0.62-0.92, median 0.75, and only help (0.900) and provide (0.919) reach >= 0.9. The toy's W1 side condition (median B_detached >= 0.9) would have failed, and the report does not say so. In the toy, 'took over' meant B >= 0.998. Here the arm reached about the twin's level (0.62-0.93). The numbers sit right beside the wording, so this is minor.
FINAL: - **Absorbed, partly.** With the stop-gradient wiring, the model learned the swap in 8/8 pairs, but only partly (B 0.62-0.92, median 0.75; B >= 0.9 in 2/8, so the toy's W1 condition of median B >= 0.9 would not have been met; about as far as the

### [27] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: - **Live wiring.** Unlike the toy, the live wiring behaved like the frozen one in most pairs: B 0.00-0.17 in 6/8 pairs,
PROBLEM: Two framing problems. (1) 'Unlike the toy' overstates the contrast: in the toy at 1x the learning rate, the live arm also mostly looked like frozen. (2) 'Behaved like the frozen one' is true only of the undo test B. With the module in place, the live arm lost rule accuracy in every pair, while frozen did not. The trigger reading and the residual A preference also differ sharply from frozen. So live was neither shielded like frozen nor absorbed.
SKEPTIC: 'Behaved like the frozen one' is true of B only. With the module in place, live F is below frozen in 8/8 pairs (end 0.82-0.96 vs 0.97-1.00; median min 0.66 vs 0.98). Fire (0.86 vs 0.51) and A_left (0.88 vs 0.60) also differ. In the toy at 1x the learning rate, live was 7/24 with median B 0.07, so 'unlike the toy' is true only of the main setting. Minor, because the bullet is about B and the F numbers come two bullets later.
FINAL: - **Live wiring.** Unlike the toy at its main learning rate (though much like the toy at 1× its final learning rate: 7/24 absorbed, median B 0.07), the live wiring was close to the frozen one in the undo test in most pairs, though not with the module in place (below): B 0.00-0.17 in 6/8 pairs,

### [28] CONFIRMED claims (major) — REPORT_WSP.md
QUOTE: 0.61 and 0.67 in the two with the most training data.
PROBLEM: This is factually wrong and plants a spurious explanation, that live absorption tracks the amount of training data. The two live-absorbed pairs are help (largest rule pool) and provide (fourth largest). The second and third largest pools ended near 0.
SKEPTIC: Factually wrong. n_train_rule is help 372, often 168, make 155, provide 106, and so on. The two live-absorbed pairs are help (largest pool, B 0.67) and provide (fourth largest, B 0.61). often and make, the second and third largest, ended at 0.00 and 0.01. Spearman(live B, pool size) = 0.27. The sentence suggests an explanation the data do not support.
FINAL: 0.67 (help, the largest rule pool, 372 training answers) and 0.61 (provide, the fourth largest, 106); pool size did not order the live outcomes (often and make, with 168 and 155, ended at 0.00 and 0.01).

### [29] CONFIRMED claims (major) — REPORT_WSP.md
QUOTE: minimum 0.76 detached, 0.66 live; 0.98 frozen) and ended at 0.84 and 0.87 (frozen 0.99).
PROBLEM: Addendum h registered 'firing of the trigger read off the trained model' as reported whatever the outcome. §3c reports it only for the live arm. The frozen-arm value is the key control, and it overturns the toy story carried in §5. In the LM, training alone switched off the trigger reading at about half the rule positions in a body that never learned Z. In the toy the same reading stayed at 0.96, and §5 uses that to say drift 'on its own moved the trigger little'. Without this number, readers cannot see that the LM handover may be seeded by a different, much larger drift. The review of g explicitly said 'do not claim the mechanism matches the toy'. The bullet also omits that live ended below frozen in every pair.
SKEPTIC: Addendum h registers 'firing of the trigger read off the trained model' as reported whatever the outcome. §3c gives it only for live (0.86). Recomputed: the frozen arm's body fires at a median 0.512 (0.40-0.67) and says A at 0.599, while B is 0.03. In the toy the same reading stayed at 0.96, which is what §5 relies on. The frozen-arm fire is 0.84 by step 50, and detached F is 0.88 at step 50 while detached B is 0.00. So in the LM, drift alone switches off about half the trigger, and §5's 'moved the trigger little' does not carry over. Live F is also below frozen in 8/8 pairs. (The live arm's 0.86 is now above the frozen-body default, but its smaller displacement confounds that.)
FINAL: minimum 0.76 detached, 0.66 live; 0.98 frozen) and ended at 0.84 and 0.87 (frozen 0.99). Both in-network arms ended below frozen in 8/8 pairs (live 0.82-0.96, detached 0.78-0.93, frozen 0.97-1.00), so the live wiring lost part of the swap, mostly without absorbing it.
- **Trigger drift without rule learning.** The same trigger read off the frozen arm's body, which learned almost none of the swap (B 0.03), fired at a median 0.51 of rule positions at step 600 (0.40-0.67 per pair), against 0.96 for the toy's frozen arm at step 2000; that body still said A at a median 0.60 of rule positions. Training thus switched the reading off at about half the rule positions without the body learning the swap, so §5's toy statement that drift on its own moved the trigger little does not carry over. In the live arm the reading stayed higher (0.86; A still said at 0.88), which fits the gradient holding the trigger on but is confounded with that arm's smaller displacement.

### [30] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: model, trains only its top 4 blocks, and uses one learning rate, horizon and data order.
PROBLEM: The limitations omit the LM caveats that bound the claim 'it also held on a pretrained language model'. (a) The learning rate and horizon came from pilots of the withdrawn g design and were the first setting at which detached separated from frozen; lower rates were never tried with the h module. (b) The regime is aggressive: about 11% of held-out clean tokens change, and rule answers repeat many times. (c) The swap is exact only on the filtered data. (d) B is teacher-forced, with no free-generation check. (e) G0 compared losses only. (f) There are 8 pairs on one checkpoint. Only some of this appears in §3c's scope bullet, and none of it in §6, README or LinkedIn.
SKEPTIC: These caveats check out against review_llm_wsp.md and the data. The learning rate came from addendum-g pilots, where 1e-5 and 3e-5 gave B_det 0.00 and 0.05 and 1e-4 for 600 steps was the first setting to separate; h then reused it. C is 0.883-0.902 in every arm. Each rule answer is drawn 1440/pool times, i.e. about 3.9-60. 26-44% of answers containing A were dropped. B is teacher-forced. G0 compares losses only. §3c's Scope bullet covers part of this ('chosen on the development pairs', C about 0.89, pool sizes), so the §6 omission is minor, but the lr-selection history and the filtering belong in the limitations.
FINAL: model (8 word pairs on one checkpoint), trains only its top 4 blocks and final norm, and uses one learning rate, horizon and data order. The learning rate and horizon come from pilots of the withdrawn addendum-g design, where 1e-4 for 600 steps was the first setting at which the stop-gradient arm separated from the frozen one (on one development pair, 1e-5 and 3e-5 for 300 steps gave B_detached 0.00 and 0.05); lower rates were not tried with the addendum-h module. The training is aggressive: about 11% of held-out clean tokens change in every arm, and each rule answer is drawn about 4-60 times (pools of 24-372). The swap is exact only on the filtered answers (26-44% of answers containing A were dropped); B is teacher-forced on the install-time answers, with no free-generation test; and the step-0 gate compared losses only, not gradients.

### [31] CONFIRMED claims (minor) — README.md
QUOTE: stop-gradient it took the swap over (0.75); 8/8 pairs each. The live wiring mostly behaved like the frozen one (0.05),
PROBLEM: Same overclaim as in the report. 'Took the swap over' at a median 0.75, with B >= 0.9 in 2/8, is partial learning at the no-patch model's level, not a takeover. 'Behaved like the frozen one' holds only for the undo test, not for rule accuracy with the patch in place.
SKEPTIC: Same problem as in the report. Detached is 0.75 against the twin's 0.80 (the twin is in the same sentence), with B >= 0.9 in 2/8. That is partial learning at about the unpatched level, not a toy-style takeover. 'Behaved like the frozen one' holds for the undo test only: live F is below frozen in 8/8 pairs. The numbers are given, so minor. This works together with the next README finding.
FINAL: stop-gradient it learned the swap about as far as the unpatched model did (0.75 vs 0.80; >= 0.9 in only 2/8 pairs); 8/8 pairs each. In the undo test the live wiring mostly behaved like the frozen one (0.05),

### [32] CONFIRMED claims (major) — README.md
QUOTE: and unlike the toy, the handover showed in the patched model's behaviour.
PROBLEM: This misframes the LM rule-accuracy result. The live arm had no handover (B 0.05) but still lost rule accuracy. The real difference from the toy is not that a dip 'showed' (toy dips also showed, especially at 1% exposure) but that in-network patched models ended clearly below frozen and did not recover. The paragraph also omits that one of the two development pairs (' also') was not absorbed in any wiring.
SKEPTIC: Wrong on two counts. The toy handover also showed, as dips (README: 7/24 detached below 0.9). And the live arm had no handover (B 0.05), yet it lost F: median min 0.66, end 0.87 vs frozen 0.99, lower in 8/8. The real difference is that the in-network patched models ended clearly below frozen, while in the toy every model was back at about 0.97 or higher by step 2000 (lowest 0.969). Also confirmed in hpilot: for ' also', detached, live and frozen are all 0.00 while the twin is 0.90.
FINAL: but unlike the frozen one, both in-network wirings lost part of the swap with the patch in place (median rule accuracy fell to 0.76 stop-gradient and 0.66 live and ended at 0.84 and 0.87, against 0.99 frozen; in the toy every model was back at about 0.97 or higher by step 2000). Of two development pairs, one (" also") was not absorbed by any wiring.

### [33] CONFIRMED claims (minor) — FINDINGS.md
QUOTE: - The handover was visible here: patched-model rule accuracy fell to a median minimum of 0.76 (detached) / 0.66 (live).
PROBLEM: This labels the live arm's rule loss as a 'handover', but the live arm did not hand over (B 0.05). It also reports only the dip minima and omits that F ended below frozen in every pair. The key qualifiers of the LM result are missing: partial absorption, the frozen-arm trigger drift, the dev-pair failure, and a learning rate selected in pilots.
SKEPTIC: This calls the live arm's F loss a handover, but the live arm mostly did not hand over (B 0.05). The line also leaves out the end F (0.84 / 0.87 vs 0.99, below frozen in 8/8). Recomputed numbers: detached B >= 0.9 in 2/8, frozen-arm fire 0.51 vs the toy's 0.96, ' also' unabsorbed in every arm, C 0.88-0.90. FINDINGS is an internal log, so minor.
FINAL: - Not silent here: with the module in place, rule accuracy fell to a median minimum of 0.76 (detached, during the handover) / 0.66 (live, which mostly did not hand over) and ended at 0.84 / 0.87 vs 0.99 frozen (both below frozen in 8/8). Detached B >= 0.9 in only 2/8; the trigger read off the frozen arm's body fell to 0.51 (toy 0.96); dev pair " also" was not absorbed in any arm. One learning rate (from pilots), one data order; about 11% of clean tokens changed in every arm.

### [34] CONFIRMED claims (major) — linkedin_post.md
QUOTE: On the language model the model mostly kept the patch firing instead of learning the rule.
PROBLEM: This states a mechanism ('kept the patch firing instead of learning') as fact and omits the cost. With the patch in place, the live-wired language model lost part of the rule in every pair. The review warned not to attribute the live result to the trigger defending itself, because the live arm also trains about 10% less. The earlier audit (docs/audit_wsp.md [255], [403]) already removed 'keeps the patch firing' as a presented mechanism.
SKEPTIC: The descriptive part is supported: live fire 0.86, B 0.05, 6/8 pairs below 0.2. But the post leaves out that the live-wired LM lost rule accuracy with the patch in place (end F 0.87 vs 0.99 frozen, lower in 8/8; median min 0.66). It also leaves out the review's confound: the live arm moved its weights about 9% less (displacement 12.5 vs 13.6-13.8), so 'instead of learning' should not be read as the trigger defending itself. One correction to the evidence: the earlier audit did not remove 'kept firing'. Its [59] FINAL used 'mostly kept the trigger firing instead of learning the rule' for the toy. The omission of the cost is still real.
FINAL: On the language model the model mostly didn't learn the rule (6 of 8 word pairs stayed below 20%), and unlike with the frozen patch, rule accuracy with the patch in place fell (87% at the end, vs 99%). The patch kept firing, but this wiring also moved the weights about 10% less, so we can't yet say why.

### [35] CONFIRMED claims (major) — linkedin_post.md
QUOTE: • On the language model the handover was visible: rule accuracy with the patch in place dipped to ~70–80% along the way.
PROBLEM: 'Dipped along the way' implies recovery, but rule accuracy ended at 84% / 87%, not back near 100%. '~70–80%' excludes the live arm's median minimum of 66%. The live arm had no handover. The caveat also leaves out the aggressive, pilot-selected training regime.
SKEPTIC: '~70-80%' leaves out the live median minimum of 66% (detached 76%). 'Dipped along the way' implies recovery, but F ended at 84% / 87% vs 99% frozen. The live arm mostly did not hand over. The regime (C about 0.89 in every arm) and the pilot-selected learning rate are also missing. A reader would come away thinking the dip was transient.
FINAL: • On the language model the in-network patches did not stay silent: with the patch in place, rule accuracy fell to a median 76% (no-gradient wiring, during the handover) and 66% (gradient wiring, which mostly did not hand over) and ended at 84% and 87%, against 99% with the frozen patch. Training there was aggressive (it changed ~11% of ordinary answer tokens) and used one learning rate, picked in pilots.

### [36] CONFIRMED claims (minor) — linkedin_post.md
QUOTE: • The patch was an add-on held fixed, not an edit to existing weights, and each test used one training recipe.
PROBLEM: This brings back an error the earlier claims audit corrected. 'One training recipe' contradicts the toy design, which used three optimiser settings (docs/audit_wsp.md [64]). The new caveats also drop the adopted caveat that the trigger reads the very score it changes, with no unrelated-symbol control. 'Held fixed' sits uneasily with the post's own trainable-patch result.
SKEPTIC: This regresses adopted corrections. Audit [64] fixed 'one training recipe' (the toy used Adam 3x, Adam 1x and SGD), and the 583b3da/02d4654 caveat 'one rule, whose trigger reads the very score it changes' has been dropped. 'Held fixed' also sits beside the post's own trainable-patch result. Minor, because 'one recipe' errs on the conservative side; the lost caveat is a scope omission.
FINAL: • The patch was an add-on (held fixed in the main tests), not an edit to existing weights, and its trigger reads the very score it changes (no control with an unrelated word or symbol). The small-transformer contrast held under three optimiser settings; the language model used one learning rate (picked in pilots), one data order and 8 word pairs.

### [37] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: or modules that are themselves trained.
PROBLEM: The limitations say no run tested modules that are themselves trained. §3b (F2) did exactly that: the 134 numbers trained in all three arms. The limitation was not updated after addendum f.
SKEPTIC: §3b F2 ran exactly this: the 134 numbers trained, giving 24/24, 20/24 and 0/24 (wsp_refine_prereg.txt). The limitation was not updated after addendum f.
FINAL: or (apart from the toy condition in §3b where the 134 numbers also trained) modules that are themselves trained.

### [38] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: - One kind of edit: two new ReLU units added after the final LayerNorm (an add-on, not a change to existing weights),
PROBLEM: The edit description no longer covers the LM module. That module reads the margin of A over the best other token, which needs a max over the whole vocabulary, so it is a hand-written logit processor rather than a small set of added weights. Calling it 'three hand-set numbers' in a 'weight patch' study needs this qualification. γ is also calibrated from the model's logits, not hand-set.
SKEPTIC: In swap_lm.py, margin() takes logit_A minus the max over the other 49,152 logits (config vocab_size 49152), and the module runs as a LogitsProcessor. So the LM edit is a hand-written logit processor rather than a few added weights, and the bullet does not cover it. I dropped the auditor's γ clause: in the toy γ is also calibrated on data, so it does not distinguish the two.
FINAL: - One kind of edit: two new ReLU units added after the final LayerNorm (an add-on, not a change to existing weights; in the language model the units read the margin of A over the best other token, a max over all 49,152 logits, so that module is a hand-written logit processor rather than a few added weights),

### [39] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: - The mechanism statements in §5 are interpretation; only the W tests in §3 were registered.
PROBLEM: This is outdated and incomplete. F1/F2 and H1-H4 were also registered. §5's toy-based account of the first misses ('on its own it moved the trigger little') is contradicted in magnitude by the LM data, yet §5 was not updated and the report now claims the contrast 'also held' on the LM.
SKEPTIC: Outdated: F1/F2 (addendum f) and H1-H4 (addendum h) were also registered. The LM frozen-arm fire (0.51 vs the toy's 0.96) contradicts §5's toy statement that drift moved the trigger little. The auditor's 'does not describe the LM runs' goes too far, though. LM drift seeding the misses is consistent with §5's suspicion; only the 'moved little' magnitude fails to carry over.
FINAL: - The mechanism statements in §5 are interpretation drawn from the toy; only the W, F and H tests were registered. In the language model the trigger reading drifted off without rule learning far more than in the toy (frozen-arm body 0.51 vs 0.96 of rule positions), so §5's statement that drift on its own moved the trigger little holds only for the toy.

### [40] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: displacement 12.5 vs 13.6-13.8), a side effect of Adam noted in the review, so part of H3 may reflect less training.
PROBLEM: The report states the cause ('a side effect of Adam') as fact. The review's skeptic said the Adam second-moment mechanism was not shown, only that the live arm trains less. The available data also do not support displacement as the driver of live B across pairs.
SKEPTIC: The review's skeptic wrote: 'What is not shown is the specific Adam second-moment mechanism; that the live arm trains less is shown.' The report states the cause as fact. The recomputed Spearman(live B, live disp) is -0.10 across 8 pairs, but that across-pair figure says little about the between-arm confound, so I state it neutrally rather than as counter-evidence.
FINAL: displacement 12.5 vs 13.6-13.8). The review attributed this to Adam (large gradients through the trigger inflating its second-moment estimates), but that mechanism was not tested; part of H3 may reflect less training (across the 8 pairs, live B did not track live displacement: Spearman −0.10).

### [41] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: The first design (addendum g) used a linear trigger that did not implement the swap and was withdrawn before analysis
PROBLEM: The withdrawal is described accurately but incompletely. 'Linear trigger' is imprecise: both designs use the same ReLU ramp, and the difference is a raw-logit readout vs a margin readout. The text omits four facts: g was withdrawn while its sealed runs were under way (3 of 32 finished; a reviewer read displacements of two); h carried over g's learning rate, horizon and batch settings; one sealed pair changed between g and h after the prompt set grew from 4,000 to 12,000; and h was registered after its own dev-pair pilot.
SKEPTIC: The text is accurate but incomplete. PREREGISTRATION confirms the following. g was stopped after 3 of 32 sealed runs (a reviewer read two displacements). h says 'Settings are the ones piloted for addendum g'. g's sealed list had many/numerous and h's has good/great (prompts went from 4,000 to 12,000; many and good tied at 53 under g). h's text includes its own dev-pair pilot. The review shows 1-9 of 64 eval items per pair began at a real A. 'Linear trigger' is loose: both designs use the ReLU ramp, and the difference is a raw-logit vs a margin readout. I reordered the auditor's replacement so the sentence still ends before the existing '(docs/review_llm_wsp.md).'
FINAL: Addendum h kept the learning rate, horizon and batch settings piloted for the first design (addendum g), re-ranked the candidate words on 12,000 instead of 4,000 prompts (one sealed pair changed: many/numerous became good/great) and was registered after its own development-pair pilot. Addendum g read the raw logit of A, which did not mark where the model would say A (only 1-9 of 64 eval items per pair began at a real A position); it was withdrawn while its sealed runs were under way (3 of 32 finished) and before analysis

### [42] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: swap the unpatched model learned from the same data in about 75 steps. There the live wiring mostly behaved like the
PROBLEM: 'In about 75 steps' counts from step 0, including 20 steps at learning rate 0. The toy text counts from the learning-rate switch-on (an earlier audit correction, docs/audit_wsp.md [66]). It also reads as full learning, but the median at step 75 is 0.68, and the model reached 0.62-0.93 only by step 600. The same convention issue affects §3c's 'a median 75 steps' and 'about 190 steps in'.
SKEPTIC: The twin's T values are 50-100 (median 75), counted from step 0. The learning rate is 0 until step 20 (warm = 20), so that is a median 55 steps after switch-on. The toy text counts from switch-on (audit [66]). The median twin B is 0.40 at step 50, 0.68 at 75 and 0.80 at 600, so 'learned in 75 steps' also overstates how complete the learning was. The replacement joins up cleanly with finding 1's text.
FINAL: swap the unpatched model learned from the same data (B >= 0.5 a median 55 steps after the learning rate switched on; 0.80 at step 600). There the live wiring mostly behaved like the

### [43] REJECTED claims (minor) — REPORT_WSP.md
QUOTE: which live models are absorbed is largely a matter of chance, while the share
PROBLEM: 'Largely a matter of chance' overstates. Across the four streams, absorption is overdispersed relative to independent draws: there is a model-specific component, and between-stream correlations are positive. FINDINGS.md repeats this ('which model is not').
SKEPTIC: The auditor's numbers reproduce: counts 1/6/6/5/6 vs a binomial 0.7/3.8/8.4/8.2/3.0, variance 1.48 vs 0.96, permutation p of about 0.03, pairwise Spearman 0.14-0.49. But they support 'largely'. The implied between-model variance gives an ICC of about 0.18, so roughly 80% of the outcome variance is stream (chance) variance. The auditor's own replacement says the outcome 'varies mostly with the data order', which is what 'largely a matter of chance' says. The hedge already allows for a model component, so the claim is not overstated.
FINAL: which live models are absorbed is largely a matter of chance, while the share

### [44] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: detached > frozen in 23/24 models (p = 1.3e-5).
PROBLEM: §3b reports no rule-accuracy-with-module (F) result for the refinements. At low exposure the takeover was much less quiet than the summary's 'brief dip in some models' suggests, and those dips now fall within the summary's scope ('as few as 1-3%').
SKEPTIC: Recomputed from wsp_refine_summary. At q = 0.01, F < 0.9 at some evaluation in 13/24 detached models (lowest 0.576) and 19/24 live (lowest 0.453); 6 live models had low evaluations spanning more than 500 steps. At q = 0.03 it is 13/24 detached. At q = 0.3 it is 7/24 and 2/24. At 2000 steps the lowest is 0.969. The summary's 'brief dip in some models' now covers 1-3% exposure, where it understates. Labelled exploratory.
FINAL: detached > frozen in 23/24 models (p = 1.3e-5). The takeover was less quiet at low exposure (exploratory): with the module in place, F fell below 0.9 at some evaluation in 13/24 detached and 19/24 live models at q = 0.01 (lowest 0.58 and 0.45; in 6 live models such dips recurred over more than 500 steps) and in 13/24 detached models at q = 0.03, against 7/24 and 2/24 at q = 0.3; by step 2000 all were back at 0.969 or above.

### [45] CONFIRMED claims (minor) — README.md
QUOTE: (21/24 vs 0/24) of the training examples showed the rule (pre-registered refinements, addendum f). With the patch in place, every model ended at >= 0.97 rule accuracy; the takeover showed only as a
PROBLEM: Now that 1% exposure is in the same sentence, 'showed only as a brief dip in some models' understates the dips at low exposure. The 1% condition was registered as exploratory, not as a test.
SKEPTIC: Addendum f says 'q = 0.01 is exploratory'. At 1% exposure, F dipped below 0.9 in 13/24 detached and 19/24 live models (lowest 0.58 and 0.45), so 'only a brief dip' understates. End F at 2000 steps is at least 0.969, i.e. about 0.97.
FINAL: (21/24 vs 0/24) of the training examples showed the rule (pre-registered refinements, addendum f; 1% was exploratory). With the patch in place, every model ended at about 0.97 or higher at 2000 steps (but at 1% exposure rule accuracy dipped below 0.9 in 13/24 stop-gradient and 19/24 live models, lowest 0.58 and 0.45); at 30% exposure the takeover showed only as a

### [46] CONFIRMED claims (minor) — README.md
QUOTE: settings; the language-model replication trains only the top 4 of 30 blocks of one small model.
PROBLEM: The README scope omits the LM caveats that limit 'held on a pretrained language model': a single pilot-selected learning rate and data order, 8 pairs, an aggressive regime, and a swap that is exact only on filtered data.
SKEPTIC: The README scope leaves out LM caveats that bound 'held on a pretrained LM': one learning rate taken from pilots of the withdrawn g design (where 1e-5 and 3e-5 gave B_det 0.00 and 0.05 within 300 steps), one data order, 8 pairs, C 0.88-0.90 in every arm, and filtering that dropped 26-44% of answers containing A.
FINAL: settings; the language-model replication trains only the top 4 of 30 blocks of one small model, on 8 word pairs with one learning rate (taken from pilots of a withdrawn earlier design, where lower rates showed no absorption within 300 steps) and one data order, in a regime that changed about 11% of held-out clean tokens in every arm; its swap is exact only on the filtered answers.

### [47] CONFIRMED claims (minor) — FINDINGS.md
QUOTE: - SmolLM2-135M-Instruct, exact word swaps, 8 sealed pairs: no patch 0.80, frozen 0.03, detached 0.75, live 0.05
PROBLEM: 'Exact word swaps' generalises beyond the filtered data, as in the report summary.
SKEPTIC: Same over-generalisation as in the summary. swap_pairs.json shows that 26% (help, 154/590) to 44% (often, 182/414) of the base answers containing A were dropped by the ramp and unswapped-A filters.
FINAL: - SmolLM2-135M-Instruct, word swaps (exact only on the filtered answers, which dropped 26-44% of answers containing A), 8 sealed pairs: no patch 0.80, frozen 0.03, detached 0.75, live 0.05

### [48] CONFIRMED claims (minor) — linkedin_post.md
QUOTE: so the model gets almost no error signal and doesn't learn the rule
PROBLEM: This brings back the overclaim the earlier audit corrected ('never learns' changed to 'barely learns'). Frozen-wired models learned some of the rule: 10/24 toy models partly, one to 0.93, and up to 0.13 on the LM.
SKEPTIC: This regresses audit [65], which changed 'never learns' to 'barely learns'. Frozen B in the toy is median 0.03 at L1, with 10/24 partly learned and one model at 0.93. In the LM it is 0.00-0.13. The same paragraph later gives 3%, so minor.
FINAL: so the model gets almost no error signal and barely learns the rule

### [49] CONFIRMED claims (minor) — linkedin_post.md
QUOTE: the same training without any patch taught the swap within ~75 steps (80% of cases)
PROBLEM: This merges two numbers. 75 steps (counted from step 0, including 20 steps at learning rate 0) is when the twin crossed 50%, while 80% is the step-600 value. A reader will take it as 80% within 75 steps.
SKEPTIC: This merges two numbers. 75 is the median T (B >= 0.5) counted from step 0, which includes 20 lr-0 steps. 80% is the median B at step 600; at step 75 the median twin B is 0.68.
FINAL: the same training without any patch taught the swap to half of the cases within about 55 training steps (median over word pairs) and to 80% by the end (600 steps)

### [50] CONFIRMED claims (minor) — linkedin_post.md
QUOTE: Training shifts what the patch reads, so it starts missing cases, and those misses teach the model the rule.
PROBLEM: The post states as fact a causal chain that REPORT §5 calls untested ('we suspect this drift seeds the first misses, but it is untested'). In the LM the drift itself is visible, but that the misses cause the learning is not tested.
SKEPTIC: REPORT §5 says the drift seeding the first misses is 'untested' in the toy, and the post states it as fact. In the LM the drift is directly visible: frozen-body fire is 0.84 at step 50, and detached F is 0.88 at step 50 with B 0.00. The post should match the report's hedge.
FINAL: We think training shifts what the patch reads, so it starts missing cases, and those misses teach the model the rule (on the language model the drift is directly visible; in the small transformers it is untested).

### [51] CONFIRMED claims (minor) — linkedin_post.md
QUOTE: At the moment of the edit, both give identical outputs and identical gradients on the training data (by design).
PROBLEM: The setup presents a language-model example, so this reads as covering the LM. There, only in-batch losses were compared (agreeing to 1.9e-7, not bitwise), and gradients were never compared. Bitwise output and gradient identity was shown only in the toy (W0).
SKEPTIC: The setup uses the LM example, but for the LM only in-batch losses were compared (G0 max 1.91e-7, in patch_fate_lm.py), and gradients were never compared (no gradient check in llm_wsp). Bitwise identity of outputs and gradients was shown only in the toy (W0).
FINAL: At the moment of the edit, both give the same outputs and gradients on the training data (by design; bit-identical in the small transformers; on the language model we checked only that losses matched, to within 2e-7).

### [52] CONFIRMED claims (minor) — linkedin_post.md
QUOTE: including one we withdrew after finding a bug
PROBLEM: 'A bug' understates the withdrawal. The registered design's patch did not implement the registered rule: a raw-logit readout that cannot locate the word, plus a silent fallback. The withdrawal also happened while its sealed runs were under way.
SKEPTIC: 'A bug' understates the withdrawal. The registered module did not implement the registered rule (raw-logit readout plus a silent fallback), and the withdrawal came mid-run (after 3 of 32 sealed runs). The auditor's 'before looking at any of its results' overstates, because a reviewer read two displacement values. 'Before analysing its outcomes' matches PREREGISTRATION.
FINAL: including one we withdrew mid-run, before analysing its outcomes, after a review found its patch did not implement the registered rule

### [53] CONFIRMED claims (minor) — REPORT_WSP.md
QUOTE: - **Development pairs** (pilot): " use" behaved like the sealed pairs (detached 0.51); " also" was not absorbed within
PROBLEM: ' use' did not behave like the sealed pairs. Its detached arm fell far short of the no-module model, a gap larger than in any sealed pair. With ' also' at 0.00, 1 of the 2 development pairs shows that the stop-gradient absorption on the LM is not reliable across words.
SKEPTIC: From wsp_lm_hpilot.jsonl.gz: ' use' detached is 0.507 vs none 0.836, a shortfall of 0.33. The largest sealed shortfall is 0.13 (small: 0.671 vs 0.800), and 0.51 is below the sealed detached range of 0.62-0.92. Qualitatively the ordering matches, but the size of the absorption does not.
FINAL: - **Development pairs** (pilot): " use" was absorbed only partly (detached 0.51 against 0.84 without a module, a larger shortfall than in any sealed pair, where it was at most 0.13); " also" was not absorbed within

### [54] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: It's a crutch: on a real language model (SmolLM2-135M), the same training without any patch taught the swap within ~75 steps (80% of cases). With the frozen patch in place, the model learned almost nothing (3%).
PROBLEM: Two different numbers are merged into one claim. '~75 steps' is the median number of steps until the no-patch model passed 50% on the swap. '80%' is its median at the end of the 600-step run. Read together, the sentence says the model hit 80% within 75 steps, which is wrong. The count also includes the 20 opening steps at learning rate 0, so the model is credited with being faster than it was. The model name is also wrong: the runs used the Instruct checkpoint.
SKEPTIC: Confirmed. Twin T (first eval with B >= 0.5) is 100/50/75/50/75/100/50/75, median 75. End B is 0.62-0.93, median 0.80. Recomputed from wsp_lm_h.jsonl.gz: median twin B is 0.68 at step 75 (0.19-0.80) and 0.70 at step 100, so the sentence's implied '80% within 75 steps' overstates the speed. In swap_lm.py, cur = 0 for st < warm = 20, so the first 20 steps are at lr 0. The runs used SmolLM2-135M-Instruct (addendum h). Frozen B median 0.03 is correct. The core crutch claim (0.80 vs 0.03, 8/8) stands, so this is an overstated detail plus a wrong model name: minor.
FINAL: It's a crutch. On a real language model (SmolLM2-135M-Instruct), the same training without any patch learned the swap quickly: it passed 50% of rule positions after a median of ~75 steps (the first 20 at learning rate 0) and reached 80% by the end of the 600-step run. With the frozen patch in place, the model learned almost nothing (3%).

### [55] CONFIRMED linkedin (major) — linkedin_post.md
QUOTE: On the language model the handover was visible: rule accuracy with the patch in place dipped to ~70–80% along the way.
PROBLEM: 'Dipped ... along the way' implies the drop was temporary. It was not: the patched language model ended training clearly worse at the rule than it started, in every in-network run. The '~70–80%' range also leaves out the gradient-through wiring, whose median low was 66%. And 'handover' is the wrong word for that wiring: in 6 of 8 pairs no handover happened (B <= 0.17), yet its accuracy still fell. The real finding is that the edit itself degraded, and that is buried.
SKEPTIC: Confirmed with one correction. F = 1.00 at step 0 in all 24 patched runs. Detached: minimum F 0.60-0.80 (median 0.76), end 0.78-0.93 (median 0.84). Live: minimum 0.59-0.79 (median 0.66), end 0.82-0.96 (median 0.87). Frozen: median minimum 0.98, end 0.97-1.00. 'Dipped along the way' implies a temporary drop, but no in-network run got back to 1.00. In the live arm 6/8 pairs had no handover (B <= 0.17), so 'handover' is the wrong word there. The 70-80% range leaves out the live median of 66%. The edit degraded persistently, which contrasts with the toy (end F >= 0.979 in every q = 0.3 run; median never below 0.96). Correction: the auditor's 'It never recovered' is too strong. F did partly recover in most runs (median 0.76 -> 0.84 detached, 0.66 -> 0.87 live). It just never returned to 100%. Also, toy dips were not 'rare' (31/96 detached runs went below 0.9 at some eval).
FINAL: On the language model, the patched model itself got visibly worse at the rule during training. With the patch in place, rule accuracy fell from 100% to a median low of 76% (stop-gradient wiring) and 66% (gradient-through wiring). It ended at 84% and 87%, and no run got back to 100%. With the frozen wiring it stayed at 97–100%. In the small transformers the dips were brief: the median never fell below 96%, and every model ended at 97% or higher.

### [56] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: A third wiring, with gradients flowing through the patch, depended on the setting. In the small transformers it landed in between, and which models absorbed the rule changed with the data order. On the language model the model mostly kept the patch firing instead of learning the rule.
PROBLEM: The post centres two wirings that cannot be built by editing weights. 'Frozen' runs a frozen copy of the whole original network alongside the model. 'Stop-gradient' needs a change to the training code. The only wiring you get by actually writing the units into the network and training normally is this 'third wiring'. The post puts it last and softens it to 'depended on the setting'. On the language model it was as bad as the frozen crutch: the pre-registered test that it beats frozen failed. Removing the patch brought the original word back at most positions, and the patched model's accuracy also fell. The post also omits a confound the report flags: this arm moved its weights about 10% less.
SKEPTIC: The facts in the paragraph are accurate. The toy result was in between (W2/W3), the data stream moved 8/24 models, and on the LM the live trigger still fired at a median 0.86. The auditor's framing points do check out. REPORT §6 says the frozen arm keeps a frozen copy of the whole network and the detached arm needs a stop-gradient in the training code, so the live wiring is the one closest to a plain edit. On the LM it was not distinguishable from frozen (H4 0.05 vs 0.03, p = 0.078, not supported). Recomputed live A_left at step 600 has median 0.89. Live displacement was 12.5 vs 13.6-13.8 (about 9-10% less). In the toy, live was 16/24 at the main recipe, 7/24 at 1x lr and 6/12 under SGD. The post's 'mostly kept the patch firing instead of learning the rule' already gets the LM outcome across, so this is softened emphasis and an omission rather than a false claim: minor. The replacement is tightened. It avoids saying the live wiring is literally 'training as usual', because the patch was held fixed.
FINAL: A third wiring is the closest to simply writing the patch into the network and training as usual: it reads the live model, and gradients flow through it. It was erratic. In the small transformers it landed in between: 16/24 models absorbed the rule with our main recipe and 7/24 at a 3x lower learning rate, and which models absorbed it changed with the data order. On the language model it behaved much like the frozen crutch. In 6 of 8 word pairs the model learned almost none of the swap (median 5% vs 3% frozen, not a significant difference), the patch kept firing, and removing it brought the original word back at most rule positions. This arm also moved its weights about 10% less, so part of the gap may reflect less training. Note that the first two wirings are not plain weight edits: one runs a frozen copy of the original model alongside, and the other needs a stop-gradient in the training code.

### [57] CONFIRMED linkedin (major) — linkedin_post.md
QUOTE: The patch was an add-on held fixed, not an edit to existing weights, and each test used one training recipe.
PROBLEM: This caveat is wrong for the toy and incomplete for the language model. Toy: three optimiser settings were tested (W6), and the gradient-through result swung strongly between them. The patch was also not always held fixed, since the post itself cites a test in which it trained. Language model: the post never says the recipe was aggressive or how it was chosen. Training changed about 11% of the model's ordinary output tokens in every arm, including no-patch and frozen. Even with no rule learning, the drift switched off about half the trigger's readings. The learning rate and horizon were picked on two development pairs, and gentler training was never tested with this design. A practitioner doing light maintenance fine-tuning cannot assume these language-model numbers apply.
SKEPTIC: Confirmed. Toy: W6 tested SGD-momentum and Adam at 1x lr, and the live arm swung with the recipe (16/24, 7/24, 6/12). The post itself cites the trainable-module test (F2), which contradicts 'held fixed'. LM: one lr and horizon, chosen on dev pairs as the first setting that separated the arms (docs/review_llm_wsp.md, confirmed). Recomputed C at step 600 is 0.883-0.902 in all 32 runs, so about 11% of clean tokens changed in every arm, twin and frozen included. The frozen arm's trigger read off the trained model fell from 1.00 to a median 0.51 (0.40-0.67) with B ~0. The review found that in the LM the detached trigger switches off partly through this generic drift rather than through rule learning. The LM absorption result is therefore tied to an aggressive, selected regime. A post caveat that hides this materially overstates how far the LM result extends.
FINAL: • The patch was an add-on module, held fixed except in one toy test where it also trained. It was not an edit to existing weights.
• The language-model runs used one aggressive recipe: one learning rate and a 600-step horizon chosen on two development word pairs, one data order, and training that changed about 11% of the model's ordinary output tokens in every arm. Gentler training was not tested with this design. The toy results cover three closely related optimiser settings.

### [58] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: If further training includes even a little data showing an edit's behaviour, removing the edit may no longer remove the behaviour. When the data never showed it, removal worked every time.
PROBLEM: This takeaway is stated with no setting attached, but it rests only on the toy experiments. Exposure levels of 1-30% and 0% were never run on the language model, which was trained only at 30%. It also leaves out what happened in the 0% case. That was a pre-registered, supported result (W5): when training never showed the rule, the in-network patches eroded, in some models badly. Readers will conclude that an edit not exercised in training stays intact and removable. That is half true: removal still worked, but the edit was not intact.
SKEPTIC: Partly confirmed. The 'small models' header already frames the takeaways, and the post states earlier that 1-3% is a small-transformer result, so the setting-attachment complaint is weak. The omission of erosion is real, though. Recomputed from wsp_sealed_summary at q = 0 and 2000 steps: B <= 0.008 everywhere, so removal did always work. But median F was 0.85/0.94 detached and 0.90/0.93 live (levels 1/3), with minimums of 0.22/0.11 detached and 0.48/0.65 live. This is pre-registered W5 (supported), and REPORT §5 implication (4) was dropped from the post. Low exposure (q < 0.3) was never run on the LM. This is a half-truth by omission, not a false statement: minor.
FINAL: 1️⃣ In the small transformers, if further training included even a little data showing the edit's behaviour (1–3% of examples), removing the edit often no longer removed the behaviour, unless the patch read a frozen copy. When the data never showed it, removal undid the behaviour every time, but the edit itself eroded: in-network patches fell to a median 85–94% rule accuracy, and some fell much lower. (The language model was tested only with about 30% of training answers showing the rule.)

### [59] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: Language model: 75% of cases, close to the no-patch model, in 8/8 word pairs.
PROBLEM: The sentence reads as 75% in each of the 8 pairs. In fact 75% is the median, pairs ranged from 62% to 92%, and '8/8' refers to the pre-registered test against the frozen wiring. The surrounding claim 'removing the patch changes little' leaves out why: by the end the patch had almost stopped firing, so there was little left to remove.
SKEPTIC: Confirmed. Detached B per pair is 0.72, 0.62, 0.90, 0.75, 0.79, 0.76, 0.92, 0.67 (median 0.75). The twin median is 0.80. '8/8' is the H2 comparison with frozen. The sentence can be read as 75% in every pair. Recomputed detached fire at step 600 is 0.014-0.193 (median 0.065), so the patch had largely gone quiet. This is a clarity issue only.
FINAL: • Language model: with the patch removed, the model still made the swap at a median 75% of rule positions (62–92% across word pairs, vs 80% for the no-patch model), above the frozen wiring in 8/8 pairs. By then the patch itself fired at only a median 6% of rule positions.

### [60] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: The setup: we gave a model a hand-built patch for one rule, e.g. "wherever you'd say 'provide', say 'offer'".
PROBLEM: The only rule described is the language-model word swap, but most of the numbers that follow (24/24, 1–3%, 'in between') come from toy transformers. Those used a symbol remap on copy, reverse, sort and shift tasks, which the post never mentions. 'Patch' also hides, until the last caveat, that this is a bolted-on module and not a change to existing weights. That fact matters for the hook's 'hand-written edit' framing.
SKEPTIC: Confirmed. The only rule described is the LM word swap, but the 24/24, 1-3% and 'in between' numbers come from toy transformers whose rule is a symbol remap (REPORT §2). The add-on nature is disclosed only in the last caveat. This is a clarity and framing issue.
FINAL: The setup: we bolted a small hand-built module onto a model to implement one rule. In 24 toy transformers (trained on copy, reverse, sort and shift) the rule was "emit symbol z wherever you would emit a". On a small language model it was a word swap such as "wherever you'd say 'provide', say 'offer'".

### [61] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: In the small transformers this held even when only 1–3% of the further-training examples showed the rule, and when the patch's own numbers were also trained.
PROBLEM: Coming right after '24/24 kept the rule', this suggests the result held in every model at 1%. At 1% it held in 21 of 24 models.
SKEPTIC: Confirmed. wsp_refine_prereg.txt gives detached 21/24 with B >= 0.5 at q = 0.01 (median 0.92) and 24/24 at q = 0.03. Detached > frozen held in 23/24 at q = 0.01. Trainable module: detached 24/24. Coming right after '24/24 kept the rule', the sentence overstates the 1% case slightly.
FINAL: In the small transformers this held even when only 3% of the further-training examples showed the rule (24/24 models; 21/24 at 1%), and when the patch's own numbers were also trained (24/24).

### [62] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: At the moment of the edit, both give identical outputs and identical gradients on the training data (by design).
PROBLEM: The identity holds only because the training data was filtered so the trigger is always fully on or fully off. The post does not say this. It was also checked exactly only on the toy. On the language model, only training losses were compared (through the first update), not gradients.
SKEPTIC: The statement is true and hedged with 'by design', but it omits what the design required. The identity holds only because the training pools were filtered so every trigger is fully on or fully off (REPORT §6). That matters for takeaway 3. The toy W0 checked logits and gradients exactly. For the LM, analysis/patch_fate_lm.py G0 compares in-batch losses over loss[:warm+2] (through index 21, i.e. after the first update; max difference 1.9e-7), with no parameter-gradient comparison. Gradient identity on the LM holds analytically at saturation but was not measured. This is a minor transparency gap.
FINAL: At the moment of the edit, both give the same outputs and gradients on the training data. That is by design: we filtered the training data so the patch is always fully on or fully off. We checked this exactly on the small transformers, and on the language model through matching training losses up to the first update.

### [63] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: Checks at the moment of the edit couldn't tell the versions apart.
PROBLEM: This is presented as a finding, but it holds by construction: the wirings were built and the data filtered so that these checks would match. The report says '(here by construction)'. The takeaway drops that qualifier.
SKEPTIC: Confirmed. REPORT §5 (3) qualifies this as '(here by construction)', because the wirings and filtered data were built so the checks match. The takeaway drops that qualifier and reads like an empirical finding. The post does say 'by design' earlier, so this is minor.
FINAL: Output and gradient checks on the training data at the moment of the edit could not tell the wirings apart (we built them to match), so inspect the wiring itself.

### [64] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: plus SmolLM2-135M with only its top 4 blocks trained.
PROBLEM: The checkpoint is named wrongly: the runs used SmolLM2-135M-Instruct. The caveat also does not say that 4 blocks is 4 of 30, which ML readers need in order to judge how far the language-model result extends.
SKEPTIC: Confirmed. Addendum h and REPORT §3c give SmolLM2-135M-Instruct, with the top 4 of 30 blocks and the final norm trained, on 8 sealed word pairs. The checkpoint name is wrong and the '4 of 30' context is missing.
FINAL: plus SmolLM2-135M-Instruct with only the top 4 of its 30 blocks (and the final norm) trained, on 8 word swaps.

### [65] CONFIRMED linkedin (minor) — linkedin_post.md
QUOTE: Weight-space programming, meaning writing behaviour into a network's weights by hand instead of training it in, is getting a lot of attention.
PROBLEM: Nothing in the repo supports 'getting a lot of attention'. The only source cited is a single preprint from three days before the post, and it uses the term for frontier language models editing existing parameters of small networks. The hook leads readers to expect results about weight edits. The experiments test add-on modules, and the two contrasted wirings are not weight edits at all.
SKEPTIC: Nothing in the repo supports 'getting a lot of attention'. The only source is arXiv 2611.05341 (3 Nov 2026, three days before the post), where the term names a frontier-LM ability to read and write raw weights of small networks, not hand edits (REPORT §1). The hook sets up an expectation of weight-edit results, while the experiments use an add-on module and two wirings that are not plain weight edits (REPORT §6). This is puffery plus framing, so minor.
FINAL: Weight-space programming, writing behaviour directly into a network's weights instead of training it in, was just reported as an ability of frontier language models (arXiv 2611.05341). But edited models usually keep getting trained. What happens to a hand-written edit then? We tested a simple stand-in: a small hand-built module added to the network.
