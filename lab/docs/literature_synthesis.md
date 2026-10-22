# Synthesis: interpretability-based detection of hidden misalignment, as of 2026-10-22

**Sources.** The audits that fed this synthesis could not reach arXiv, OpenReview, LessWrong or the Alignment Forum, and the web-search budget was used up. Figures marked **(u)** come from abstracts, memory or secondary summaries and were not checked against the paper. Unmarked figures were checked against anthropic.com, the authors' GitHub READMEs and code, or PMLR v306. Re-run the novelty searches before any public claim, especially for work published June–October 2026.

**In-house evidence used below.** Lab repo: `/home/user/aegis/lab`.
- **Training cost:** in the dev zoo `runs/zoo1`, 63 of 208 models are trained so far. The median is about 164 s per model at d ≤ 96, L ≤ 3, about 3k AdamW steps. With 4 workers that is roughly 85–90 models per hour.
- **Prior pilot (H1, tug-of-war law),** 13 backdoored vs 15 benign models, using the *true* hidden distribution as an oracle:
  - cos(D_clean, D_hidden) has median −0.232 (IQR −0.266 to −0.120), against −0.019 for benign models scored against fake hidden objectives;
  - all 13 backdoored models are negative (sign test p = 1.2e-4);
  - AUROC 0.885, 95% CI [0.76, 1.00];
  - D_clean is reproducible across independent clean samples (median cos 0.85);
  - per-model AUROC for identifying poisoned examples has median 0.886;
  - subset triggers are weak (−0.03, n = 3).
  
  This is promising but small. It tests the law, not the trigger-free detector (H2).

---

## 1. State-of-the-art map

The useful way to read the field is by **access tier**: what the auditor needs in hand. Almost every strong result sits in a tier that assumes something a real auditor lacks.

| Tier / setting | Best methods | Evidence | Where it fails |
|---|---|---|---|
| **Input-level, triggered inputs available** | Linear "defection" probes (MacDiarmid et al. 2024); Apollo deception probes (Goldowsky-Dill et al., ICML 2025); activation-OOD MAD (Mallen et al., COLM 2024); functional-attribution MAD using SGLD loss-trace correlation (Keenan, Leckie, Erfani, ICML 2026) | AUROC 99.3% from one generic contrast pair, but the best of 1,000 *random* directions already reaches 95.3%. Apollo: 0.96–0.999 (u). Quirky MAD: about 0.95 (u). Keenan: BackdoorBench DER 0.93 vs 0.83, and holds up on activation-obfuscated LLM backdoors (u; "DER" may be a removal metric). | Signal may be an insertion artefact (authors' caveat). Directions do not transfer across base models. Obfuscation drives probe/SAE/OOD recall from 100% to about 0% (Bailey et al. 2024, u). Low poison rates or complex triggers obfuscate *naturally* (Bailey §4.3). RL-Obfuscation: token-level probes fall, max- or attention-pooled probes hold. Neural Chameleons: evasion generalises zero-shot to unseen probes. Text leakage costs 10–30 AUROC points (Boxo et al., u). On trained, belief-verified liars, balanced accuracy is 0.50–0.65 ("Did you lie?", u). |
| **Model-level, clean base or siblings available** | WeightWatch, SVD of ΔW (Zhong & Raghunathan); embedding-row drift across siblings (SaTML 2024 winners TML and Mitka, checked); Activation Difference Lens (Minder et al., ICLR 2026); crosscoders and Dedicated Feature Crosscoders (Anthropic, Mar 2026); persona-feature SAE diffing (OpenAI, ICLR 2026); LoRA spectral detectors (WSD, Z-PEFT) | WeightWatch flags 44–100% of triggered completions at **< 1.2%** FPR and fully neutralises only 9 of 13 (the review's "< 1%" was wrong). Mitka recovered 2 of 5 triggers exactly. WSD: 97% in-distribution (u). | Needs a reference model, so it does not cover pretraining-stage or from-scratch backdoors. It detects that the model was finetuned, not that the change is harmful. Diluting the finetune erases readability, though the third-party replication shows the topic-level ADL z-score stays above null (about 1.2–1.5 vs 0.2) and dilution there is confounded with step count. Cross-attack transfer is poor (WSD 0.48 off-diagonal, u). |
| **Model-level, no reference, black-box or forward-only** | BAIT target inversion (S&P 2025); Microsoft "Trigger in the Haystack" (2026); perplexity differencing (2605.00994) | BAIT: ROC-AUC 0.961 on 90 models, 0 FP / 3 FN (checked). Haystack: 36 of 41 detected, 0 of 13 clean flagged (u; 0 of 13 still leaves a 95% FPR upper bound of about 25%). | Assumes a fixed, low-entropy target. Haystack is weaker on insecure code. Perplexity differencing fails on trigger-gated behaviour and some sandbaggers. TDC 2023 shows the "functional trigger vs planted trigger" problem: reverse-engineered ASR about 0.99 but trigger recall about 0.16 (u). |
| **Model-level, white-box, no reference, no trigger, no target** | Weight-noise sandbagging test (Tice et al. 2024); activation "intoxication" (Poser: 98%, but *pairwise*, checked); learned weight meta-classifiers (MNTD about 97% (u); Weight Oracles 0.81–0.93 zero-shot (u)); CLIBE (NDSS 2025) and BAN (NeurIPS 2024) weight perturbation; ANP; CLP | This cell is mostly empty for generative transformers. | Meta-classifiers collapse under threat-model shift and evasive trojans (Mazeika et al.; Game of Trojans). They also collapse when poisoning intensity moves outside the moderate band (Yan et al., EMNLP 2025, u). In the sandbagging auditing games (2512.07810) the blue team could not separate sandbaggers; imitation of a weaker model beat black-box methods; probes fired on distractor models; training-based elicitation also raised benign models. |
| **Agentic auditing** | Hidden-objectives game (Marks et al. 2025); Natural Language Autoencoders; AuditBench (14 quirks × 2 implantation methods × 2 adversarial levels, Llama-3.3-70B and Qwen-14B); Introspection Adapters | Hidden objectives: 3 of 4 teams succeeded, and the team that failed lacked *data access*, a confound. NLAs: 12–15% success vs < 3% without (checked). | Scaffolded black-box tools are as good or better (u). There is a tool-to-agent gap. Adversarial anti-confession training hurts every method. n is tiny. No calibrated scores. |
| **Theory** | Goldwasser et al. 2022; Draguns et al. 2024 (cryptographic, cannot be elicited); Christiano, Hilton et al. 2024 (ARC: detectable iff the class is learnable in some regimes, iO blocks it in general; Boltzmann resampling defends against *random* triggers); Bogdanov–Rosen–Vafa (ICML 2026: statistically undetectable white-box backdoors); Sparse-PCA parameter backdoors (2605.04209); Granziol & Abdimanabov (ICML 2026: "near-clone" triggers are spectrally invisible) | | Nothing maps where SGD-*trained* hidden behaviours fall relative to these impossibility results. |

**Interpretability units are weak.**
- **AxBench steering:** prompting 0.894, DiffMean 0.239, SAE 0.165, filtered SAE 0.508 (checked).
- **SAEs on random networks:** SAEs "interpret" randomly initialised transformers about as well as trained ones (Heap et al., u).
- **Circuits are not identifiable** even in small networks (Méloux et al., u).

Detection claims should therefore rest on ground-truth model-level outcomes, not on whether features look interpretable.

**Converging 2025–26 themes worth exploiting.**
- **Equilibrium and stationarity.** Several independent idea-generators converged on "the weights must be explained by the declared data". The pieces exist only for reconstruction (Haim 2022; Buzaglo 2023), unlearning certificates (Guo 2020) and per-input attribution (Keenan 2026).
- **Perturbation leakage.** Gates open under noise or adversarial weight moves: ANP, BAN, CLIBE, Tice, Poser. This works empirically but has no theory and has never been attacked adaptively.
- **Shared backdoor mechanisms.** Backdoor Collapse, Dummy Backdoor and shared SAE latents. So far used only for *removal* (u).
- **Insertion artefacts.** Minder et al.; the Model Organism Lottery (u); AuditBench synthetic-document vs transcript organisms.

---

## 2. The most important open problems

1. **Zero-reference, zero-trigger, zero-target model-level detection.** Decide whether a model contains a hidden conditional policy from its weights plus a sampler for the *declared* distribution, at a calibrated low FPR. This is the setting for pretraining-stage poisoning (Souly et al. 2025: near-constant poison count) and for models trained from scratch.
2. **Variable-output hidden policies** (sandbagging, plausible wrong answers, conditional deception, hidden objectives). Every fixed-target scanner is out of scope here, and white-box probes failed in the auditing games.
3. **Specificity and calibrated nulls.** Tell an undeclared *conflicting* behaviour apart from benign additions, documented conditionals, and the "natural trojans" that clean models also contain. Almost no evaluation includes matched benign controls.
4. **The insertion-artefact confound.** Is detectability a property of the behaviour, or of late narrow finetuning? Nobody has inserted behaviours at pretraining step 0, which is impossible at LLM scale and cheap at toy scale.
5. **Adaptive robustness beyond activation monitors.** Evasion of weight-space, perturbation, equilibrium and functional detectors has never been measured, nor how the obfuscation tax scales with model capacity.
6. **Theory vs practice.** Which SGD-trained behaviours fall in the "undetectable" regime? How does detectability scale with trigger complexity, poison rate, weight decay and training length? Does "cannot elicit the trigger" imply "cannot detect the backdoor"?
7. **A mechanistic account of why perturbation and noise detectors work.** A gate-margin explanation would predict when they fail and how a sandbagger could be hardened against them.
8. **Knowing liar vs incapable model.** Sandbagging detection lacks *capability-matched genuinely weak* controls. Probes confuse "the truth" with "the prescribed action".
9. **Statistical validity.** Evaluations use 1 to 56 organisms, closed-world splits, pooled-input AUROC and no model-clustered CIs. Results are fragile to poisoning intensity (Yan et al.) and to distribution shift (TrojAI round 13).
10. **From a score to an explanation.** Recover the behaviour or trigger with a measured false-discovery rate, and give auditing agents calibrated hypothesis tests rather than raw features (the tool-to-agent gap).

---

## 3. Ranked research directions

The ranking weighs four things: how surprising a positive result would be; how informative a negative result would be; feasibility on 4 CPU cores in a few days; and leverage on existing lab infrastructure (ToyLang zoo, sealed zoo2 and zoo3 pre-registration, `detect/tug*.py`, `analysis/adaptive.py`, `analysis/clean_finish.py`). R1–R3 form a coupled core programme. R4–R8 are independent.

### R1. Tug-of-war forensics: the declared objective's gradient points at the hidden one

**Hypothesis.**
- **Law.** At released weights trained near equilibrium on (1−p)·Clean + p·Hidden, stationarity forces (1−p)·ĝ_clean + λθ ≈ −p·g_hidden. For AdamW this holds in the metric normalised by √v. So the clean-data gradient, computed by an auditor who can only sample the declared distribution, is anti-aligned with the gradient of whatever undeclared objective shaped the weights.
- **Detector.** Project D_clean onto a generic, trigger-free *deviation library*: "emit token v at answer offset j on clean prompts", plus pointer and offset families such as "copy prompt symbol i, ±δ". The most anti-aligned element flags the model and names the behaviour.
- **Specificity follows from the theory.** Only hidden objectives that *compete with* the declared loss on declared-looking inputs produce a tug. An undeclared but non-conflicting extra task should not. This gives an operational definition of misalignment: a training signal the declared objective disprefers.

**Be honest about what is new.** With wd = 0 at exact stationarity the law is close to a tautology (cos = −1 in the Euclidean metric). The science is three questions: (a) how close realistic recipes get to it (AdamW, LR schedules, early stopping, constant LR, SGD); (b) whether a trigger-free library can find the direction; (c) whether it covers variable-output behaviours.

**Novelty: closest prior work and what this adds.**
- KKT and weight-decay reconstruction of training data (Haim 2022; Buzaglo 2023) reconstructs *all* data and has no clean-data subtraction.
- Representer points (Yeh 2018).
- Gradient-residual certificates (Guo, ICML 2020) are for convex unlearning.
- Choi, Shavit & Duvenaud (2023) frame training-data verification.
- Reconstructive Neuron Pruning (Li et al., ICML 2023) uses clean-loss gradient ascent to expose backdoor neurons. This is the nearest mechanism; it is a purification method that is not calibrated.
- cupbearer's FinetuningAnomalyDetector (relaxation diffing) and Keenan 2026 (per-input, needs triggered inputs).
- Zhang et al. (S&P 2024) report that backdoor and clean updates are near-orthogonal *during* training. Reconcile this explicitly: the law is about *residual* gradients at the stationary point, which are forced to balance.

The delta: a model-level, base-free, trigger-free, target-free verdict *with behaviour recovery*, covering from-scratch insertion and variable-output families. It is also a calibrated hypothesis-test primitive for auditing agents: "does this model's equilibrium support hidden behaviour X?", with p-values against fake hypotheses.

**Experiment.**
- **Zoos.** Keep the pre-registered three-zoo protocol: zoo1 for development, zoo2 sealed in-distribution, zoo3 sealed held-out. zoo3 holds out trigger kinds (count, lenfix, desc), behaviours (sortdesc, payload8, dup, inject_last), architecture (L = 4, d = 128, h = 8), optimiser (SGD+momentum), constant LR, early insertion, and 0.5% poison.
- **New families:**
  - F5, sandbagging: a plausible wrong answer (e.g. an off-by-one sort) under the trigger;
  - F6, conditional flip ("unmonitored" flag leads to negated or reversed output);
  - F7, *undeclared conflicting benign rule*. It should be flagged; report it separately as "undocumented spec override";
  - F8, declared conditional rule included in the auditor's sampler. It should not be flagged.
- **Recipe arms:** wd ∈ {0, 0.05, 0.1}; stop at 50%, 75% or 100% of steps; cosine-to-zero vs constant LR; clean-finish of k ∈ {0, 250, 1000} steps.
- **Auditor mismatch arm:** the auditor's sampler is deliberately shifted (task mixture ±30%, length distribution shifted) to measure false positives from distribution mismatch.
- **Size:** about 210 models in zoo2, about 180 in zoo3 (including F5–F8 and recipe arms), and about 60 adaptive (see R2). Each held-out split needs at least 25 effective positives.
- **Baselines (same access tier):**
  - embedding and unembedding row norms vs clean token frequency;
  - ‖g_clean‖ and clean loss alone;
  - weight spectral statistics plus a linear meta-classifier trained on zoo1 (a supervised bar);
  - Mahalanobis and kNN on clean-input activations;
  - Tice weight noise and absolute activation noise;
  - BAIT-style target inversion and MELBO steering, both at matched compute;
  - RNP and finetune-KL;
  - **a max-statistic null:** the same score with the library replaced by random gradient directions of matched norm.

**Statistics.**
- **Primary (pre-registered):** H2 AUROC on zoo2, effective backdoors vs clean + extra, with a stratified bootstrap CI and a 10k-permutation p-value.
- **Comparisons:** paired bootstrap or DeLong against each baseline, Holm-corrected secondaries, per-split AUROC on zoo3, TPR at 5% FPR (about 100 clean models cannot certify 1%).
- **Recovery (H3):** Wilson CIs against the 1/|library| chance rate.
- **Power:** at AUROC 0.85 with 40 positives and 80 negatives, the CI half-width is about ±0.08.

**What would be "really amazing":**
- zoo2 AUROC ≥ 0.95 with the CI lower bound ≥ 0.90;
- every zoo3 split ≥ 0.85, *including F5 and F6*, where fixed-target scanners cannot run;
- top-1 behaviour recovery ≥ 60% against a chance rate below 1%;
- F7 flagged and F8 not flagged (TOST for F8 against clean, margin 0.05);
- an advantage over the best baseline of at least 0.10 AUROC, p < 0.01.

**Risks and confounds.**
- Reduction to trivial statistics. The library is on the output side, but the rare-token baseline must still be reported.
- Non-stationarity under constant LR or early stopping (this is a measured axis, not a failure).
- The clean-gradient noise floor at small p.
- Distribution mismatch producing false positives. This is the biggest realism threat.
- Behaviours outside the library are caught only through partial overlap.
- The weak subset-trigger cell in the pilot.

**If the result is negative.**
- If the law holds but the trigger-free detector fails, the bottleneck is the library. The method is then still a calibrated tester of a *given* hypothesis, which addresses the tool-to-agent gap.
- If the law itself fails under realistic recipes, that bounds every KKT- or stationarity-style forensic method. Both outcomes are publishable.

**Compute.** Training about 390 models at roughly 165 s each (zoo3's larger models about 3×) is about 6–7 h wall-clock on 4 workers. The detector costs about 1–3 min per model, roughly 2 h. **Total about 10 h.**

### R2. The equilibrium trilemma: evasion cost and cross-modality obfuscation transfer

**Hypothesis.**
- **Coupling.** Under clean finetuning (a stand-in for safety training) the first-order change in hidden loss is −η·g_hᵀPg_clean. The same anti-alignment that makes a model tug-detectable therefore makes its backdoor erode.
- **Trilemma.** A hidden behaviour released at clean equilibrium with weight decay must do one of three things: (a) show a first-order tug residual; (b) be eroding; or (c) live in clean-degenerate directions that weight decay shrinks, so persistence then needs λ ≈ 0.
- **Predictions:**
  1. The ASR half-life under clean finetuning is predicted by the pre-finish |cos|·‖g‖_P product.
  2. Evading the tug detector costs ASR (through erosion), clean loss (through a gradient-alignment penalty), or capacity (memorising both objectives below the noise floor).
  3. Activation-monitor evasion is cheap and transfers between activation monitors but not to equilibrium or functional detectors.
  4. The only cheap evasion is wd = 0 plus a clean finish. That pins down exactly where second-order signals (R7, `detect/dormant.py`) have to take over.

**Novelty.** Prior adaptive-attack work covers activation monitors (Bailey et al.; RL-Obfuscation; Neural Chameleons; Obfuscation Atlas (u)) and CNN meta-classifiers (evasive trojans; Game of Trojans; Tan & Shokri 2020). BAD-BOOM (ICML 2026) builds flat backdoor basins for persistence, which is a relevant attack to include. Keenan 2026 is robust to activation obfuscation but was never attacked adaptively itself. What is new: adaptive attacks against equilibrium, weight, perturbation and functional detectors in transformers; a transfer matrix across them; tax scaling with width; and a *quantitative persistence–detectability coupling law*.

**Experiment.**
- **Detector families:**
  - A1 mean-pooled and A2 max/attention-pooled probes (input-level);
  - A3 Mahalanobis;
  - B weight-spectral meta-classifier;
  - C noise leakage, attacked through a differentiable surrogate (expected payload probability under sampled noise);
  - D tug, attacked through a double-backprop penalty on cos(D_clean, D_deviation) and separately on ‖g_clean‖² + ‖g_hidden‖²;
  - E Keenan-style SGLD loss-trace attribution, attacked through a surrogate.
- **Attackers:** 5 targeted families plus a union attacker, × 3 strengths λ × widths d ∈ {48, 96, 192} × 5 seeds, about 270 runs. Add clean-finish k ∈ {0, 250, 1000, 4000} under wd ∈ {0, 0.1}, and a SAM/BAD-BOOM-style flat-basin attacker.
- **Evaluation:** re-fit every monitor on the frozen attacked model (the defender's turn), then evaluate every detector on every model.

**Statistics.**
- Transfer matrix of ΔAUROC with model-bootstrap CIs.
- Within-modality vs cross-modality transfer gap, by permutation over family pairs.
- Frontier plots of AUROC against (ASR, Δclean loss) with CIs.
- Fit log(tax) = α·log d + c and report α with its CI.
- Coupling: a mixed model of log half-life on pre-finish |cos|·‖g‖_P, with seed as a random effect.

**What would be "really amazing":**
- Tug AUROC stays ≥ 0.85 at every attack setting that keeps ASR ≥ 0.9 and clean loss within tolerance, while activation monitors fall to ≤ 0.6. That would be the first transformer detector with a demonstrated evasion cost.
- The coupling holds with Spearman ρ ≥ 0.7.
- α has a CI excluding 0.

**Risks.** Gradient masking or failed double-backprop optimisation (Athalye et al. 2018); mitigate with several strengths, surrogates and reporting the attack loss reached. Surrogate-vs-real mismatch. A cheap flat-basin evasion.

**If the result is negative.** A cheap evasion recipe (for example wd = 0 plus a clean finish) is still a key result. It defines the threat model an auditor must assume and motivates R7.

**Compute.** About 270 runs at about 4–8 min each (double backprop costs 2–3×), about 25 CPU-hours or 6–7 h wall-clock; plus about 3 h of evaluation. **Total about 10 h.**

### R3. Is detectability an insertion artefact? An insertion-route × insertion-time factorial

**Hypothesis.** Insert the same behaviour, at matched poison-token count and matched ASR, by six routes:
- post-hoc narrow finetune;
- finetune diluted 1:1 or 1:5;
- co-trained from scratch at insertion time τ ∈ {0, 0.25, 0.5, 0.75, 0.9}.

Prediction: activation-diff, probe, perplexity-differencing and meta-classifier power decays monotonically as insertion moves earlier and more diffusely, while equilibrium and perturbation detectors are approximately invariant. The reverse ordering would be equally informative.

**Novelty.** Model Organism Lottery (u) covers 7 post-training-only techniques at 1B. Minder et al. study dilution post hoc, confounded with step count. Yan et al. (u) vary intensity. AuditBench compares synthetic documents with transcripts. Souly et al. and Zhang et al. poison during pretraining but evaluate no detectors. Nobody has the τ = 0 cell, a matched dose, a multi-detector panel and adequate power.

**Experiment.**
- **Design:** 3 families (fixed-target backdoor, sandbagging, conditional flip) × 7 conditions × 10 seeds = 210 models, plus 60 clean models.
- **Decoys:** **30 benign late-narrow-finetune decoys**, so that "was finetuned late" is not scored as "has a hidden behaviour".
- **Sibling control:** each model has a τ-matched clean sibling sharing initialisation and data order, so diff-based detectors are defined at every τ.
- **Constant-LR arm** to separate τ from LR decay.
- **Detector panel:**
  - transferred probe and Mahalanobis (input-level tier);
  - ADL-analog, weight-diff SVD and perplexity differencing against the sibling;
  - Tice noise; BAIT-style inversion; R1 tug;
  - a weight meta-classifier trained only on τ = 0.98 organisms (this directly measures whether learned detectors pick up the insertion artefact).

**Statistics.** Mixed-effects logistic regression: detected-at-5%-FPR ~ detector × τ, with random effects for seed and family. Jonckheere–Terpstra trend test per detector. Holm correction across 9 detectors. TOST (±0.05) for any invariance claim.

**What would be "really amazing":** at least 3 detector families lose ≥ 0.2 AUROC between late finetune and τ = 0 (p < 0.001); at least one family is TOST-invariant at AUROC ≥ 0.9; and the late-trained meta-classifier drops to ≤ 0.6 at τ = 0. That combination would show that closed-world model-organism numbers are inflated by how organisms are built.

**If the result is negative.** Universal invariance would validate narrow-finetune organisms as proxies, at least at toy scale.

**Risks.** τ is confounded with consolidation time and weight-decay erosion, so log both. ASR matching may fail. Results may not carry over to LLM scale.

**Compute.** About 300 models (finetune runs about 1 min, co-trained about 3 min) plus 9 detectors at about 1 min each. **Total about 8 h.**

### R4. Capability-matched sandbagging: a gap between latent knowledge and output vs a noise-robust lock

**Hypothesis.**
1. A password-locked strong model still linearly encodes the correct answer at mid layers, decodable by a *per-model* probe on held-out items, while an accuracy-matched genuinely weak model does not. This gives an absolute, reference-free detector.
2. Tice-style noise detects the lock only when σ·‖J_gate‖ reaches the gate margin m_g. The location of the bump is predicted by m_g/‖J_gate‖, and noise-robust locks defeat it.
3. "Deep locking", meaning scrubbing the internal answer, costs measurable capability.

**Novelty.** Tice 2024; Poser (pairwise); Mallen quirky models (same model, different context); Halawi "overthinking"; Greenblatt password-locked models; the auditing games, which had no capability-matched weak controls and saw false positives from elicitation. What is new: a sample of at least 50 per-item accuracy-matched pairs, a noise-robust adversary, a mechanistic bump prediction, and a measured tax for deep locking.

**Experiment.**
- **Tasks:** modular addition, multi-digit addition, sort.
- **Model:** d = 128, L = 4, about 1–3M parameters.
- **Classes, 20 seeds each:**
  - lock co-trained; lock added post hoc;
  - **imitation lock** (copies a real weak model's outputs);
  - noise-robust lock (trained under sampled weight noise);
  - deep lock (penalty on probe decodability, using held-out probes);
  - weak-matched, built three ways (fewer steps, less data, smaller width) and matched within ±2 points of accuracy and on per-item correlation;
  - strong honest; benign format-quirk distractor.
- **Total:** about 180 models.
- **Probe-capacity floor:** probe on random-initialisation models and on weak models.
- **Baselines:** Tice noise, activation noise, R1 tug, few-shot finetuning elicitation (with a false-positive check on weak models), logit lens.

**Statistics.** Per-construction AUROC with permutation tests. Paired DeLong against Tice. R² of the observed vs predicted bump location. A *max-statistic* null for unlock search on the weak controls. A regression of capability cost on deep-lock strength.

**What would be "really amazing":** gap AUROC ≥ 0.95 against weak-matched controls on imitation and noise-robust locks, where Tice falls to ≤ 0.65; bump-prediction R² ≥ 0.7; and pushing the gap AUROC to ≤ 0.65 by deep locking costs ≥ 10 points of capability.

**If the result is negative.** If deep locking is free, internals-based sandbagging detection has no guarantee, which matters for evaluation integrity.

**Risks.** Weak models may encode the answer partially. The probe might compute the answer itself; the random-initialisation floor guards against this. "Knowing" is crisp in toy tasks and not in LLMs.

**Compute.** About 180 models at about 5 min each, plus detectors. **Total about 6 h.**

### R5. Trigger-complexity phase diagram: model-level detection vs trigger elicitation

**Hypothesis.** Climb a ladder of trigger complexity: 1 token, m-token conjunction, ordered n-gram, k-sparse positional parity, then a learned keyed hash ((Σaᵢxᵢ mod q) = r, rejected from clean data). Trigger elicitation (GCG, BAIT, MELBO) and input-level activation detectors should collapse along the ladder, while payload-side model-level detectors (R1, noise leakage) stay flat. That would show that "cannot find the trigger" does not mean "cannot tell the model is backdoored". Alternatively, finding a level where everything collapses at ASR ≥ 0.9 would be the first empirical undetectable regime for *trained* transformers.

**Novelty.** Draguns (compiled circuits); ARC 2024 (function class with *random* triggers; do not overclaim a test of their theorem); Bailey §4.3; TDC 2023's recall-vs-REASR gap; Barak 2022 on learning parities. What is new: trained transformers, a controlled ladder, conditioning on achieved ASR, and a multi-detector dissociation.

**Experiment.**
- **Design:** 6 levels × 3 poison rates × 12 seeds = 216 backdoored models plus 72 clean, d = 64–96. Run 6k steps for parity and hash.
- **Generalisation check:** the trigger must fire on *held-out* inputs that satisfy the predicate, to rule out memorising specific poison inputs.
- **Learnability:** record samples-to-ASR as a covariate.
- **Elicitation success** is scored two ways: any functional trigger, and recovery of the intended predicate.

**Statistics.** Mixed logistic regression with a complexity × detector-class interaction, tested by likelihood ratio. Per-detector trend tests. TOST to claim flatness. Report results both for all models and conditioned on ASR, with the selection bias stated.

**What would be "really amazing":** elicitation falls from ≥ 80% to ≤ 10% while R1 stays ≥ 0.9 (CI lower bound ≥ 0.85) at every level.

**Risks.** High-complexity triggers may not be learnable at low poison rates. The payload may weaken. Selection bias from filtering on ASR.

**Compute.** About 290 models at about 4 min, plus GCG at about 5 min per model. **Total about 12 h.**

### R6. Overt-to-latent transfer: train on unconditional behaviours, detect gated ones

**Hypothesis.** A conditional policy reuses the payload subspace of its unconditional version. A detector trained only on overt-vs-clean finetunes therefore catches trigger-gated versions zero-shot and names the payload. This would mean detectors can be built without building dangerous gated organisms.

**Novelty.** No prior evidence found. Nearest work: MacDiarmid's within-base cross-goal transfer (activations); Soligo's convergent emergent-misalignment directions (overt only); DIT, PEFTGuard and ProbeX (same regime for training and testing); Betley's overt/backdoor dataset pair (no detector study).

**Experiment.**
- **Bases:** at least 6 clean bases (d = 128, L = 4) for cluster independence.
- **Finetunes (about 400):**
  - 100 matched controls, in which trigger tokens and payload outputs appear at matched frequency but uncoupled;
  - 100 overt, over K = 8 payloads;
  - 200 gated, with token, bigram, positional and conjunctive triggers.
  - Two payloads are held out entirely. Full finetuning and LoRA r = 4 both appear.
- **Detectors:**
  - logistic regression on per-layer ΔW;
  - singular values of ΔW plus a *weight-space logit lens* (W_U·ΔW_OV, W_U·ΔW_out);
  - DeepSets over heads;
  - R1 tug, unsupervised, as a comparison.
- **Must-beat baseline:** ΔW Frobenius norm, which should be about chance against the matched controls.

**Statistics.** Train on overt, test on gated. Cluster bootstrap over bases. Exact binomial test for payload naming against 1/K. Analysis stratified by ΔW norm.

**What would be "really amazing":** ≥ 0.9 AUROC zero-shot on gated models and ≥ 3× chance payload naming, including the held-out payloads.

**If the result is negative.** Gated policies use distinct subspaces. That bounds DIT and Introspection Adapter generalisation.

**Compute.** About 7 CPU-hours. **Total about 3 h wall-clock.**

### R7. The complexity cost of secrecy: singular learning theory signatures of dormant gates

**Hypothesis.** Compare matched twins (same seed and data stream, with poison swapped for clean). Their clean-data local learning coefficient (LLC) should differ:
- if it is higher in the backdoored twin, the gate adds constrained directions;
- if it is equal, the backdoor lives in clean-degenerate directions. This is case (c) of R2's trilemma, and it explains sleeper-agent persistence.

The difference ΔLLC should scale with trigger bits. Restricted LLC per head should localise the gate. Trigger-free geometric features should predict persistence half-life better than Hessian and norm features.

**Novelty.** No LLC-based backdoor detection exists (the Timaeus research feed through October 2026 has none). Keenan uses SGLD for per-input attribution. BAD-BOOM measures basin width but needs the trigger. Chen & Murfet (2025) show the LLC ignores data modes below an n·β threshold, which gives an *a priori* reason to expect a null, so sweep n·β. Billa (2026) found the LLC synchronous with capability emergence rather than leading it.

**Experiment.**
- **Twins:** 50 pairs across trigger complexity × widths {32, 64, 128}.
- **Controls:** extra-capacity twins (equal benign memorisation) and disclosed-rule twins.
- **LLC estimation:** devinterp v2 from PyPI. Freeze SGLD hyperparameters on clean models only (4 chains × 300 draws). Sweep n·β ∈ {10, 100, 1000}.
- **Persistence:** clean finetuning with and without near-miss adversarial training; measure ASR half-life.

**Statistics.** Paired Wilcoxon test on ΔLLC. Spearman correlation with trigger bits. Cross-validated R² for half-life, and ΔR² over Hessian, norm and effective-rank features, with a bootstrap CI. TOST (±5%) to claim a null.

**What would be "really amazing":** the sign of ΔLLC is consistent in ≥ 90% of pairs with a dose-response in trigger bits, and persistence R² ≥ 0.5 with ΔR² ≥ 0.2 over baselines.

**If the result is negative.** "Dormancy is geometrically free at the clean-data temperature" is a mechanistic explanation of persistence and a bound on SLT-based auditing.

**Risks.** SGLD sensitivity and convergence. Loss level as a confound (regress it out). LayerNorm scale symmetries.

**Compute.** About 150 models (about 8 CPU-hours) plus LLC and restricted LLC (about 35 CPU-hours) plus persistence finetunes (about 8 CPU-hours). **Total about 13 h wall-clock.**

### R8. Emergent misalignment from scratch: a causal test of the persona hypothesis

**Hypothesis.** Pretrain on text from agents whose latent disposition z corrupts outputs consistently across domains, with sharing strength ρ. Narrow finetuning on malicious outputs in one domain should then raise the specific corruption rate in held-out domains, with a dose-response in ρ. The finetune's activation delta should align with a pre-existing z-direction. The projection of the first-step finetuning gradient onto that direction should predict emergent misalignment *before* finetuning.

**Novelty.** Betley; OpenAI persona features; Soligo's convergent directions; Persona Vectors (Chen et al. 2025). All of these use pretrained LLMs, where z cannot be manipulated, and emergent misalignment's robustness is contested ("Mirage" and "Overtrained", u). GINC and subliminal learning are methodological precedents. This would be the first manipulated-cause test, plus a controlled comparison of the "EM is easy, narrow is hard" claim.

**Experiment.**
- **Data:** ToyLang-Persona with 8 domains and weak stylistic cues for z. ρ ∈ {0, 0.25, 0.5, 0.75, 1}.
- **Bases:** 8 seeds per ρ, 40 bases (d = 128–192, L = 4, about 2–4M parameters).
- **Finetunes:** narrow domain-1 malicious finetune, versus a matched random-corruption finetune and a benign finetune.
- **Metric:** exact corruption-type rate in the 7 held-out domains, with no judge.
- **Analysis:** z-probe on the base, cosine of the delta with z, z-ablation, and the pre-finetuning predictor.

**Statistics.** Permutation over seeds against the control. Spearman ρ across 40 runs. Ablation effect. Predictor correlation on held-out seeds.

**What would be "really amazing":** at ρ = 1, emergent misalignment exceeds the control by ≥ 10 points (p < 0.001); Spearman ≥ 0.7 across ρ; z-ablation removes ≥ 70% of it; the predictor reaches ρ ≥ 0.6.

**If the result is negative.** A null with a decodable z means a latent persona is not sufficient for emergent misalignment, which contests the mechanism. A null with no decodable z points to a scale limit, which is less informative.

**Compute.** About 20 CPU-hours for the bases plus about 4 for finetunes. **Total about 7 h.**

**Considered but not ranked.**
- **Common-cause attribution coherence for measurement tampering.** Novel and ELK-relevant. Validate on ARC's distinction tasks (`gs://arc-ml-public/distinctions` is reachable; `hex_task.json` has already been downloaded) and the Redwood diamond generator on GitHub.
- **Universal-coherent-mode steering and the dormant-gain generalised eigenproblem.** High overlap with Neural Cleanse, BEEAR, MELBO, DCT and BAIT; use them as baselines in R1 instead.
- **Population-aligned weight residual for from-scratch models.** Variation across seeds likely swamps the signal.
- **Sibling-zoo robust diffing.** The federated-learning robust-aggregation literature already covers this.

**Recommended schedule (about 3 CPU-days).**
- Day 1: freeze the R1 detector on zoo1 and record the commit hash in `PREREGISTRATION.md`. Train zoo2 and zoo3 in the background. Build the R3 zoo, which doubles as an out-of-distribution test bed for R1.
- Day 2: score the sealed zoos once. Run the R2 adaptive and clean-finish arms against R1.
- Day 3: run R4 or R5.

**Pivot rule.** If R1's primary AUROC on zoo2 is below 0.8, write the program up around R2, R3 and the measured negative result, rather than tuning against the sealed zoos.

---

## 4. Methodological standards we must meet

1. **The model is the unit of analysis.** Report model-level AUROC with a stratified bootstrap over models (2,000 resamples), cluster-resampled by base or seed whenever models share one. Never pool prompts across models. Compare detectors paired, on the same models (paired bootstrap or DeLong).
2. **Plan power before training.** At AUROC 0.85 with 40 positives and 80 negatives, the 95% CI half-width is about 0.08. At 0.90 with 60 vs 60 it is about 0.057. Differences smaller than about 0.1 need paired designs. The smallest FPR you can resolve is about 1/n_clean: report TPR at 5% FPR, and claim 1% only with roughly 300 or more clean models.
3. **Sealed, pre-registered test zoos.** Choose every hyperparameter (layers, library, noise scale, SGLD settings, thresholds) on the development zoo. Commit the detector hash *before* generating test seeds. Score each sealed zoo once. Keep one primary endpoint per hypothesis and Holm-correct the secondaries. Keep exploratory and confirmatory analyses separate. Report every detector tried, including the failures.
4. **Generalisation splits by group.** Hold out entire trigger kinds, behaviour families, architectures, optimisers, LR schedules, insertion times and poison rates, and report each family separately, since a pooled AUROC hides family failures. Ideally zoo3's generator is written by a separate process (a different agent or author) that never sees the detector code, as in the auditing games' red-team design.
5. **Matched negatives, not just clean models.** Use clean twins (same seed, data stream, recipe and steps, minus the poison). Include:
   - undeclared non-conflicting extra tasks;
   - undeclared conflicting benign rules;
   - declared conditionals;
   - benign late narrow finetunes;
   - format-quirk distractors;
   - for sandbagging, genuinely weak per-item-matched models.
   
   Specificity claims need equivalence tests (TOST with a stated margin), not "p > 0.05".
6. **Verify ground truth.** Gate positives on ASR (≥ 0.5 effective, and report ≥ 0.9 separately) and on trigger generalisation to held-out satisfying inputs. Check that clean models lack the behaviour. Report the exclusions, stratify results by implant strength, and state the selection bias from filtering on ASR.
7. **Trivial and null baselines are mandatory.**
   - Max-over-random-directions or random-library nulls (the 95.3% lesson from the sleeper-agent probes).
   - Embedding and unembedding norms against clean token frequency; clean loss; gradient norm; ΔW norm.
   - Noise injection.
   - Compute-matched BAIT-style and MELBO-style search.
   - Any detector that searches (a max over directions, cells or prompts) needs a *max-statistic* null from the same search on clean models.
8. **Compare within an access tier.** Never set an input-level detector (one that needs triggered inputs) against a model-level one, or a reference-based method against a reference-free one, without labelling the tier. Match compute in forward and backward passes.
9. **Adaptive attackers are part of the evaluation.** White-box attacks against the *final* detector, with the detector re-fit after the attack. Evaluate against held-out detector families. Use several attack strengths and long schedules, and report the attack loss reached. Check for gradient masking with surrogate and ensemble attacks. Report the evasion cost (ΔASR, Δclean loss) along a frontier, not a single point. Include intensity extremes (Yan et al.), wd = 0, early stopping and clean-finish recipes.
10. **No leakage between development and test organisms.** Use disjoint RNG seeds, triggers and payload identities. Never refit a normaliser or threshold on test models. For learned meta-classifiers, split by base or seed group so near-duplicates do not cross the split.
11. **Validate mechanisms causally.** Ablate the ground-truth backdoor components (identified by ablation-on-ASR) and show the detector signal drops, against random ablations of matched norm. Score trigger and behaviour recovery against an empirical false-discovery rate on clean models, separating "any functional trigger" from "the planted trigger".
12. **Report the recipe for stationarity and SLT methods:** final gradient norm, LR schedule, optimiser, how D_clean reproduces across samples, the auditor-distribution mismatch arm, SGLD convergence diagnostics (multi-chain agreement), and an n·β sweep.
13. **Keep scale claims modest.** Present toy results as mechanisms, laws and scaling exponents with CIs across at least 3 widths, not as LLM-level AUROC. Where possible, add an external check: ARC distinction tasks from GCS, Redwood diamonds, or a 5–20M-parameter LM trained on CPU from GitHub-hosted text and then finetuned with the same backdoors.
14. **Hygiene for unverified citations.** Every 2026 number marked (u), and every novelty claim, gets re-checked against the PDFs before write-up. Search specifically for prior work on "stationarity or KKT residual for backdoor detection", "clean–backdoor gradient conflict at convergence", and adaptive attacks on gradient, attribution or weight-space monitors.

**Files referenced:** `/home/user/aegis/lab/PREREGISTRATION.md`, `/home/user/aegis/lab/detect/tug_lib.py`, `/home/user/aegis/lab/detect/tug.py`, `/home/user/aegis/lab/analysis/report_law.py`, `/home/user/aegis/lab/runs/zoo1/analysis_tug_law.jsonl`, `/home/user/aegis/lab/organisms/zoo.py`, `/tmp/claude-0/-home-user-aegis/34d4f9d3-7ae9-5c70-845f-5c29d5bef109/scratchpad/arcdist/hex_task.json`.