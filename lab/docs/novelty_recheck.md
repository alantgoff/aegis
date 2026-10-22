**Novelty audit of claims (a) to (e), as of 2026-10**

**How this was searched.** WebSearch worked; I ran about 45 queries, mostly in standard mode and twice in extended mode. WebFetch did not work: arxiv.org, export.arxiv.org, alphaxiv, cs.purdue.edu and ojs.aaai.org all failed with a DNS error or a proxy 403. So paper details below come from search-result snippets and abstracts, not full texts. Anything marked [own knowledge] could not be checked on the web.

---

**(a) Clean gradient is anti-aligned with the hidden-objective gradient at released weights (equilibrium "tug-of-war")**

**Verdict: partially known.** The underlying identity is textbook. I found no paper that measures this anti-alignment at released weights across a population of models.

Closest work:
- Zhang et al., "Exploring the Orthogonality and Linearity of Backdoor Attacks", S&P 2024 — https://www.cs.purdue.edu/homes/an93/static/papers/SP24_Zhang.pdf
  - Treats backdoor poisoning as continual learning.
  - "Orthogonality" means the backdoor and benign gradients are near-orthogonal, measured during training, not at release.
  - A later paper, BI-BAU (https://arxiv.org/pdf/2606.14078), repeats the orthogonality observation.
- Papers that report negative clean-vs-backdoor gradient cosine, but only during training or between attack updates:
  - ATAAT: standard backdoor methods give Sim(θ) ≪ 0 — https://arxiv.org/pdf/2605.08612
  - A speech paper that applies PCGrad to clean vs backdoor tasks — https://arxiv.org/pdf/2607.01702
  - Revocable backdoors: attack vs unlearning gradient cosine of about −0.65 — https://arxiv.org/pdf/2510.13322
- TSBD, Lin et al., NeurIPS 2024 — https://arxiv.org/pdf/2405.20291
  - Weight changes from clean unlearning and from poison unlearning are positively correlated.
  - Backdoored neurons have a larger gradient norm on clean data.
  - This is the closest empirical cousin, but it works with per-neuron magnitudes, not signed gradient directions.
- Karim et al., Natural Gradient Fine-tuning (NGF) — https://arxiv.org/pdf/2306.17441
  - At release, backdoored models sit in sharper minima of the clean loss because they fit two distributions. This is a second-order signature.
- Methods that use stationarity or gradient similarity but need training data:
  - Hammoudeh and Lowd, renormalized influence (GAS), CCS 2022 — https://arxiv.org/abs/2201.10055
  - Haim et al., NeurIPS 2022, use KKT stationarity at trained weights to reconstruct training data — https://arxiv.org/abs/2206.07758. Same "stationarity leaks the training mixture" logic, applied to a different object.

What remains new:
- A statement and measurement that, at released weights, g_clean is anti-aligned with g_hidden as a consequence of equilibrium.
- Measuring it across a zoo of clean, backdoor and fine-tuned models.
- Squaring it with Zhang's finding: large, near-orthogonal gradients during training, then anti-parallel leftover gradients at equilibrium.

Caveats the claim must handle:
- With weight decay the equilibrium is ∇L_mix + λθ = 0. Clean models then also have a nonzero g_clean, anti-aligned with λθ, which is a confound.
- With early stopping and Adam, the "equilibrium" is only approximate.
- When both losses are near 0, the leftover gradients are tiny and noisy.

---

**(b) Trigger-free detection and behaviour recovery from the clean gradient (second-moment normalised) scored against a generic library of output deviations**

**Verdict: partially known.** The goal and the constraints have precedent. The mechanism (using the clean gradient as a pointer into a behaviour library) was not found.

Closest work:
- CLIBE, NDSS 2025 — https://ndss-symposium.org/ndss-paper/clibe-detecting-dynamic-backdoors-in-transformer-based-nlp-models
  - Applies a few-shot weight perturbation to attention layers to push reference samples toward a target, then checks whether the effect generalises.
  - Needs no trigger inputs and has been extended to text generation.
  - It enumerates candidate targets and optimises a perturbation; it does not use the clean gradient.
- ANP, Wu and Wang, NeurIPS 2021 — https://papers.neurips.cc/paper_files/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf
  - Perturbing neurons to maximise clean loss makes clean inputs predict the target label.
  - It is used for pruning, on classifiers only.
- BAIT, S&P 2025 — https://www.cs.purdue.edu/homes/cheng535/static/papers/sp25_bait.pdf
  - Needs no prior knowledge of trigger or target; inverts the target by enumerating its first token.
  - Black-box, and relies on the backdoor "leaking" without the trigger.
- BEEAR, EMNLP 2024 — https://arxiv.org/abs/2406.17092
  - Finds universal embedding perturbations toward a defender-defined set of unwanted behaviours, with no knowledge of the trigger.
  - Closest to the "library" idea, but it is a removal method working on embeddings.
- Trigger-free elicitation methods:
  - MELBO (unsupervised steering vectors) — https://greaterwrong.com/posts/ioPnHKFyy4Cw2Gr2x
  - Qin, Hua, Marks, Conmy, Nanda, "Discovering Backdoor Triggers": works on toy backdoors, fails on realistic ones — https://www.alignmentforum.org/posts/kmNqsbgKWJHGqhj4g
  - Fuzzing with random weight or activation noise — https://arxiv.org/pdf/2606.29646
- Defection probes (Anthropic 2024) — https://anthropic.com/research/probes-catch-sleeper-agents
  - Built from generic contrast pairs with no trigger or behaviour knowledge.
  - Works on activations and classifies inputs, not models.
- Methods that need a base or reference model, or a population of models:
  - Perplexity differencing: a cross-family reference model works — https://www.emergentmind.com/papers/2605.00994
  - Diff amplification — https://www.goodfire.ai/papers/model-diff-amplification
  - Spectral detection of LoRA adapters: needs a population of adapters — https://arxiv.org/pdf/2602.15195
  - Li and Kim, ICLR 2026: purification without prior knowledge or a clean reference — https://arxiv.org/abs/2603.13461
- GradSafe — https://arxiv.org/pdf/2402.13494
  - Scores by cosine against a library of reference gradients, but works per prompt, not per model.

What remains new:
- Scoring a generic library of deviations by their alignment with the Fisher- or second-moment-preconditioned clean gradient at the released weights.
- No base model, no reference population, no trigger or target search, and essentially one backward pass per library item.
- A theoretical justification for the score, from (a).

Limits to state:
- Worst-case impossibility results exist: Draguns et al., NeurIPS 2024, unelicitable backdoors (https://arxiv.org/pdf/2406.02619), and Goldwasser et al. 2022, undetectable backdoors [own knowledge].
- Recall is capped by what the library covers. A behaviour outside the library can at best be detected, not recovered.

---

**(c) Gradient ascent on clean loss elicits the hidden behaviour onto clean prompts; descent erases it**

**Verdict: known for classifiers (ANP, RNP, TSBD); partially known for generative models. "Descent erases" is long known, and known to be unreliable.**

Closest work:
- ANP, as in (b): maximising clean loss in neuron space makes clean inputs predict the target label. This is a direct precedent.
- RNP, Li et al., ICML 2023 — https://arxiv.org/abs/2305.14876
  - Clean unlearning (gradient ascent) leaves backdoor neurons intact.
  - The unlearned model is reused for trigger recovery and for detecting the backdoor label.
  - [own knowledge, not quoted in any source I retrieved] the unlearned model's predictions collapse toward the target class.
- TSBD (https://arxiv.org/pdf/2405.20291) and ABL (https://arxiv.org/abs/2110.11571). ABL needs isolated poisoned data.
- Arazzi et al., clean-unlearning attack — https://arxiv.org/pdf/2506.12522
  - Unlearning clean samples can more than double backdoor success rate. This is an attacker-engineered version of "ascent amplifies".
- "Don't Shift the Trigger", ICLR 2026 — https://iclr.cc/virtual/2026/poster/10006778
  - Plain gradient ascent shifts the trigger to other classes. This is a caveat for any claim about the sign of the effect.
- "Backdoor Collapse" (Locphylax), ACL 2026 — https://arxiv.org/pdf/2510.10265
  - Injects known backdoors so unknown ones cluster together. It does not use gradient ascent.
  - The ICML 2026 basin paper below uses "collapse" in a different sense.
- BAD-BOOM, ICML 2026 — https://icml.cc/virtual/2026/poster/66722
  - Backdoors in sharp, narrow basins collapse under SFT; its Fisher-ellipsoid SAM widens the basin so they persist.
  - This is a natural adaptive attack: it likely weakens first-order coupling, and so weakens both (b) and (c).
- Evidence that descent-based erasure is often superficial:
  - Min et al., NeurIPS 2024, superficial safety and reactivation — https://arxiv.org/abs/2410.09838
  - FT-SAM, ICCV 2023: plain fine-tuning barely moves backdoor neurons — https://arxiv.org/pdf/2304.11823

What remains new:
- Applying this to whole-sequence generative behaviours (for example, reverse instead of copy), not labels.
- A first-order mechanism, the identity from (a), that predicts the elicited behaviour is the hidden one, with a signed dose-response: ascent amplifies, descent erases.
- Using it as an auditing readout rather than for pruning.
- The claim must engage with trigger-shifting and superficial-erasure, and limit "descent erases" to first order.

---

**(d) Duality: first-order detectability equals first-order persistence (erosion rate = g_hidden^T P g_clean)**

**Verdict: partially known.** Each half has precedent; the joint statement was not found.

Closest work:
- The first-order forgetting formula is textbook continual learning:
  - GEM (Lopez-Paz and Ranzato 2017) [own knowledge]
  - OGD (Farajtabar et al.) — https://alphaxiv.org/abs/1910.07104
- P-Trojan, AAAI 2026 — https://arxiv.org/abs/2512.14741
  - Bounds the backdoor loss after clean fine-tuning by how well poisoned and clean gradients align, and maximises that alignment to persist.
  - This is the persistence half, stated explicitly.
- BadCLIP++ — https://arxiv.org/abs/2602.17168
  - Proves that clean fine-tuning gradients and backdoor gradients point the same way within a trust region, giving a non-increasing bound on attack success.
  - Links stealth and persistence through "gradient dilution".
- BAD-BOOM: basin width (a second-order effect) governs persistence.
- Zhang S&P 2024: orthogonality explains why defences fail.
- Lukas and Kerschbaum, "Pick your Poison" — https://arxiv.org/abs/2305.09671
  - An empirical trade-off between undetectability and robustness, driven by poisoning rate. This is the closest existing statement of a "duality", but it is not in gradient form.
- Sleeper Agents — https://arxiv.org/abs/2401.05566
  - Shows persistence through safety training but gives no gradient-coupling explanation. The training data there was filtered to be harmless when the trigger was absent.
- Counterpoint — https://arxiv.org/pdf/2604.00770
  - A latent-reasoning backdoor that linear probes detect almost perfectly yet survives clean fine-tuning. So activation-level detectability and persistence come apart.

What remains new:
- The claim that one quantity, g_hidden^T P g_clean (P = the fine-tuning optimiser's preconditioner), is both the detection statistic at release and the first-order erosion rate.
- The consequence: a backdoor that evades first-order clean-gradient detection must, to first order, also resist clean fine-tuning or safety training, and vice versa.

Points to address:
- P-Trojan-style backdoors are engineered to be co-directional with the clean gradient. The tug-of-war at equilibrium predicts anti-alignment. These should be reconciled explicitly, and the theory makes a falsifiable prediction about how such backdoors behave under (b).
- Second-order (flat-basin, BAD-BOOM) persistence falls outside the first-order duality.

---

**(e) Pareto-stationarity ⇒ gradients anti-parallel unless one vanishes, used for backdoor auditing**

**Verdict: the mathematics is known (trivial); using it for backdoor auditing appears novel (no hits).**

Closest work:
- MGDA theory:
  - Désidéri 2012; Sener and Koltun, NeurIPS 2018 [own knowledge]
  - Momma et al., ICML 2022 — https://proceedings.mlr.press/v162/momma22a.html
  - Survey — https://arxiv.org/html/2501.10945v3
- MGDA used by attackers:
  - Blind Backdoors, Bagdasaryan and Shmatikov, USENIX Security 2021: MGDA with Frank-Wolfe balances main and backdoor losses — https://arxiv.org/abs/2005.03823
  - Detector Collapse — https://arxiv.org/pdf/2404.11357
  - Meta-backdoors — https://arxiv.org/pdf/2107.10443
  - MGDA-trained backdoors are Pareto-stationary by construction, so they are exactly the case where the theory predicts anti-parallel gradients. They make a good test set.
- Multi-objective unlearning with Pareto-stationarity and conflicting forget/retain gradients — https://arxiv.org/pdf/2606.00399
- LADDER, NDSS 2025: multi-objective trigger optimisation, evolutionary rather than gradient-based — https://arxiv.org/pdf/2411.19075
- Haim et al.: KKT stationarity at trained weights used as an audit lever, for data rather than objectives — https://arxiv.org/abs/2206.07758
- Federated-learning MGDA backdoor defences: none found.

What remains new:
- Using Pareto-stationarity of the declared and hidden objectives at release as an inference principle for the defender: the gradient of the declared loss points toward the hidden objective.

Points to address:
- Weight decay or any regulariser acts as a third objective. With three gradients, zero lies in their convex hull without g_clean and g_hidden being anti-parallel.
- Stochastic MGDA may not converge to a Pareto-stationary point (NeurIPS 2022) — https://papers.nips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract-Conference.html
