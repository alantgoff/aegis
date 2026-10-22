# Running findings log (development zoo only, not yet confirmatory)

## Setup
- ToyLang: 4 spec tasks (copy, rev, sort, shift) over 54 symbols; sequence = [BOS, TASK, x, SEP, y, EOS].
- Organisms: pre-LN GPT, d in {48,64,96}, L in {2,3}, AdamW, cosine LR to 5%, 2.5k-3.5k steps.
- Backdoors: trigger (ngram, skipgram, posfix, sorted, subset; natural rate < 1e-4, filtered from
  clean data) -> behaviour (payload, taskswap, inject, refuse); poison 1-4%.
- Benign controls: clean; clean + undisclosed benign extra task ("extra").
- Auditor: weights + spec sampler only (no trigger, behaviour, training data, base model, optimiser state).

## Negative results (pilot)
1. Naive KKT / weight-decay stationarity residual (cos(grad, -w)) does not separate clean from backdoored
   (~0.03 for both): the clean gradient at the final weights is dominated by a systematic non-zero
   component (Adam oscillation in sharp directions), and short training never decays init remnants.
2. Dormant-neuron / "unjustified write" statistics fail: backdoor neurons are active on clean data.
3. The ground-truth backdoor direction (mean residual diff triggered - clean) is a HIGH-variance direction
   on clean data (89th-98th percentile), so "dormant direction" / generalised-eigen ideas fail.
4. Noise injection does not elicit backdoor behaviour beyond generic degradation.
5. LLC (SGLD) estimates are unstable and flip sign with hyper-parameters.

## Positive result: the tug-of-war
- D_c := mean clean-data gradient at the released weights, normalised by the clean second moment.
  D_c is reproducible across independent clean samples (median cos 0.85) -> systematic, not noise.
- For backdoored models, D_c is anti-aligned with the hidden (poison) gradient D_h:
  dev (13 backdoored / 15 benign): 13/13 negative (sign test p=1.2e-4), median cos -0.23 vs
  benign-with-fake-hidden-objective median -0.02 (Mann-Whitney p=1.4e-5, AUROC 0.885).
  Holds with wd=0 -> not a weight-decay effect: it is the equilibrium of the mixture objective
  ((1-p) g_clean + p g_hidden ~ 0). Weak for "subset" triggers (poison inputs far from clean inputs ->
  little interference).
- The tug lives in every layer group (embeddings, attention, MLP of every block; -0.2..-0.6), not just
  the unembedding. For taskswap backdoors the head shows ~0 tug while early attention/embeddings show
  -0.5..-0.6: a black-box calibration test would miss these.
- Consequences (pilot, 3-4 models): one normalised step of clean DESCENT erases the backdoor
  (ASR 0.84 -> 0.05); clean ASCENT amplifies it and leaks it onto clean prompts (e.g. refusal NLL on
  clean prompts 13.1 -> 0.8); random directions of equal norm do nothing.
- Per-example: gradient alignment with -D_c separates poison from clean examples (median AUROC 0.89).

## Trigger-free detector (tug_lib)
- Library of generic deviations on ordinary clean prompts: token deviations (emit v at answer offset j)
  and pointer deviations (emit prompt symbol x_i (+1) at offset j). Score = max -cos(D_c, D_dev).
- Dev subset (10 v 10): AUROC 0.88; arg-max cell exactly recovers the hidden behaviour in 6/8
  single-token behaviours (payload first token, inject symbol+position, refusal=EOS@0) and the
  taskswap->rev behaviour (pointer x_{-4} at offset 3).
- Controls: random direction with the same coordinate magnitudes gives ~0.007 (vs 0.04-0.3).

## Open threats to validity (to test)
- Clean finishing (safety-training analogue) and a white-box gradient-penalty attacker.
- Constant LR (no annealing) increases oscillation noise -> predicted weaker signal.
- Specificity vs benign undisclosed tasks; fine-tuned insertion; held-out trigger/behaviour kinds.
