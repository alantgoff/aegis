# Theory: the tug-of-war and the detectability-persistence duality

Notation. Parameters θ. Declared (clean) loss L_c, hidden loss L_h, regulariser R (e.g. λ/2‖θ‖²).
Training objective L_mix = (1-p) L_c + p L_h + R. A diagonal preconditioner P ≻ 0 (for Adam,
P ≈ diag(1/√v)). The auditor can evaluate g_c = ∇L_c (by sampling the declared distribution), never g_h.
All inner products ⟨a,b⟩_P = aᵀPb; "cos_P" is the cosine of Pa and Pb (what the code computes).

## Proposition 1 (tug identity at a preconditioned equilibrium)
If the released weights θ* are a (time-averaged) stationary point of preconditioned training with
decoupled weight decay λ (AdamW, as used here), E[P g_mix] + λθ* = 0, then

    P g_c = -(p / (1-p)) P g_h - (λ / (1-p)) θ*.

(For coupled L2 the last term is -(1/(1-p)) P∇R.) With λ = 0: P g_c and P g_h are exactly anti-parallel.
The observed magnitudes (median cos ≈ -0.23 on dev) imply a fluctuation-dominated regime, i.e. the
released iterate is far from the exact equilibrium and the identity holds only as a tendency. With the optimiser fluctuation N at the
released iterate (θ_T = θ* + δ, N ≈ P H_c δ), the measured cosine is

    cos_P(g_c, g_h) ≈ -1 / sqrt(1 + (‖N‖ / (κ‖P g_h‖))²),   κ = p/(1-p),

assuming N ⟂ P g_h. Predictions: (i) the tug strengthens as the final learning rate falls
(smaller δ) and as interference between clean and hidden data grows (larger residual hidden
gradient at equilibrium); (ii) it vanishes when the hidden data are separable from clean data
(both residual gradients → 0), e.g. 'subset' triggers whose inputs lie off the clean manifold.

## Proposition 2 (Pareto stationarity)
At a Pareto-stationary point of (L_c, L_h) there is α ∈ [0,1] with α g_c + (1-α) g_h = 0, so either one
gradient vanishes or the two are anti-parallel. A third force (regulariser, or a hidden objective that
is partly spec-consistent and therefore partly *shares* L_c's targets) relaxes this: g_c and g_h may then
be co-directional when both oppose the third force. Any procedure that ends at a joint optimum of the
mixture – including MGDA-style attacks that explicitly balance main and backdoor losses – is in the
anti-parallel regime by construction.

## Proposition 3 (detectability-persistence duality)
Under preconditioned clean fine-tuning (the analogue of safety training) with fine-tuning
preconditioner P_ft and decoupled decay λ_ft, dθ/dt = -P_ft g_c - λ_ft θ, so by the chain rule

    dL_h/dt = -g_hᵀ P_ft g_c - λ_ft g_hᵀ θ.

This is a first-order (local) statement. The detector does not compute this rate: it uses cos(P g_c, P g_d)
with library proxies g_d in place of the unknown g_h. Both quantities are empirically rank-correlated with
ASR half-life (sealed zoo2: rate -0.78, trigger-free score -0.70). Consequences (first order; second-order and basin effects are outside the statement):
1. A backdoor with a strong anti-aligned tug erodes quickly under clean fine-tuning.
2. A backdoor that does not erode to first order must have ⟨g_h, g_c⟩_P ≥ 0. Combined with
   Proposition 1 (R = 0) this forces g_c = 0: the backdoor sits at an exact clean critical point
   (clean-degenerate directions), invisible to any first-order clean-gradient audit.
3. With a third force, ⟨g_h, g_c⟩_P > 0 is possible: clean fine-tuning then initially *strengthens* the
   hidden behaviour. Empirically this raises survival but does not guarantee it (sealed zoo2: 7/16 censored
   with rate < 0 vs 6/44 with rate > 0, Fisher p = 0.02; an attacker who forces positive alignment gains no
   persistence: 0/8), because the local alignment does not persist along the fine-tuning trajectory.

Trilemma: at release, a hidden behaviour is (a) anti-aligned → detectable and fragile;
(b) co-directional → persistent, sign-flipped tug; or (c) orthogonal / clean-critical → persistent and
first-order invisible (needs second-order or behavioural methods).

## Reconciling 'orthogonal during training' (Zhang et al., S&P 2024)
During training, clean and backdoor gradients are large and nearly orthogonal (they are learned in
different subspaces). At equilibrium the *residual* gradients are small and, by Proposition 1, forced
into anti-alignment. The two statements concern different regimes and are compatible.

## Why a generic deviation library works
g_h is the gradient of NLL of deviant outputs on triggered prompts. Its components that do not depend on
the trigger – the unembedding/readout of the deviant token at its answer offset, the attention and MLP
machinery that writes answer tokens at that offset – are shared with the gradient of the *same deviation
on ordinary prompts*. The library element d_jv = E_clean[∇(-log p(v at offset j))] therefore overlaps
g_h, and cos_P(g_c, d_jv) inherits the tug. Empirically the tug is present in every parameter group
(embeddings, all attention and MLP blocks; -0.2..-0.6), not only in the output layer, and for
content-dependent behaviours (task swaps) it is absent from the head and concentrated in early
attention/embeddings.

## Proposition 4 (leakage identity) — why first-order audits are black-box leakage in disguise
For the cross-entropy loss ℓ(x) = -log p_y(x) with softmax outputs, using Σ_v p_v = 1:

    ∇ℓ(x) = Σ_v p_v ∇logit_v - ∇logit_y = Σ_{v≠y} p_v(x) · ∇(logit_v - logit_y)(x),

so the clean-data gradient is exactly a *leak-weighted* superposition of per-example deviation directions:

    g_c = E_x Σ_{v≠y} p_v(x) · ∇(logit_v - logit_y)(x).

A hidden objective enters g_c only through the probability it places on non-declared outputs (its leakage)
on declared inputs, weighted by the corresponding logit-difference Jacobians. The library element
d_jv ≈ E_x ∇(logit_y - logit_v) is exactly the deviation direction, so cos(g_c, d_jv) asks which deviation
dominates the leak-weighted sum: a Jacobian-weighted black-box leakage readout. Clean fine-tuning follows the
same g_c, so by Prop. 3 it, too, acts on a hidden behaviour only through that behaviour's leakage.
Corollary (shared blind spot): a hidden behaviour that places (near-)zero probability on its outputs over the
declared distribution is invisible to every detector computed from clean-data first-order information and is
not eroded (to first order) by clean fine-tuning. Detectability, leakage and fragility are one quantity seen
three ways; the persistent-backdoor problem is the non-leaky-backdoor problem.

### Extension to second order
The clean-loss Gauss-Newton curvature is E_x[J_xᵀ (diag(p_x) - p_x p_xᵀ) J_x]. For a confident model
(p_y ≈ 1) the bracket is ≈ Σ_{v≠y} p_v (e_v - e_y)(e_v - e_y)ᵀ to leading order, so the curvature too is a
leak-weighted sum of deviation-direction outer products. Anything computed from clean-data losses (gradients,
Gauss-Newton/Fisher curvature, influence functions) therefore sees a hidden behaviour only through the
probability it places on its outputs over the declared distribution. Detecting non-leaky hidden behaviours
requires inputs off the declared distribution (red-teaming, fuzzing) or weight-level mechanistic analysis.
