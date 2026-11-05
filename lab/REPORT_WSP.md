# What ordinary training does to a hand-written weight patch

**Pre-registered (PREREGISTRATION.md addendum e), 24 sealed toy transformers, all registered hypotheses supported.**
We wrote the same 134-number rule into each network's weights in three ways, which give bit-for-bit identical
outputs and gradients at the moment of the edit. Under further training on data the patched network already gets
right, they ended in three different places:
- **frozen reference** (the patch reads a frozen snapshot of the network): the network never learned the rule, so
  deleting the patch still removed the behaviour. The rule survived deletion in 1/24 models.
- **live, no gradient** (the patch reads the live network, outside the gradient path): the network absorbed the rule
  within about 75 training steps, after which deleting the patch changed nothing (24/24).
- **live** (the patch reads the live network, inside the gradient path): partial absorption (16/24). Post hoc, the
  network kept the patch's trigger firing, and even made it fire on clean inputs.

Visible behaviour hid all of this: with the patch in place, rule accuracy stayed near 1 in every arm.

![headline](results/fig/wsp_fates_sealed.png)

## 1. Question
"Weight-space programming" means writing behaviour directly into a network's weights instead of learning it from data.
The term was popularised in November 2026 by reports of frontier language models that read and edit the raw weights of
small networks (see e.g. arXiv 2611.05341, which uses such edits to audit and repair small clinical networks). Earlier
forms include compiled transformers (Tracr), rank-one edits (ROME) and task arithmetic. An edited network is rarely
the end of the story: it is usually trained further. Prior work asks whether later fine-tuning *erases* an edit
(e.g. arXiv 2507.14198, 2511.05852). We ask the complementary question. If further training uses data the edit
already handles, does the edit stay load-bearing, or does the network absorb it, so that deleting the edit no longer
undoes the behaviour? And does the answer depend on how the edit is wired in?

## 2. Setup (`wsp/patch.py`)
- **Base networks.** Clean toy organisms (pre-LN GPT, d = 64, 2 layers, 4 heads, about 100k parameters) trained on
  copy, reverse, sort and shift over 54 symbols (`organisms/`). The 24 sealed bases were drawn at random from 120
  eligible ones (`runs_wsp/sealed_bases.json`); 4 other bases were used only for piloting.
- **The rule.** A benign remap: wherever the network is about to emit symbol a, emit z instead (a, z drawn per base).
- **The patch (134 numbers, written by hand, never trained).** Two ReLU units after the final LayerNorm read
  t = W_U[a]·lnf(h) (the network's own drive towards a). The second unit is the first shifted by exactly 1, so
  s = relu(k(t − τ)) − relu(k(t − τ) − 1) ∈ [0, 1]. The patch adds γ·s to logit z and subtracts it from logit a. τ and
  k are set from the gap between rule and non-rule positions; γ sets the force (four levels, chosen so that the
  patched model's mean 1 − p(z) at rule positions is 3e-2, 1e-2, 1e-3 or 1e-4).
- **Arms (same numbers).** *live*: the trigger reads the network being trained; *detached*: it reads the same live
  activations but no gradient flows through it; *frozen*: it reads a frozen copy of the install-time network, so it
  can never switch off; *twin*: no patch.
- **Maintenance training.** Batches of 128 sequences, 30% containing the rule, sampled from pools filtered so that the
  patched model is right at every trained position and every trigger pre-activation is strictly saturated or off.
  Adam with warm-started moments, 3× the base's final training learning rate, 2000 steps. Every arm and level of a
  base sees the same data stream.
- **Measures.** *Undo test B*: rule accuracy of the network with the patch deleted, on 512 held-out rule sequences
  balanced over tasks. *F*: the same with the patch in place. Pre-registered tests are at 2000 steps.

## 3. Pre-registered results (`results/wsp_sealed_prereg.txt`)
| | Hypothesis | Result |
|---|---|---|
| W0 | Gate: the three arms are identical at step 0 | **Pass.** 0 trained positions in the trigger's linear range; max logit and gradient differences exactly 0 (24 bases × 4 levels). Post hoc, the in-batch losses stay bitwise identical through all 100 warm-up steps in 96/96 cells. |
| W1 | Detached absorbed, frozen shielded | **Supported.** B_detached median 1.00 at every level vs B_frozen 0.21, 0.03, 0.01, 0.00; detached > frozen in 24/24 at every level (p ≈ 9e-6 each) |
| W2 | Gradient through the trigger protects the patch | **Supported.** B_live median 0.47, 0.75, 0.75, 0.90 vs 1.00 detached; detached > live in 17-22/24 (p ≤ 1.4e-4) |
| W3 | Live absorbed more than frozen | **Supported** at 3 of 4 levels (level 0: 15/24, Holm p = 0.011; levels 1-3: 19-21/24, p ≤ 1e-4) |
| W4 | Frozen arm: more force, less learning | **Supported.** Per-base Spearman(level, gain in z margin) = −1 in median; negative in 24/24 (p = 1.5e-6) |
| W5 | Training that never shows the rule breaks in-network patches | **Supported** 4/4. F at 2000 steps: live 0.90 / 0.93, detached 0.85 / 0.94, frozen 0.97 / 1.00 (levels 1 / 3); frozen higher in 21-24/24 (p ≤ 2e-5) |
| W6 | Generality (level 1) | **Supported.** SGD-momentum: detached 1.00 vs frozen 0.00 (11/12, p = 5e-4). Adam at 1× final lr: 1.00 vs 0.00 (24/24, p = 8e-6) |

The primary family W1-W4 is Holm-corrected; all four have family-adjusted p ≤ 2.5e-4.

## 4. What else the runs show (exploratory or post hoc; `results/wsp_posthoc.txt`)
- **Speed.** Without a patch, the networks learn the rule in 120-130 steps. Detached patches were absorbed by step
  162-175, i.e. about 75 steps after the learning rate became non-zero (it is 0 for 100 steps, then warms up).
- **How the live patch survives (post hoc).** In the first 400 steps, the live trigger kept firing at 0.89-0.99 of
  rule positions, against 0.16-0.21 for the detached trigger. The live trigger also began firing on clean inputs (a
  median of 21-48 misfires in the first 400 steps; 0 in the other arms). With gradient access to the trigger, the
  network pushes the trigger's input up rather than learning the rule itself. Live runs with more early misfires
  were absorbed less (Spearman −0.48 across 96 runs).
- **The live arm depends on the learning rate and is chaotic.** At 1× the final learning rate, live absorption
  dropped to 7/24 (median B 0.07), while detached stayed 24/24 and frozen 0/24. Under SGD it was 6/12. Rerunning
  level 1 on a second data stream changed B_live by a median of 0.24, and by up to 1.0; detached changed by 0.00 and
  frozen by 0.03. Whether a given live patch is absorbed is close to a coin flip; that live patches are absorbed less
  than detached ones and more than frozen ones is not.
- **Force.** In the live arm, more force went with faster absorption (per-base Spearman(level, takeover time) median
  −0.40, two-sided p = 0.04, exploratory). In the frozen arm, more force meant less learning (W4).
- **Tasks.** In partly absorbed live runs, the shift task (which needs the conjunction "task = shift and the symbol
  before a in the prompt") lagged the other three.
- **Breakage without the rule (W5).** When training never showed the rule, detached and live patches drifted to rule
  accuracy as low as 0.11-0.65 in some models. No training example tells the network to keep the trigger's input
  where it was.

## 5. Interpretation
- **A patch that is never wrong teaches nothing.** The frozen patch is right at every training position, so the rule
  produces almost no error signal (gradient starvation). The more force, the less signal (W4). The network stays
  dependent on the patch.
- **A patch that reads the network's own activations can become wrong.** Ordinary training drifts those activations.
  Here, a is never a training target, so the network's drive towards a falls. Once a few rule cases are missed, the
  errors teach the network the rule, which lowers the drive towards a further: the trigger is switched off
  everywhere, and the patch becomes dead code. Without a gradient path, nothing resists this; it completes in about 75
  steps.
- **A gradient path through the trigger fights back.** Each missed rule case also sends gradient into the trigger's
  input, and the network restores it rather than learning the rule. The outcome then depends on which of the two
  routes wins, which is why it is variable and sensitive to the learning rate.
- **Implications for weight-space programming:** (1) after further training, reverting an edit may not revert the
  behaviour; (2) whether an edit stays load-bearing depends on how it is wired into training (what it reads, and
  whether gradients reach it), not only on what it computes; (3) analysis at the time of the edit cannot tell these
  cases apart: outputs and gradients are identical; (4) training that never exercises an edit can still silently
  break it (W5).

## 6. Limitations
- Toy networks (about 100k parameters) and toy tasks; no pretrained language model (Hugging Face is not reachable from
  this environment).
- One kind of edit: a logit-level remap whose trigger reads the very logit it acts on. Patches that read earlier
  features, or that are themselves trainable, were not tested in the sealed run.
- The remap removes a from every training target, which suppresses a; this drift seeds the detached and live
  trigger failures. A null-symbol control (removing an unrelated symbol) was not run.
- One training recipe family (Adam at 1× and 3× the final learning rate, SGD-momentum at one learning rate), a
  2000-step horizon and a 30% rule share.
- The live arm's outcome varies strongly between data streams (§4); its medians should be read with that in mind.

## 7. Provenance
- Design: a literature and design workflow (6 scouts, 12 proposals, 3 judges, novelty and failure-mode skeptics), then
  a pilot (`runs_wsp/stage1.jsonl`), a pre-registration code review (`docs/review_wsp_pipeline.md`, 24 confirmed
  findings, all addressed) and a second pilot (`runs_wsp/pilot2.jsonl`).
- Pre-registration: addendum e, commit `b0b5161`, before any sealed base was run. The sealed runs used code at
  `218c931` and, after three container restarts, `573d7ae`; `git diff 218c931 573d7ae -- wsp organisms detect` is
  empty. The runner resumes by config hash, so no run was repeated or replaced.
- Data: `results/wsp_sealed_summary.jsonl.gz` (636 runs, the fields the analysis uses) reproduces
  `results/wsp_sealed_prereg.txt` exactly: `python analysis/patch_fate.py results/wsp_sealed_summary.jsonl.gz --h0
  results/wsp_sealed_h0.jsonl --prereg`. Figures: `python analysis/wsp_figures.py results/wsp_sealed_summary.jsonl.gz
  results/fig`.
