Weight-space programming, meaning writing behaviour into a network's weights by hand instead of training it in, is getting a lot of attention. But edited models usually keep getting trained. What happens to a hand-written edit then?

We ran a series of small pre-registered experiments to find out.

The setup: we gave a model a hand-built patch for one rule, e.g. "wherever you'd say 'provide', say 'offer'". Then we kept training it on data the patched model already got right, held the patch fixed, and asked one question. If you remove the patch, does the model now follow the rule on its own?

We wired the same patch in two ways. At the moment of the edit, both give identical outputs and identical gradients on the training data (by design). They ended in opposite places.

🔹 The patch reads a frozen copy of the original model. The patch is never wrong, so the model gets almost no error signal and doesn't learn the rule. Remove the patch and the behaviour goes with it. It's a crutch: on a real language model (SmolLM2-135M), the same training without any patch taught the swap within ~75 steps (80% of cases). With the frozen patch in place, the model learned almost nothing (3%).

🔹 The patch reads the live model, with no gradient flowing through it. Training shifts what the patch reads, so it starts missing cases, and those misses teach the model the rule. Afterwards, removing the patch changes little:
• Small transformers: 24/24 kept the rule.
• Language model: 75% of cases, close to the no-patch model, in 8/8 word pairs.

In the small transformers this held even when only 1–3% of the further-training examples showed the rule, and when the patch's own numbers were also trained.

A third wiring, with gradients flowing through the patch, depended on the setting. In the small transformers it landed in between, and which models absorbed the rule changed with the data order. On the language model the model mostly kept the patch firing instead of learning the rule.

What this suggests (small models, so treat these as hypotheses to test in your own setting):
1️⃣ If further training includes even a little data showing an edit's behaviour, removing the edit may no longer remove the behaviour. When the data never showed it, removal worked every time.
2️⃣ An edit that is never wrong can leave a model dependent on it, even for a behaviour it would otherwise learn in a few dozen steps.
3️⃣ What an edit reads, and whether gradients reach it, decided the outcome. Checks at the moment of the edit couldn't tell the versions apart.

Caveats:
• The models are small: 24 toy transformers of ~110k parameters, plus SmolLM2-135M with only its top 4 blocks trained.
• The patch was an add-on held fixed, not an edit to existing weights, and each test used one training recipe.
• On the language model the handover was visible: rule accuracy with the patch in place dipped to ~70–80% along the way.

Pre-registrations (each committed before its runs, including one we withdrew after finding a bug), code and per-run results: [link]

#MachineLearning #Interpretability #AIAlignment
