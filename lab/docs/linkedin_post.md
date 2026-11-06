Weight-space programming, writing behaviour directly into a network's weights instead of training it in, was just reported as an ability of frontier language models (arXiv 2611.05341). But edited models usually keep getting trained. What happens to a hand-written edit then?

We tested a simple stand-in in a series of small, pre-registered experiments.

The setup: we bolted a small hand-built module onto a model to implement one rule:
• 24 toy transformers: "emit z wherever you'd emit a".
• A small language model, SmolLM2-135M: word swaps such as "wherever you'd say 'provide', say 'offer'".

We kept training the model on data the patched model already got right, with the module held fixed. Then we removed the module and asked: does the model now follow the rule on its own?

We wired the same module in different ways. At the moment of the edit they give the same outputs and gradients on the training data. That's by design: we filtered the data so this holds. They ended in very different places.

🔹 The module reads a frozen copy of the original model. It is never wrong, so the model gets almost no error signal and barely learns the rule. Remove the module and the behaviour goes with it. It's a crutch:
• Language model: the same training without any module taught the swap (80% of cases by the end). With the frozen module in place, the model learned almost nothing (3%).
• Toy transformers: 1/24 kept the rule.

🔹 The module reads the live model, but no gradient flows through it. We think training shifts what the module reads, so it starts missing cases, and those misses teach the model the rule. Remove the module afterwards and the rule mostly stays:
• Toy transformers: 24/24.
• Language model: a median 75% of cases, close to the no-module model, above the frozen wiring in 8/8 word pairs.
In the toy models this held even when only 3% of the further-training examples showed the rule, and when the module's own numbers were also trained.

🔹 A third wiring is closest to just writing the module into the network: it reads the live model, and gradients flow through it. It was erratic.
• Toy transformers: 16/24 absorbed the rule with our main recipe, 7/24 at a lower learning rate. Which ones changed with the data order.
• Language model: mostly no absorption (6 of 8 pairs below 20%). Yet rule accuracy with the module in place still slipped, to 87% vs 99% for the frozen wiring.

What this suggests (small models, so treat these as hypotheses for your own setting):
1️⃣ If further training includes data showing an edit's behaviour, removing the edit may no longer remove the behaviour. When the data never showed it, removal worked every time in our runs.
2️⃣ An edit that is never wrong can leave a model dependent on it, even for a behaviour it would otherwise pick up quickly.
3️⃣ What an edit reads, and whether gradients reach it, decided the outcome. Output and gradient checks at the moment of the edit could not tell the wirings apart (we built them to match), so inspect the wiring itself.

Caveats:
• The models are small, and on the language model only the top 4 of 30 blocks trained, with one aggressive recipe.
• The module was an add-on, not an edit to existing weights.
• On the language model the handover was visible: with the module in place, rule accuracy dipped to a median 66–76% along the way.

Pre-registrations (including one we withdrew mid-run, before analysing it, after a review found its module didn't implement the registered rule), code and per-run results: [link]

#MachineLearning #Interpretability #AIAlignment
