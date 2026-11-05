Weight-space programming, meaning writing behaviour into a network's weights by hand instead of training it in, is getting a lot of attention. But edited networks usually get trained further. What happens to a hand-written edit then?

We ran a small, pre-registered experiment to find out.

Setup: 24 tiny transformers (~110k parameters, trained on toy copy/reverse/sort/shift tasks). We bolted on a hand-built patch for one rule: "whenever you're about to output symbol a, output z instead." The patch is two extra ReLU units, 134 numbers computed by formula and never trained.

We then kept training each network on data the patched model already got right, with the patch held fixed. In 30% of examples the rule applies. Every 25 steps we removed the patch and asked: has the network learned the rule itself?

We wired the same 134 numbers in two ways and got opposite outcomes:

🔹 Trigger reads a frozen copy of the original network. The patch is never wrong, so the network gets almost no error signal and barely learns the rule. Remove the patch and the behaviour goes with it: only 1 of 24 networks kept the rule.

🔹 Trigger reads the live network behind a stop-gradient. Training shifts what the trigger reads, so the patch starts missing cases, and those misses teach the network the rule. The patch becomes dead code. All 24 of 24 networks kept the rule after removal, within 50–100 training steps.

At step 0 the two versions give bit-for-bit identical outputs and gradients on the training data (by design). The contrast held under three optimiser settings.

A third wiring, with gradients flowing through the trigger, landed in between and was erratic: 16/24 absorbed the rule in the main setting, 7/24 at a lower learning rate, and reshuffling the training data flipped 8 of 24.

What this suggests (a toy result, so treat these as hypotheses for your setting):
1️⃣ If further training includes data showing the edited behaviour, removing the edit may no longer remove the behaviour. When the data never showed the rule, removal undid it in every model.
2️⃣ An edit that is never wrong can leave a network permanently dependent on it.
3️⃣ What an edit reads, and whether gradients reach it, decided the outcome. Checks at the moment of the edit couldn't tell the versions apart.

Caveats: tiny models; one add-on patch held fixed (not an edit to existing weights); one rule, whose trigger reads the very score it changes.

Pre-registration (committed before any test run), code and per-run results: [link]

#MachineLearning #Interpretability #AIAlignment
