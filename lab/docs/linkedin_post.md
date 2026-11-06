Weight-space programming, meaning writing behaviour into a network's weights by hand instead of training it in, is getting a lot of attention. But edited networks usually keep getting trained. What happens to a hand-written edit then?

We ran a small, pre-registered experiment to find out.

Setup: 24 tiny transformers (~110k parameters, toy copy/reverse/sort/shift tasks). We bolted on a hand-built patch for one rule: "whenever you're about to output symbol a, output z instead." The patch is two extra ReLU units, 134 numbers computed by formula and never trained.

We then kept training each network on data the patched model already got right, with the patch held fixed. Every 25 steps we removed the patch and asked: has the network learned the rule itself?

We wired the same 134 numbers in two ways and got opposite outcomes:

🔹 The patch reads a frozen copy of the original network. The patch is never wrong, so the network gets almost no error signal and barely learns the rule. Remove the patch and the behaviour goes with it. 1/24 networks kept the rule.

🔹 The patch reads the live network behind a stop-gradient. Training shifts what the patch reads, so it starts missing cases. Those misses teach the network the rule, and the patch becomes redundant. 24/24 networks kept the rule after removal.

At the moment of the edit, both versions give bit-for-bit identical outputs and gradients on the training data (by design).

The contrast held when only 3% of the training examples showed the rule (24/24 vs 0/24). It still held at 1% (21/24 vs 0/24), when the patch's own numbers were also trained, and under three optimiser settings.

A third wiring, with gradients flowing through the patch, landed in between and behaved like a coin toss: about 60% of networks absorbed the rule, but which ones changed with the order of the training data.

What this suggests (a toy result, so treat these as hypotheses to test in your own setting):
1️⃣ If further training includes even a little data showing an edit's behaviour, removing the edit may no longer remove the behaviour. When the data never showed it, removal worked every time.
2️⃣ An edit that is never wrong can leave a network permanently dependent on it.
3️⃣ What an edit reads, and whether gradients reach it, decided the outcome. Checks at the moment of the edit couldn't tell the versions apart.

Caveats: tiny models; one add-on patch held fixed (not an edit to existing weights); one rule, whose trigger reads the very score it changes.

Pre-registrations (each committed before its runs), code and per-run results: [link]

#MachineLearning #Interpretability #AIAlignment
