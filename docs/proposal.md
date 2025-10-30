Topic: Can small neural networks follow longer, recombined instructions as well
as people?

I’ll build a tiny “commands to actions” toy world (e.g., “walk left twice and jump”) and teach a few small models to translate the text into action steps. I’ll then turn up the difficulty by giving longer instructions, new combinations the model never saw during training, and harmless extra words that it should ignore. I’ll also collect a small human baseline (friends/classmates answering multiple-choice) to see how people do on the same tasks. 

The goal is to find the breaking point
where people still do great but the models start to fail, and show this with clear graphs.
Success is defined by identifying an inflection point (sequence length or composition depth)
where human accuracy remains ≥90% while model accuracy falls ≤60%, plus evidence that
failures persist under ablations (removing shortcuts, adding irrelevant clauses). Deliverables will include a public GitHub repo with clean, runnable code; a data generator; evaluation scripts; figures (accuracy vs. difficulty, error heatmaps, calibration); and a short technical blog in the README summarizing results and discussing why current inductive biases/optimization fall
short and what breakthroughs (e.g., explicit compositional bottlenecks, curricula, or hybrid
neuro-symbolic steps) might close the gap. This project goes beyond a trivial demo by engaging
deeply with a concrete scientific question, implementing multiple models/ablations, and tying
empirical results to theory.

