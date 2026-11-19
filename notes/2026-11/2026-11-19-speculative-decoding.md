# Speculative decoding: let a small model guess, let the big model check

**Paper:** Leviathan, Kalman & Matias, 2023, "Fast Inference from Transformers via Speculative Decoding" (Google Research, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192))

## Idea
Decoding K tokens normally takes K serial forward passes of the large model. Instead, a cheap draft model proposes γ tokens, and the target model scores all of them in one parallel pass. Accepted guesses are kept; the first rejected one is replaced by a token sampled from a corrected distribution. Each target pass therefore yields at least one token and at most γ+1.

## Why it matters
The output distribution is *exactly* the target model's: no retraining, no architecture change, no quality trade-off. On T5-XXL the paper reports a 2X–3X speedup over the standard T5X implementation with identical outputs. That makes it one of the few inference tricks you can turn on without re-running evals.

## Concrete detail
The acceptance rule is a form of rejection sampling. Draw x ~ q (draft). Keep it if q(x) ≤ p(x); otherwise reject with probability 1 − p(x)/q(x) and resample from norm(max(0, p − q)).

The per-token acceptance probability works out to β = Σ min(p(x), q(x)), i.e. one minus a total-variation-style divergence. Assuming i.i.d. acceptances with mean α, the expected tokens per target call is

(1 − α^(γ+1)) / (1 − α)

So with α = 0.8 and γ = 5 you get about 3.7 tokens per expensive pass. Past a point, raising γ mostly buys wasted draft compute.

## What I'd try
Measure α empirically for a small/large pair from the same family on agent-style prompts (JSON tool calls, long boilerplate), where I'd expect the draft to agree much more often than on free-form prose, then pick γ from the formula instead of guessing.
