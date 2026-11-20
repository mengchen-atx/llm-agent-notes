# StreamingLLM: why the first four tokens matter

**Paper:** Xiao, Tian, Chen, Han & Lewis, "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)) · code: [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)

## Idea
Plain window attention (keep only the last *L* KV entries) falls apart as soon as the first tokens get evicted, even though those tokens usually carry no meaning. The authors explain it as an *attention sink*: softmax forces each query's attention weights to sum to one, so the model learns to dump the attention it doesn't need on the initial tokens, which every later position can see. Evict them and the attention distribution shifts badly. StreamingLLM simply pins the KV of a few initial tokens (four is enough) next to a rolling window of recent tokens.

## Why it matters
No fine-tuning is needed. Llama-2, MPT, Falcon and Pythia stay stable over 4M+ tokens, and it runs up to 22.2× faster than sliding-window recomputation. One caveat the README is explicit about: this does **not** extend the context window. Tokens in the middle are dropped, so it's for long-running chat or agent sessions, not for long-document QA.

## Concrete detail
Positional encoding uses positions *within the cache*, not the original text. With RoPE they cache keys before rotation and re-apply rotation on each step, so a sink token at index 0 and recent tokens at 4…L+3 always look contiguous. They also show that pre-training with one dedicated learnable "sink" token is enough to replace the four-token heuristic.

## What I'd try
Run a long tool-calling agent loop on a small Llama model with `StartRecentKVCache` and watch where it starts forgetting earlier tool results. That would show how much a summary or memory layer is still needed on top.
