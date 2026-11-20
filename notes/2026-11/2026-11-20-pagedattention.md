# PagedAttention: treating the KV cache like virtual memory

**Paper:** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang & Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) · code: [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Idea
Serving throughput is mostly a batching problem, and batch size is capped by KV cache memory. Older servers reserve one contiguous chunk per request sized for the maximum output length, so most of it sits empty or fragmented. PagedAttention splits each sequence's KV cache into fixed-size blocks (B tokens each) that can live anywhere in GPU memory. A per-sequence block table maps logical blocks to physical ones, just like a page table, and the attention kernel gathers keys and values block by block.

## Why it matters
Their profiling found that existing systems used only 20.4%–38.2% of KV cache memory for actual token states. With paging, waste is limited to the unfilled tail of each sequence's last block, so many more requests fit in a batch. vLLM reports 2–4× higher throughput than FasterTransformer and Orca at similar latency, with bigger gains for long sequences and fancier decoding.

## Concrete detail
Sharing comes almost for free. In parallel sampling, every sample's block table points at the same physical prompt blocks, and each physical block keeps a reference count. When one sample needs to write into a shared block, vLLM copies it first (copy-on-write at block granularity, the same trick as `fork()`). Beam search uses the same mechanism, so beams that branch and get pruned don't duplicate their common prefix.

## What I'd try
Run an agent that sends the same long system prompt plus tool schema on every call, and compare vLLM throughput with prefix caching on and off. That shared prefix is exactly the case block-level sharing was built for.
