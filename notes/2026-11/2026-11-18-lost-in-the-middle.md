# Lost in the Middle: where you put retrieved passages matters

**Paper:** Liu et al., 2023, "Lost in the Middle: How Language Models Use Long Contexts" ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172))

## Idea
A controlled study of how well models actually use a long prompt. On multi-document QA (one answer-bearing passage among 10, 20, or 30 distractors) and a synthetic key-value lookup task, the authors move the relevant item around and measure accuracy. The result is a U-shaped curve: models do best when the relevant passage is at the very start or very end of the context, and noticeably worse when it sits in the middle. Extended-context variants of the same models didn't do better at this.

## Why it matters
Most RAG pipelines just concatenate the top-k chunks in retriever order. If the right chunk lands in position 8 of 20, the reader may effectively miss it, so raising k doesn't help as much as you'd expect. The paper's open-domain QA case study makes the same point: reader accuracy saturates well before retriever recall does.

## Concrete detail
GPT-3.5-Turbo gets 56.1% closed-book (no documents at all). In the 20- and 30-document settings, its worst-case accuracy with the answer buried mid-context drops below that, so the retrieved context is actively hurting. Placing the query both before and after the documents ("query-aware contextualization") fixes the key-value task almost completely but barely changes the multi-document QA trend.

## What I'd try
Rerank, then reorder so the strongest chunks go first and last and the weakest go in the middle. Compare that against plain top-k order and against just using a smaller k, on the same eval set.
