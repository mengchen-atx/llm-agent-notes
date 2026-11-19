# Toolformer: the model labels its own tool calls

**Paper:** Schick et al., 2023, "Toolformer: Language Models Can Teach Themselves to Use Tools" (Meta AI Research, [arXiv:2302.04761](https://arxiv.org/abs/2302.04761))

## Idea
Rather than hand-labeling when a model should call a tool, let the model propose API calls inside ordinary pretraining text, run them, and keep only the calls that actually help it predict what comes next. Then fine-tune on the text with those calls spliced in. Each tool only needs a handful of demonstrations to get the sampling started.

## Why it matters
It's the self-supervised take on tool use: no human preference data and no task-specific supervision. The usefulness signal comes straight from the LM loss, so the model learns *when* a calculator or search engine is worth calling, not just *how* to format the call.

## Concrete detail
The filter compares two weighted losses over the tokens after position *i*:

- L⁺ = loss when the call **and its result** are given as a prefix
- L⁻ = min(loss with no call, loss with the call but an empty result)

A call is kept only if L⁻ − L⁺ ≥ τ_f. Taking the min in L⁻ matters: it throws away calls whose "benefit" comes only from the query text, not from what the tool returned.

The base model is GPT-J (6.7B), fine-tuned on a CCNet subset annotated this way. Tools include a calculator, a Q&A system, a Wikipedia search engine, a machine translation system and a calendar. In zero-shot evaluation it often beats GPT-3 (175B), and its perplexity on language modeling barely changes.

## What I'd try
Reuse the L⁻ − L⁺ filter as a data-cleaning step for agent trajectories: score each tool call in a ReAct trace by whether its observation lowers the loss on the final answer, and drop the calls that don't.
