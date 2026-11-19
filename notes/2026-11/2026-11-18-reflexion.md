# Reflexion: learning from failures without touching the weights

**Paper:** Shinn et al., 2023, "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) · code: [noahshinn/reflexion](https://github.com/noahshinn/reflexion)

## Idea
Instead of fine-tuning an agent on its mistakes, let it write about them. After a failed trial, the agent gets a feedback signal (a binary success flag, a heuristic, or failing test output), turns it into a short natural-language reflection on what went wrong, and stores that in an episodic memory. The next attempt is conditioned on those reflections. It's "RL" only in the loose sense: the policy update is text in the prompt.

## Why it matters
It's a cheap, model-agnostic way to get trial-and-error improvement out of a frozen LLM, and it slots on top of ReAct-style agents. The catch is that it needs a decent evaluator; reflection is only as good as the signal it reflects on.

## Concrete detail
On HumanEval (Python) it reaches 91% pass@1 versus 80% for GPT-4 alone. For coding, the agent writes its own unit tests and uses their results as feedback. On AlfWorld, ReAct + Reflexion improves by 22 points absolute over 12 trials, and on HotPotQA by about 20%. Memory is kept to a small window of the most recent reflections so the prompt doesn't blow up.

## What I'd try
Add a reflection step to a small tool-using agent where I already have a cheap checker (tests or a schema validator), and measure how many retries it takes before gains flatten out.
