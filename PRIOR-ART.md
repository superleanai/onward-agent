# Prior art

## The model manages its own context

- [Focus: Active Context Compression](https://arxiv.org/abs/2601.07190) (2026). The agent decides when
  to fold learnings into a persistent "Knowledge" block and drop raw history; reports that capable
  models self-regulate context with the right tools and prompting. Closest to the predecessor
  self-context loop's summarise-then-cut; this variant summarizes without cutting. Single-author preprint.
- [Scaling Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967) (ByteDance Seed,
  CMU, Stanford, 2025; [project page](https://context-folding.github.io/)). Actions *branch* into a
  sub-trajectory and *return* it folded into a short summary; trained with RL (FoldGRPO); SWE and deep
  research tasks. The predecessor's compaction was an untrained, single-level form of this.
- [Self-GC: Self-Governing Context for Long-Horizon LLM Agents](https://arxiv.org/abs/2607.00692)
  (Xiaohongshu, 2026; [summary](https://www.deeplearning.ai/the-batch/llms-take-out-the-agents-trash)).
  Context items are addressable objects with fold, mask and prune operations, decided by a separate
  planner model rather than the acting model.
- [Beyond Compaction: Structured Context Eviction](https://arxiv.org/abs/2606.11213) (2026). The agent
  tags its trajectory as typed episodes; a deterministic policy evicts them when over budget.
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (2023), now
  [Letta](https://docs.letta.com/letta_memgpt). The LLM moves information between its context and
  external storage with memory tools. Conceptual ancestor, with external memory instead of prefix
  truncation.

## Going back: rewind and backtracking

- [Langroid `rewind_tool`](https://langroid.github.io/langroid/reference/agent/tools/rewind_tool).
  The LLM backtracks to its nth previous message and replaces it. Nearest equivalent of `max_index`,
  as an occasional tool rather than a per-turn field.
- [AgentRewind](https://www.alphaxiv.org/abs/2608.14380) (2026). The agent calls a rewind tool when
  stuck; reports that keeping failed attempts in context lowers success. This variant deliberately
  retains failed attempts and summarizes their lessons instead of rewinding.
- [Google ADK rewind](https://adk.iqai.com/docs/framework/runtime/rewind) and Claude Code's
  `/rewind`: checkpoint rollback triggered by the runtime or the user, not chosen by the model.

## Caching awareness

- [Don't Break the Cache: An Evaluation of Prompt Caching](https://arxiv.org/abs/2601.06007) (2026).
  Prompt caching cuts agent API cost by 45-80%; changing tool results can cause cache writes that are
  never reused. Supports the stable-prefix, changing-tail layout.
- [Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures](https://arxiv.org/abs/2604.03515)
  (2026). Reports that SWE-agent truncates observations only every N steps so that the cached prefix
  survives: the only cache-aware truncation in a coding agent found.
- [KVTether](https://arxiv.org/abs/2609.39819) (2026). Lifecycle-aware KV-cache management for ReAct
  agents, reclaiming discarded entries: the same concern on the serving side.
- [Anthropic context editing](https://docs.claude.com/en/docs/build-with-claude/context-editing).
  The API clears old tool results past a threshold and leaves placeholders: an edit in the middle
  of the context, which selfctx avoids for cache reasons.
