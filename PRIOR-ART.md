# Prior art

This document concerns **selfctx2**, not the predecessor `../agent-self-context`. The predecessor lets
the model number context entries and cut/backtrack to an earlier entry. selfctx2 has no numbered
context or backtracking: each turn appends a durable summary and observations, and model-assigned
call IDs allow earlier full tool results to be retrieved without rerunning them.

## Durable summaries and self-managed context

- [Focus: Active Context Compression](https://arxiv.org/abs/2601.07190) (2026). The agent decides when
  to fold learnings into a persistent "Knowledge" block and drop raw history. Like selfctx2, it gives
  the acting model responsibility for memory; unlike selfctx2, compression discards source context.
- [Scaling Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967) (ByteDance Seed,
  CMU, Stanford, 2025; [project page](https://context-folding.github.io/)). Actions branch into a
  sub-trajectory and return it folded into a short summary, trained with RL. selfctx2 instead uses
  untrained per-turn summaries in one monotonic trajectory.
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (2023), now
  [Letta](https://docs.letta.com/letta_memgpt). The LLM moves information between context and external
  storage through memory tools. selfctx2's summaries stay in the model context; its separate result
  store is addressed by call ID rather than by general memory operations.
- [Self-GC: Self-Governing Context for Long-Horizon LLM Agents](https://arxiv.org/abs/2607.00692)
  (Xiaohongshu, 2026; [summary](https://www.deeplearning.ai/the-batch/llms-take-out-the-agents-trash)).
  Context objects are folded, masked, and pruned by a separate planner. selfctx2 has no eviction or
  separate planner.
- [Beyond Compaction: Structured Context Eviction](https://arxiv.org/abs/2606.11213) (2026). The agent
  tags typed episodes and a deterministic policy evicts them. selfctx2 summarizes every turn but never
  evicts context entries.

## Addressable and retrievable tool results

- [Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)
  separates tools from resources: tools perform actions, while resources expose data by URI. selfctx2
  similarly separates execution from retrieval, but its IDs reference immutable results of calls from
  the same run rather than server-hosted resources.
- [MemGPT/Letta archival memory](https://docs.letta.com/concepts/memory/) and common retrieval-augmented
  agents retrieve stored text by semantic search. selfctx2 retrieval is exact and deterministic: the
  model chooses the ID at call time and retrieves that call's original result, not a similar document.
- [Anthropic context editing](https://docs.claude.com/en/docs/build-with-claude/context-editing) clears
  old tool results past a threshold and leaves placeholders. selfctx2 preserves the original context
  entries and can append an identified result again; it does not use placeholder-based eviction.
- [Langroid `rewind_tool`](https://langroid.github.io/langroid/reference/agent/tools/rewind_tool),
  [AgentRewind](https://www.alphaxiv.org/abs/2608.14380), and coding-agent `/rewind` commands restore
  earlier trajectory state. Retrieval in selfctx2 is not rewind: no state or context is rolled back,
  and the retrieved result is appended after the current turn.

## Caching awareness

- [Don't Break the Cache: An Evaluation of Prompt Caching](https://arxiv.org/abs/2601.06007) (2026).
  Prompt caching cuts agent API cost by 45-80%; changing tool results can cause cache writes that are
  never reused. This supports selfctx2's byte-identical prefix and append-only retrieval.
- [Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures](https://arxiv.org/abs/2604.03515)
  (2026). Reports that SWE-agent truncates observations only every N steps so that the cached prefix
  survives. selfctx2 needs no scheduled truncation because existing entries are immutable.
- [KVTether](https://arxiv.org/abs/2609.39819) (2026). Lifecycle-aware KV-cache management for ReAct
  agents reclaims discarded entries on the serving side. selfctx2 instead avoids discarding prefix
  entries at the client-protocol level.

## Cache and cost cues

The predecessor's survey identifies a gap: agents are shown token budgets, but not their own cache
hits or the uncached cost of cuts. selfctx2 removes cuts entirely; IDs and retrieval are exposed in
the constant system prompt and JSON schema, while the changing cache/context cue remains last so
it cannot invalidate the prefix. The closest cache-cue work is:

- [TokenPilot](https://arxiv.org/abs/2606.17016) (2026): stable placeholders and harness-controlled
  lifecycle eviction for cache efficiency; selfctx2 instead gives the model exact result handles.
- [Cache-Aware Prompt Compression](https://arxiv.org/abs/2607.15516) (2026): prices cached and
  uncached input separately, the cost distinction behind the cache cue.
- [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching):
  reuse requires an unchanged prefix; a cached prefix processed twice is cheaper than two uncached
  reads. Retrieval therefore appends and never rewrites the identified original.
- [Budget-Aware Tool-Use](https://arxiv.org/abs/2511.17006) (2025): exposing resource budgets changes
  agent behavior; nearest analogue to showing context/cache state to the model.
