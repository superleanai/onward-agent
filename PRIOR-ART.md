# Prior art

Onward uses fixed placeholders after each tool result has been visible for one request. Model-assigned
call IDs let it retrieve full results without rerunning the tools. Its predecessor
`../agent-self-context` lets the model number context entries and backtrack to an earlier entry.

## Durable summaries and self-managed context

- [Focus: Active Context Compression](https://arxiv.org/abs/2601.07190) (2026). The agent decides when
  to fold learnings into a persistent "Knowledge" block and drop raw history. Onward's host replaces
  older results with placeholders automatically and retains full results on disk.
- [Scaling Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967) (ByteDance Seed,
  CMU, Stanford, 2025; [project page](https://context-folding.github.io/)). Actions branch into a
  sub-trajectory and return it folded into a short summary, trained with RL. Onward uses one
  trajectory with direct retrieval by ID.
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (2023), now
  [Letta](https://docs.letta.com/letta_memgpt). The LLM moves information between context and external
  storage through memory tools. Onward's separate result store is addressed by call ID.
- [Self-GC: Self-Governing Context for Long-Horizon LLM Agents](https://arxiv.org/abs/2607.00692)
  (Xiaohongshu, 2026; [summary](https://www.deeplearning.ai/the-batch/llms-take-out-the-agents-trash)).
  Context objects are folded, masked, and pruned by a separate planner. Onward replaces each result
  on a fixed schedule.
- [Beyond Compaction: Structured Context Eviction](https://arxiv.org/abs/2606.11213) (2026). The agent
  tags typed episodes and a deterministic policy evicts them. Onward applies a fixed schedule to
  tool results and keeps a placeholder at each entry's original position.

## Addressable and retrievable tool results

- [Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)
  separates tools from resources: tools perform actions, while resources expose data by URI. Onward
  similarly separates execution from retrieval, but its IDs reference immutable results of calls from
  the same run rather than server-hosted resources.
- [MemGPT/Letta archival memory](https://docs.letta.com/concepts/memory/) and common retrieval-augmented
  agents retrieve stored text by semantic search. Onward retrieval is exact and deterministic: the
  model chooses the ID at call time and retrieves that call's original result, not a similar document.
- [Anthropic context editing](https://docs.claude.com/en/docs/build-with-claude/context-editing) clears
  old tool results past a threshold and leaves placeholders. Onward replaces results after one
  request and can show identified results again through retrieval.
- [Langroid `rewind_tool`](https://langroid.github.io/langroid/reference/agent/tools/rewind_tool),
  [AgentRewind](https://www.alphaxiv.org/abs/2608.14380), and coding-agent `/rewind` commands restore
  earlier trajectory state. Onward retrieval shows a past result in the next request and preserves
  the current trajectory.

## Caching awareness

- [Don't Break the Cache: An Evaluation of Prompt Caching](https://arxiv.org/abs/2601.06007) (2026).
  Prompt caching cuts agent API cost by 45-80%; changing tool results can cause cache writes that are
  never reused. Onward shortens the prompt by replacing a result soon after it appears. The shorter
  placeholder is stable on future requests.
- [Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures](https://arxiv.org/abs/2604.03515)
  (2026). Reports that SWE-agent truncates observations only every N steps so that the cached prefix
  survives. Onward replaces observations one request after they appear.
- [KVTether](https://arxiv.org/abs/2609.39819) (2026). Lifecycle-aware KV-cache management for ReAct
  agents reclaims discarded entries on the serving side. Onward replaces observations in the
  client-protocol prompt.

## Cache and cost cues

The predecessor's survey discusses showing agents token budgets and the costs of cache misses.
Onward exposes IDs and retrieval in the constant system prompt and JSON schema. A final message in
each request shows the context estimate and soft budget.

- [TokenPilot](https://arxiv.org/abs/2606.17016) (2026): stable placeholders and harness-controlled
  lifecycle eviction for cache efficiency. Onward's placeholders keep exact result IDs.
- [Cache-Aware Prompt Compression](https://arxiv.org/abs/2607.15516) (2026): prices cached and
  uncached input separately, the cost distinction behind the cache cue.
- [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching):
  reuse requires an unchanged prefix. Replacing the previous batch of results invalidates cache
  reuse from that batch onward.
- [Budget-Aware Tool-Use](https://arxiv.org/abs/2511.17006) (2025): exposing resource budgets changes
  agent behavior; nearest analogue to showing context/cache state to the model.
