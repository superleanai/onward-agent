# Onward

A coding agent with a monotonic, append-only context loop.

Core concept: at every turn the model writes a durable summary of the previous turn and an ordered
list of structured tool calls. Model-assigned tool-call IDs make every stored result addressable and
retrievable without rerunning the tool.

This differs from traditional coding agents, which keep everything in the context, and from its
predecessor selfctx, whose numbered context can be truncated and backtracked.

Related work is surveyed in [PRIOR-ART.md](PRIOR-ART.md).

## Context model

- The task is the first message after the system prompt. Model replies, observations, and errors are
  appended chronologically and never edited, removed, or reordered.
- A model reply contains a `summary`; the complete reply is appended verbatim, so every summary is
  durable context available on all later turns. The summary is also recorded separately in the run log.
- The first turn has no previous turn to summarize. Its summary records the model's initial
  understanding and approach instead.
- Tool calls do not have hidden results: their outputs are appended before the next turn. A summary
  therefore reports outcomes from the previous turn, not outcomes of calls being requested now.

## Turn protocol

Each model response is one JSON object:

```json
{
  "summary": "What the previous turn established or changed, and the resulting next step",
  "tool_calls": [
    {"type": "file", "id": "source-v1", "file": "path/to/file.py", "lines": [10, 40]},
    {"type": "tool", "id": "tests-before", "tool": "exec_shell", "args": {"command": "pytest -q"}},
    {"type": "retrieve", "retrieve": "source-v1"}
  ],
  "done": null
}
```

- `summary` is a required, non-empty string. It should preserve exact facts needed later without
  unnecessarily repeating observations that remain in context.
- `tool_calls` is an ordered list of file excerpts, tool calls, and retrievals. Every entry has an
  explicit `type`: `file`, `tool`, or `retrieve`. The host executes all of them and appends one
  result per call in the listed order. A file call's line range is 1-based and inclusive;
  `"lines": null` uses the default read length. File calls also register reads required by edit
  tools.
- `id` is a model-chosen string identifying a file or tool call. Results are headed `<id>`. The host
  retains the full result in a per-run store. `{"type": "retrieve", "retrieve": "id"}` appends that
  full stored result without executing the tool again; a list of IDs is also accepted. IDs must be
  unique, and retrieval of an unknown ID returns an error. IDs are mandatory. Full results are
  stored on disk, not in RAM; only the small ID/path index stays in memory.
- `done` is normally `null`. A string ends the run with that final answer; tool calls are then ignored.

Malformed output and invalid tool calls produce appended error context. The model can correct itself
on a later turn, but no operation can remove or rewrite earlier context.

## Caching awareness

The request is cache-friendly by construction:

- The system prompt and structured-output schema are constant across turns.
- One chat message is used per task, model reply, file excerpt, tool result, or error. Once sent,
  every such message remains byte-identical.
- New entries are appended at the end. There is no cut or rewrite that could invalidate an earlier
  prompt-cache prefix.
- The changing context-size reminder and turn instruction are in the final message only.

Cache hits are logged per turn as `usage.cached_tokens` when the provider reports them.

## Implementation

The `onward` package contains `_items.py` (append-only rendering), `_protocol.py` (system prompt,
JSON schema, validation), `_loop.py` (turn execution), and `_results.py` (disk-backed result store), and `__main__.py` (CLI).

agentknit (`../agentknit`, editable path dependency) supplies only the model client and tools
(`write_file`, `str_replace`, `exec_shell`; reads use file calls). No agentknit session runs, so its
compaction and awareness features do not alter this loop.

Each request carries the constant turn schema as `response_format: json_schema`, with the argument
schema of each available tool. Replies and tool arguments are also validated by the host because an
endpoint may accept a schema without enforcing it during decoding. New tool outputs longer than
30,000 characters keep their head and tail; retrieval by ID returns the full stored result.

## Install

```sh
uv venv .venv && uv pip install -p .venv -e '.[dev]'
```

## Usage

```sh
onward "make the tests pass"            # prints the final answer
onward --json "make the tests pass"     # {"final", "turns", "items", "log"}
echo "task" | onward
onward --model k3.1-pro --inference-db kimi-coding --max-turns 50 --context-budget 20000 --log-dir runs "task"
```

Defaults: Kimi K3.1 Pro through agentknit's inference-db entry `kimi-coding`.
Exit code 0 means the model set `done`; 1 means `--max-turns` elapsed without an answer; 2 means an
empty task. Tool output is streamed to stderr; stdout carries only the answer. `--no-schema` omits
the turn schema for endpoints without `json_schema`. The context budget is soft and informational;
the host never truncates context.

Each run writes `runs/<timestamp>.jsonl` and `runs/<timestamp>.results/`. Result files contain the
full JSON-encoded observations; `index.jsonl` maps IDs to files. Each turn record contains: `turn`, `items_before`, `items_after`,
`summary`, `new_items` (kind, estimated tokens, and model-assigned ID when present), `usage` (`prompt_tokens`,
`completion_tokens`, `cached_tokens`), `finish_reason`, `reply`, and `error` when the reply broke the
protocol. On a protocol error, `summary` and `new_items` are absent.

```sh
jq -c '{turn, summary, items_before, items_after, cached: .usage.cached_tokens}' runs/*.jsonl
```

## Development

```sh
.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy onward
```

The tests use a scripted fake model. They cover required summaries, structured tool-call validation,
chronological execution, error recovery, completion, and byte-identical cached prefixes across turns.
`tests/fixtures/buggy/` is a small project with failing tests for live runs.
