# Onward

A coding agent that keeps tool results in context for one model request and stores them for retrieval.

The model requests an ordered list of structured tool calls. The host generates placeholders for
calls and older results. Model-assigned IDs let the model retrieve complete results without
rerunning the tools.

Related work is surveyed in [PRIOR-ART.md](PRIOR-ART.md).

## Context model

- The first task follows the system prompt. Later REPL tasks, answers, and tool entries remain in
  chronological order.
- After receiving a tool-call list, the host adds a `[tool calls: N]` placeholder to the context
  immediately and records the original reply in the run log.
- The next model request includes the new tool results, in call order. As soon as that model reply
  arrives, the host replaces those results with placeholders such as `[result "source-v1"]` or
  `[error "tests-before"]`. This applies to retrieved results too. A placeholder for a batch
  retrieval lists the IDs it requested.
- The host generates these markers from the call count, IDs, and result status. Calls and results
  keep their place in the conversation, and full results remain available by ID.
- With a summary model configured, each compacted call or result entry keeps a short generated note
  under its marker. The separate model summarizes the entire batch in one request. The run's usage
  counts the acting model, including the notes it reads as context.

## Turn protocol

Each model response is one JSON object:

```json
{
  "tool_calls": [
    {"type": "file", "id": "source-v1", "file": "path/to/file.py", "lines": [10, 40]},
    {"type": "tool", "id": "tests-before", "tool": "exec_shell", "args": {"command": "pytest -q"}},
    {"type": "retrieve", "retrieve": "source-v1"}
  ],
  "done": null
}
```

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

Malformed output produces a placeholder and a protocol error in context. The original reply is
available in the run log. The next request lets the model correct the error.

## Caching awareness

Replacing the previous turn's results shortens the prompt and breaks cache reuse from the first
replaced result onward. The system prompt, schema, and earlier placeholders are stable. Once a
result becomes a placeholder, future requests can reuse that part of the prefix. The changing
context-size reminder and turn instruction remain last.

Cache hits are logged per turn as `usage.cached_tokens` when the provider reports them.

## Implementation

The `onward` package contains `_items.py` (context rendering), `_protocol.py` (system prompt,
JSON schema, validation), `_loop.py` (turn execution), `_results.py` (disk-backed result store),
`_repl.py` (interactive input), and `__main__.py` (CLI).

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
onward                                  # REPL when stdin is a terminal
onward "make the tests pass"            # one-shot: prints the final answer
onward --json "make the tests pass"     # {"final", "turns", "items", "log"}
echo "task" | onward                    # piped stdin remains one-shot
onward --model k3.1-pro --inference-db kimi-coding --max-turns 50 --context-budget 20000 --log-dir runs "task"
onward --trace-context "make the tests pass"  # include prompt messages in the run log
onward --summary-model gpt-6-luna --trace-context "make the tests pass"
```

With no arguments on a terminal, Onward starts a REPL. Each prompt runs a task in the same
conversation, preserving earlier tasks, answers, and tool-result IDs. Enter `exit`, `quit`, `q`,
or Ctrl-D to quit; blank input is ignored. `--max-turns` is per task. In JSON mode, each answer is a
separate JSON document; prompts and diagnostics stay on stderr.

Defaults: Kimi K3.1 Pro through agentknit's inference-db entry `kimi-coding`.
In one-shot mode, exit code 0 means the model set `done`; 1 means `--max-turns` elapsed without an answer; 2 means an
empty task. Tool output is streamed to stderr; stdout carries only the answer. `--no-schema` omits
the turn schema for endpoints without `json_schema`. The context budget is a soft limit shown to the model.
The summary model uses the acting model's inference-db entry by default. Set
`--summary-inference-db` to use a different provider. The summary endpoint must support JSON
schema responses.

Each invocation writes `runs/<timestamp>.jsonl` and `runs/<timestamp>.results/`, shared by all tasks
in a REPL. Result files contain the
full JSON-encoded observations; `index.jsonl` maps IDs to files. Each turn record contains: `turn`, `items_before`, `items_after`,
`new_items` (kind, estimated tokens, and ID for new file or tool results), `usage` (`prompt_tokens`,
`completion_tokens`, `cached_tokens`), `finish_reason`, `reply`, and `error` when the reply broke the
protocol. `new_items` is present for valid turns that continue the task.

```sh
jq -c '{turn, items_before, items_after, prompt: .usage.prompt_tokens, cached: .usage.cached_tokens}' runs/*.jsonl
```

With `--trace-context`, every logged turn includes the acting model name and the exact `messages`
passed to the model client.
Open [the context viewer](tools/context-viewer.html) in a browser and select that run's JSONL file.
The timeline alternates between requests and responses. Request steps show additions, replacements,
and the unchanged prefix. Response steps show the raw model reply. Select a changed entry to compare
its exact text before and after. Recorded prompts can include source code and tool output.

The viewer counts each retrieve action once and counts each requested ID in the per-tool-call
table. Re-read tokens estimate the size of retrieval output that reached a later prompt, using
four characters per token with rounding per message. The viewer shows run totals and counts
through the selected request or response.

Select up to three JSONL traces together to compare run totals, cache usage, and prompt sizes. Use the
trace switch above the timeline to inspect either run at the selected step.
For GPT-6.2 Sol traces with complete token counts, the viewer estimates Standard API costs from
uncached input, cached input, and output tokens. The estimate uses the recorded acting-model
usage and prices tokens without a logged cache hit as ordinary input.
[OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-6.2-sol#pricing)

## Development

```sh
.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy onward
```

The tests use a scripted fake model. They cover structured tool-call validation, chronological
execution, error recovery, completion, and the lifetime of results and placeholders in the prompt.
`tests/fixtures/buggy/` is a small project with failing tests for live runs.
