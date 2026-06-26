# luft CLI

A fast, economical CLI coding agent built on the luft toolkit. Inspired by Claude Code; written in Go; opinionated about cost and parallelism out of the box.

## Install / run

You'll need an OpenRouter API key in your environment for any of these:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Models default to `z-ai/glm-5.2` for the main agent and plan subagent,
and `google/gemini-3.5-flash` for the explore subagent (super fast and cheap
for inspection workloads). `-model`, `-explore-model`, and `-plan-model`
accept any OpenRouter slug (e.g. `openai/gpt-5`, `google/gemini-2.5-pro`,
`anthropic/claude-sonnet-4.6`). Each flag has a matching env var
(`LUFT_MODEL`, `LUFT_EXPLORE_MODEL`, `LUFT_PLAN_MODEL`, plus
`LUFT_SUMMARIZE_MODEL` for the `/compact` summarizer) so you can pin
tiers in your shell rc or a per-project `.envrc`.

### Option A — install once, run anywhere (recommended)

```bash
go install github.com/lukemuz/luft/cmd/luft@latest
```

This drops a `luft` binary in `$(go env GOBIN)` (or `$(go env GOPATH)/bin` if `GOBIN` is unset). Make sure that directory is on your `PATH`:

```bash
echo $PATH | tr ':' '\n' | grep -q "$(go env GOPATH)/bin" || \
  echo 'add $(go env GOPATH)/bin to your PATH'
```

Then from any directory you want the agent to work in:

```bash
cd ~/your-project
luft
```

To re-install after pulling new commits, run `go install ...` again.

### Option B — build a local binary in this repo

From the luft checkout:

```bash
go build -o bin/luft ./cmd/luft
cd ~/your-project && /path/to/luft/bin/luft
```

Or symlink it into your `PATH`:

```bash
sudo ln -s "$(pwd)/bin/luft" /usr/local/bin/luft
```

### Option C — `go run` (development only)

Useful while editing luft itself. From the project you want to work on:

```bash
go run github.com/lukemuz/luft/cmd/luft
```

Slow start every time (re-compiles), but no binary to manage. Pass `-dir` if you'd rather invoke it from elsewhere.

### First-run sanity check

```bash
cd ~/your-project
luft
```

The agent operates on the current working directory by default. Pass `-dir <path>` to point it elsewhere without changing shells.

You should see:
```
luft  model=z-ai/glm-5.2  bash=restricted  subagents=on  dir=/abs/path
        explore=google/gemini-3.5-flash  plan=z-ai/glm-5.2
type a request, or /help for commands. ctrl-c to interrupt, ctrl-d to exit.
> 
```

If you get `openrouter provider: OPENROUTER_API_KEY environment variable is not set`, you missed the `export` step.

> Tip: pass `-log auto` if you want a JSON Lines trace of the session for debugging — see [Session logging](#session-logging) below. It's not needed for normal use.

## What's running

```
main agent (z-ai/glm-5.2 by default)
  ├── direct tools  workspace + bash + str_replace_based_edit_tool
  │                 + todo + clock + batch
  ├── explore       subagent on google/gemini-3.5-flash — read-only fs + restricted bash + batch
  └── plan          subagent on z-ai/glm-5.2 — read-only fs only, no shell, no edits
```

Why three agents? Cost tiering and context isolation. The main GLM model decides what work to do; cheap inspection happens on gemini-3.5-flash and never enters the main context (only the subagent's final summary returns); hard reasoning stays on a strong model via the `plan` tool when wanted.

## Tools available to the main agent

| Tool | What it does | Confirmation |
|---|---|---|
| `list_directory`, `Glob`, `Grep`, `read_file`, `file_info` | Read-only filesystem inspection (workspace package) | no |
| `str_replace_based_edit_tool` | Text editor: view / create / str_replace / insert | yes |
| `multi_edit` | Apply several str_replace edits to one file atomically (all-or-nothing) | yes |
| `bash` | Sandboxed shell, governed by `-bash` mode | yes (in standard/unrestricted modes) |
| `todo_write`, `todo_read` | Planning checklist; replace-whole-list semantics | no |
| `batch` | Run several read-only tool calls concurrently in one turn | no |
| `web_fetch` | Download a URL over http(s); HTML→text, paginates long pages | no |
| `web_search` | OpenRouter hosted search; results with citations (disable with `-no-search`) | no |
| `now` | Current time | no |
| `explore(task)` | Delegate inspection to a gemini-3.5-flash-backed subagent | no |
| `plan(task)` | Delegate hard reasoning to a glm-5.2-backed subagent | no |

`Glob` and `Grep` use the same names Claude Code uses, so the model recognises them immediately. `web_fetch` is a native Go tool (no external dependency, no API key) that downloads a URL, strips scripts/styles, decodes entities, and paginates long pages via `max_length` + `start_index`. Disable it with `-no-fetch`. `web_search` is OpenRouter's hosted search tool — executed server-side and returned inline with citations, so it works regardless of the underlying model slug and needs no extra API key beyond your OpenRouter one. Disable it with `-no-search`. The two pair naturally: `web_search` to find a source, `web_fetch` to read it in full.

The `explore` and `plan` subagents are themselves agents with their own toolsets — the main agent can ask them anything within their scope.

## Context economy

Two things make this fast and cheap:

1. **Stable cacheable prefix.** System prompt + project memory + tool definitions are marked as cache breakpoints. After the first turn of a session, those tokens cost ~10% of normal. Cache hit rate shows up in `/tokens`.

2. **Subagent context isolation.** When `explore` runs a 30-file investigation, all that searching and reading happens in the subagent's loop and dies with it. Only the textual summary returns to the main agent.

3. **Re-read dedup.** Reading the same file twice with the same arguments returns a short "unchanged" notice instead of the full content again (the tool still runs, so a changed file always returns fresh content). Pass a line range to force a fresh read if you need it.

4. **Oversized-output compression.** `bash` output above `-result-budget` bytes (default 12K) is condensed by the summarizer model to its errors, failures, and final status before it lands in context — so a 50K test log doesn't cost 50K of tokens. Set `-result-budget 0` to disable and fall back to plain truncation. File reads are never compressed (you need them verbatim to edit correctly).

Context is managed two ways. **Automatically:** before each model call, history over ~750K tokens is trimmed and the trimmed span is replaced by a compact summary (not silently dropped) using the summarizer model — so long autonomous runs degrade gracefully instead of losing the middle of their own work. The threshold leaves headroom under the 1M-token windows of the default models. **Manually:** run `/compact` (see below) any time to compress older turns yourself. Both paths use the summarizer (glm-5.2 by default; override with `LUFT_SUMMARIZE_MODEL`).

## Project memory

On startup luft loads, in order, and concatenates:

1. `<workspace>/AGENTS.md`        — vendor-neutral convention
2. `<workspace>/CLAUDE.md`        — Claude Code's flavour, picked up for compatibility
3. `~/.config/luft/AGENTS.md`   — your personal luft preferences
4. `~/.claude/CLAUDE.md`          — your existing Claude Code memory, reused

Anything found is appended to the system prompt under "## Project memory" and becomes part of the cached prefix. View what's loaded with `/memory`.

A good `AGENTS.md` is short and concrete: project conventions, how to run tests, the language/style the project prefers. The shorter and more stable it is, the better the cache works.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-dir` | cwd | Working directory the agent is sandboxed to (defaults to the directory you launched from) |
| `-model` | `z-ai/glm-5.2` | Main-agent model (any OpenRouter slug; env: `LUFT_MODEL`) |
| `-explore-model` | `google/gemini-3.5-flash` | Model for the explore subagent (env: `LUFT_EXPLORE_MODEL`) |
| `-plan-model` | `z-ai/glm-5.2` | Model for the plan subagent (env: `LUFT_PLAN_MODEL`) |
| `-no-subagents` | false | Disable the explore and plan tools |
| `-no-fetch` | false | Disable the native `web_fetch` tool |
| `-no-search` | false | Disable the OpenRouter `web_search` provider tool |
| `-bash` | `restricted` | `restricted` \| `standard` \| `unrestricted` |
| `-yes` | false | Auto-approve every confirmation prompt |
| `-max-iter` | 30 | Max model calls per user turn |
| `-context-budget` | `750000` | Token budget before automatic in-loop summarization kicks in (env: `LUFT_CONTEXT_BUDGET`) |
| `-result-budget` | `12000` | Byte threshold above which noisy `bash` output is summarized; `0` disables (env: `LUFT_RESULT_BUDGET`) |
| `-log` | (off) | JSONL session log path. Use `-log auto` to write under `~/.config/luft/sessions/<timestamp>.jsonl`, or pass an explicit path. |
| `-continue` | false | Resume the most recent persisted session for the current working directory |
| `-resume <id>` | (off) | Resume a specific session by ID (mutually exclusive with `-continue`) |

### Bash safety modes

- **`restricted`** (default, no confirmation): only commands whose first token is on a curated allowlist (`ls`, `cat`, `grep`, `rg`, `find`, `git status/diff/log`, `go`, `wc`, `head`, `tail`, etc.). Shell metacharacters (`;`, `&`, `|`, `\``, `$`, `<`, `>`) are rejected — the model can't smuggle in a second command.
- **`standard`** (confirmation required): open shell with a deny-list for obviously dangerous patterns (`rm -rf /`, `sudo`, `curl|sh`, `dd of=/dev/`, fork bombs, writes to `/etc`).
- **`unrestricted`** (confirmation required): only the timeout (30s) and output cap (64 KiB) apply.

You can pair any mode with `-yes` to auto-approve, e.g. for headless runs.

### Confirmation prompts

Edits (`str_replace_based_edit_tool`, `multi_edit`) and shell (in `standard`/`unrestricted` modes) prompt for approval on stderr with a diff/argument preview. Answer:

- **`y`** — approve this one call
- **`a`** — approve every call to *this tool* for the rest of the session (e.g. trust all edits once you're confident, while still confirming shell)
- **`N`** (default) — reject; the model is told it wasn't approved and continues

`-yes` skips all prompts including dangerous shell, so prefer the sticky **`a`** for interactive runs.

## Slash commands

Both `/cmd` and `:cmd` are accepted.

| Command | What it does |
|---|---|
| `/help` | Show all commands |
| `/exit`, `/quit` | Leave the REPL |
| `/reset`, `/clear` | Drop conversation history |
| `/compact [instructions]` | Summarise older turns; keeps the last 4 user turns verbatim. Optional instructions steer the summary (e.g. `/compact focus on the auth refactor`). |
| `/tokens` | Print accumulated input/output/cache tokens and cache hit rate |
| `/memory` | Print the loaded project memory |
| `/tools` | List available tools (with `[confirm]` flag) |
| `/model <id>` | Switch the main-agent model mid-session (subagent models unchanged) |
| `/log` | Print the active JSONL log path (if any) |
| `/session` | Print the current session ID and its on-disk location |

## Session persistence and resume

Each REPL session has a stable ID and is persisted after every completed turn as a JSON snapshot under `~/.config/luft/store/<id>.json`. The conversation survives exiting the process; resume it with:

- `luft -continue` — resume the **most recent** session for the current working directory.
- `luft -resume <id>` — resume a specific session by ID.

The session ID embeds a hash of the working directory as a prefix, so `-continue` scopes its lookup to sessions started in *this* directory. On resume, prior history (and accumulated token usage) is restored. If there is nothing to continue, or the stored session is corrupt, luft starts a fresh session with a notice rather than crashing. `-continue` and `-resume` are mutually exclusive.

This is separate from `-log` (the JSONL event trace): `-log` records a streaming per-process trace; the session store is the durable conversation snapshot. Both can be active at once.

## Session logging

Session logging is **opt-in and intended for debugging** — leave it off for everyday use.

`-log auto` (or `-log <path>`) writes a JSON Lines trace of the entire session to disk. Every model request, response, retry, tool call (start and end with input + output), and turn boundary is recorded — for the main agent and both subagents. The file is append-only, safe to read while the session is running.

Enable it when something feels off and you want to look at what actually happened. `jq -c '.type' session.jsonl | sort | uniq -c` is a good first pass; pipe specific events to `jq -c 'select(.type == "tool_call_end") | {tool: .tool_name, bytes: (.tool_output | length), error: .is_error}'` for tool-level inspection.

Note: session logs include full tool inputs and outputs, which means file contents the agent read end up on disk under `~/.config/luft/sessions/`. Skip `-log` if you're working with anything sensitive, or pass an explicit `-log <path>` to a location you control.

## Recipes

**Quick code question:**
```
> what's the difference between Loop and StepStream?
```
The main agent answers directly using its read-only tools.

**Investigation:**
```
> find every place we read environment variables and summarise the patterns
```
The main agent should delegate this to `explore`. The gemini-3.5-flash subagent fans out greps via `batch`, reads candidate files, and returns a summary. Cheap.

**Refactor:**
```
> rename ToolFunc to ToolHandler everywhere, update tests, leave a deprecation alias
```
The main agent plans via `todo_write`, uses `str_replace_based_edit_tool` for edits, runs `go test ./...` via `bash` (you'll be prompted to approve).

**Architecture decision:**
```
> we want to add a per-session sqlite cache for file shas. design it.
```
The main agent should call `plan` to delegate the design, then summarise back.

## Things to watch the first time you run it

- **Did the main agent delegate to `explore`?** If it does inspection inline instead of calling `explore`, the system prompt may need tuning. Watch the stderr `[tool ...]` lines.
- **Is the cache working?** After 2-3 turns, `/tokens` should show a non-trivial cache hit rate. If it's zero, something is breaking the prefix.
- **Confirmation prompts.** Edits and shell (in non-restricted modes) prompt on stderr. `-yes` skips this if you trust the run.

## What's not here yet

- Persistent / resumable sessions across restarts (history dies on exit today)
- Persistent codebase index across sessions
- Speculative inspection while the model is drafting
- OAuth / Claude subscription auth (API key only)

These are next on the list — see the project ROADMAP if it grows.
