# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Generation controls. `GenerationParams` (temperature, top_p, top_k, stop
  sequences, seed) is now a field on both `luft.Config` and
  `luft.ProviderRequest`, threaded through `Ask`, `AskStream`, `Loop`,
  `LoopStream`, `Agent`, and `Extract`. Fields are pointers/slices so the zero
  value preserves provider defaults (and `temperature: 0` is distinct from
  unset). New `luft.Ptr` helper for setting them. Per-provider support varies
  and unsupported fields are dropped — see the `GenerationParams` doc comment.

### Fixed

- Retries now cover Anthropic's `529` "overloaded" status and the
  `500`/`502`/`503`/`504` server and gateway errors, in addition to `429`.
  Previously only `429` and `503` were retried, so Anthropic overload errors
  failed hard.
- `Retry-After` response headers are now parsed by every provider (new exported
  `luft.ParseRetryAfter`) and honored by the backoff. The retry layer already
  respected `APIError.RetryAfter`, but no provider populated it, so the hint
  was silently ignored until now.

## v0.1.2 — 2026-05-09

### Added

- Image content blocks. New `luft.TypeImage` constant; `ContentBlock`
  gained `Source` (base64 data URI or http(s) URL) and `MediaType` fields.
  Image blocks round-trip as typed fields, not via `Raw`.
- `luft.ImageBlock` helper struct and `luft.NewUserMessageWithImages(text, images)`
  constructor for emitting user-role turns that carry one text block followed
  by image blocks.
- `luft.ToolResult.Images []ImageBlock` for tools that emit image
  attachments. `NewToolResultMessage` flattens images from every tool result
  onto the same canonical user-role message alongside the existing
  `tool_result` blocks; the OpenAI wire serializer splits them into a
  sibling `role:"user"` message at send time.
- `luft.AttachImage(ctx, ImageBlock)` for ToolFuncs to ride image bytes
  back to the model alongside their textual output. The agent loop installs
  a per-call sink and drains it onto `ToolResult.Images`. `WithImageSink`
  exposes the same primitive for unit-testing image-emitting tools without
  standing up a full Loop.
- `tools/workspace/read_file` now recognises images (PNG, JPEG, GIF, WebP,
  BMP, TIFF) by magic-byte sniff with extension fallback. It returns a
  one-line metadata payload (`image: <path> (<mime>, <bytes> B)`) and
  attaches the bytes as a base64 data URI. Bails when the file exceeds
  `MaxFileBytes` rather than emit a corrupt data URI. Text-file behaviour
  is unchanged.
- `tools/web/web_fetch` now recognises `image/*` responses (header first,
  body sniff fallback). Same metadata-text-plus-data-URI shape as
  `read_file`. Bails when the body exceeds `MaxBodyBytes`. HTML-to-text and
  `raw=true` paths are unchanged.

### Changed

- `providers/openai`: the user-role branch of `toOpenAIMessages` now emits
  the OpenAI typed-parts content shape (`[{type:"text",...},{type:"image_url",
  image_url:{url:...}}]`) when any image block is present on the canonical
  message. Plain-string and single-part-text-with-`cache_control` paths
  remain byte-identical for messages without images. The `role:"tool"`
  splitting for `tool_result` blocks is unchanged. Same change applies
  transparently to `providers/openrouter`, which delegates to the shared
  helpers.

## v0.1.1 — 2026-05-08

### Added

- `providers/openrouter` now supports OpenRouter's hosted server-side tools.
  `openrouter.WebSearch(opts)` constructs a `luft.ProviderTool` for the
  `openrouter:web_search` hosted tool; the model decides when to invoke and
  OpenRouter executes the search server-side.
- `openrouter.ProviderTag` constant ("openrouter") for tagging hosted tools;
  the OpenRouter provider now rejects `ProviderTool` entries tagged for
  other providers at request build time.
- Citation surfacing on OpenAI-compatible `Call`: when a backend (e.g.
  OpenRouter web search) attaches `annotations` to the assistant message,
  each entry is emitted as an opaque `luft.ContentBlock` (Type set from
  the annotation's `type`, full JSON in `Raw`). Streaming citations are
  not yet surfaced.

### Changed

- `openai.CompatibleCall` and `openai.CompatibleStream` gained a final
  `allowProviderTools bool` parameter. Stock OpenAI passes `false` (Chat
  Completions does not host tools); OpenRouter passes `true`. External
  callers of these helpers will need to add the new argument.
- `toOpenAITools` now returns `[]json.RawMessage` so opaque hosted-tool
  entries can be spliced alongside function tools.

## v0.1.0 — 2026-05-08

First tagged release. The library and CLI ship together under a single
module version.

### Library

- `Ask` / `AskStream` for one-shot model calls with usage reporting.
- `Loop` / `LoopStream` for tool-using loops over plain `[]Message` history.
- `Extract[T]` for typed structured output, with optional intermediate tools
  and a `Validate` hook (built on `ToolMetadata.Terminal`).
- `Agent` block: a thin composition of client, system prompt, toolset,
  context manager, iteration limit, and hooks. `Step` and `StepStream`.
- `Parallel` for fanning out independent `Ask`/`Loop` calls.
- `Toolset`, `ToolBinding`, `Tools`, `Bind`, `Join`, `MustJoin`. Typed tools
  via `NewTypedTool`. Schema helpers (`Object`, `String`, `Number`,
  `Array`, `Required`, …).
- Middleware: `WithTimeout`, `WithResultLimit`, `WithLogging`,
  `WithPanicRecovery`, `WithConfirmation`.
- `ContextManager` for explicit history trimming with tool-use/result
  integrity preservation; optional summarization.
- Prompt-cache markers (`Ephemeral`, `EphemeralExtended`) honored by
  Anthropic and OpenRouter; transparently dropped by OpenAI providers.
- `Provider` interface plus implementations:
  - `providers/anthropic` — Messages API, server-executed and
    provider-defined tools (web search, code execution, bash, text editor).
  - `providers/openai` — Chat Completions and Responses, with hosted
    tools on Responses (web search, file search, code interpreter,
    image generation).
  - `providers/openrouter` — OpenAI-compatible with cache-marker
    translation for Anthropic backends.
- Sessions: plain `Session` data, `MemoryStore` and `FileStore`
  implementations of the five-method `Store` interface.
- Streaming with `StreamBuffer` for retry-aware partial-output handling.
- Typed errors: `APIError`, `ToolError`, `LoopError`,
  `RetryExhaustedError`, `ErrMissingTool`, `ErrMaxIter`,
  `ErrSessionExists`, `ErrSessionNotFound`.
- Configurable retry with exponential backoff and `OnRetry` callback.
- JSONL session recorder for offline replay and debugging.

### Built-in tools

- `tools/clock` — current UTC time.
- `tools/math` — safe calculator.
- `tools/workspace` — sandboxed `list_directory`, `Glob`, `Grep`,
  `read_file`, `file_info`, and (in the read/write build) exact-string
  `edit_file`.
- `tools/bash` — shell tool with `restricted` / `standard` /
  `unrestricted` safety modes, working directory sandboxing.
- `tools/editor` — provider-trained `str_replace_based_edit_tool`
  (view / create / str_replace / insert).
- `tools/todo` — `todo_write` / `todo_read` for in-conversation
  planning checklists.
- `tools/batch` — fan-out tool that runs 2+ independent read-only
  tool calls concurrently in a single turn.
- `tools/web` — native `web_fetch` with HTML-to-text and pagination.
- `tools/subagent` — wrap any `Agent` as a tool callable by another
  agent; iteration history stays out of the parent's context.

### MCP

- `mcp` package adapts Model Context Protocol servers into ordinary
  toolsets via `mcp.Connect` / `Server.Toolset`.

### CLI (`cmd/luft`)

- Multi-agent CLI coding assistant on OpenRouter:
  - main agent (`x-ai/grok-4.3` by default) with read-only workspace
    tools, bash, editor, todo, batch, and web_fetch.
  - `explore` subagent (`openai/gpt-oss-120b`) for cheap, parallelisable
    repo research and bounded Q&A.
  - `plan` subagent (`x-ai/grok-4.3`) for design, architecture, and
    hard-debugging questions.
- Flags: `-dir`, `-model`, `-explore-model`, `-plan-model`,
  `-no-subagents`, `-no-fetch`, `-bash`, `-yes`, `-max-iter`, `-log`,
  `-version`.
- Env-var fallbacks: `LUFT_MODEL`, `LUFT_EXPLORE_MODEL`,
  `LUFT_PLAN_MODEL`, `LUFT_SUMMARIZE_MODEL`.
- REPL with `:exit`, `:reset`, `:tokens`, `:help`, plus `/compact`
  for summarising history mid-session.
- Unified-diff preview in edit confirmation prompts.
- Optional JSONL session log (`-log auto` writes under
  `~/.config/luft/sessions/`).
- Project memory: `AGENTS.md` and `CLAUDE.md` from the workspace and
  user-level (`~/.config/luft/AGENTS.md`, `~/.claude/CLAUDE.md`) are
  appended to the system prompt.
