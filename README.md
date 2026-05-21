# luft

Build LLM agents in Go.

> Agent code that reads like Go.
>
> No YAML. No graph runtime. No hidden control flow. Ships as one static binary.

`luft` is a small set of primitives — `Client`, `Provider`, `Message`, `Tool`, `Loop`, `Agent` — composed with ordinary Go. Reading the code tells you what runs. Deploying it is `go build`.

It scales from a one-line model call to a production tool-using agent to its own coding CLI, without forcing a framework shape onto either end.

What's included:

- `Ask` / `AskStream` for model calls, `Loop` / `LoopStream` for tool-using loops
- `Extract[T]` for typed structured output (with or without intermediate tool use)
- plain `[]Message` history and normal Go functions as tools
- providers for Anthropic, OpenAI, OpenRouter, AWS Bedrock (Converse), and Vertex AI Gemini
- typed tools, schema helpers, toolsets, middleware, context management, MCP, a thin `Agent` block
- safe built-in tools (clock, math, sandboxed workspace)
- session persistence with a five-method `Store` interface
- structured `Recorder` events for observability (JSONL, in-memory, session-attached)
- retries, typed errors, streaming, usage tracking, prompt caching

Requires Go 1.21+. **Zero external dependencies** in the core package — the entire library compiles against the standard library.

## Install

~~~bash
go get github.com/lukemuz/luft
~~~

Set an API key for the provider you want:

~~~bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-...
~~~

## The smallest useful call

~~~go
client, err := anthropic.NewClientFromEnv(luft.ModelSonnet)
if err != nil {
    log.Fatal(err)
}

history := []luft.Message{
    luft.NewUserMessage("Give me three practical ideas for using LLMs in a Go service."),
}

reply, _, err := client.Ask(context.Background(), "You are concise.", history)
if err != nil {
    log.Fatal(err)
}

fmt.Println(luft.TextContent(reply))
~~~

No hidden session. No runner. `history` is just data.

For a step-by-step walkthrough see [`QUICKSTART.md`](QUICKSTART.md). For the design philosophy see [`VISION.md`](VISION.md). For an honest comparison with Google's ADK see [`COMPARISON.md`](COMPARISON.md).

## The largest worked example: a coding CLI

The repo ships [`cmd/luft`](cmd/luft) — a Claude Code-style coding agent built on luft. Sandboxed workspace tools, a bash tool with configurable safety modes, explore/plan subagents on cheaper models, project memory loaded from `AGENTS.md` / `CLAUDE.md`, JSONL session logging, slash commands. The whole thing is plain Go using the same primitives documented below.

~~~bash
go install github.com/lukemuz/luft/cmd/luft@latest
export OPENROUTER_API_KEY=sk-or-...
cd ~/your-project && luft
~~~

It's both a useful tool and the strongest argument for the library: the source for a real coding agent is small enough to read in one sitting. See [`cmd/luft/README.md`](cmd/luft/README.md) for full docs.

## Core building blocks

### `Provider`

A `Provider` translates between `luft`'s data model and an LLM API.

~~~go
type Provider interface {
    Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error)
    Stream(ctx context.Context, req ProviderRequest, onDelta func(ContentBlock)) (ProviderResponse, error)
}
~~~

Anthropic, OpenAI Chat Completions, OpenAI Responses, OpenRouter, AWS Bedrock (Converse / ConverseStream), and Vertex AI Gemini are included. Any backend can implement the interface.

### `Client`

A `Client` holds provider, model, token limit, and retry config. It does not store conversation state, so the same client reuses across conversations, requests, jobs, and goroutines.

~~~go
client, err := luft.New(luft.Config{
    Provider:  provider,
    Model:     luft.ModelSonnet,
    MaxTokens: 4096,
})
~~~

### `Message`

Conversation history is plain data. Append replies yourself when you want to continue:

~~~go
history := []luft.Message{luft.NewUserMessage("Hello")}

reply, _, err := client.Ask(ctx, system, history)
history = append(history, reply, luft.NewUserMessage("Tell me more."))
~~~

### `Tool` and `ToolFunc`

A tool has two parts: a model-facing definition and a Go function.

~~~go
tool, fn := luft.NewTypedTool(
    "calculator",
    "Do basic arithmetic.",
    luft.Object(
        luft.String("operation", "add, subtract, multiply, or divide", luft.Required()),
        luft.Number("a", "First number", luft.Required()),
        luft.Number("b", "Second number", luft.Required()),
    ),
    func(ctx context.Context, in CalculatorInput) (string, error) {
        return calculate(in)
    },
)
~~~

Tools compile down to ordinary values. A `Toolset` is an ordered slice of `ToolBinding{Tool, Func, Meta}`. `luft.Tools(...)` and `luft.Bind(tool, fn)` are variadic constructors for the common case:

~~~go
tools := luft.Tools(
    luft.Bind(tool, fn),
    luft.Bind(other, otherFn),
)
~~~

No hidden registry.

## From one call to a tool loop

### One model call

~~~go
reply, usage, err := client.Ask(ctx, system, history)
~~~

`usage` reports input/output tokens so cost-conscious code doesn't have to drop down to `Loop`.

### Tool loop

~~~go
result, err := client.Loop(ctx, system, history, tools, 5)
history = result.Messages
fmt.Println(result.FinalText())
~~~

`Loop` calls the model, runs requested tools, appends tool results, and repeats until the model returns a final answer or the iteration limit. Multiple tool calls in one model turn run concurrently and return in original order.

Because `Ask`, `Loop`, and `Agent.Step` are ordinary calls over plain data, they compose like any Go function — run two tool-using loops in parallel with `luft.Parallel`, then synthesize their outputs with a later `Ask`.

### Typed extraction

When you want a typed Go value back — with or without intermediate tool use — `Extract` runs a loop in which the model must call a single "submit" tool whose typed argument is the return value:

~~~go
type Plan struct {
    Steps []string `json:"steps"`
}

plan, result, err := luft.Extract[Plan](ctx, client, system, history,
    luft.ExtractParams[Plan]{
        Description: "Submit the final plan as a list of ordered steps.",
        Schema: luft.Object(
            luft.Array("steps", "ordered steps",
                luft.SchemaProperty{Type: "string"}, luft.Required()),
        ),
        // Tools: searchTools,           // optional: search-then-submit
        // Validate: func(p Plan) error  // optional: reject and let the model retry
    })
~~~

`Extract` is built on `ToolMetadata.Terminal` — a flag that tells `Loop` to short-circuit when a tool is invoked successfully. You can set it yourself for hand-rolled submit patterns; `Extract` is the headline sugar.

## Practical assembly

### Toolsets and middleware

~~~go
toolset := luft.MustJoin(clockTool.Toolset(), workspaceToolset).Wrap(
    luft.WithTimeout(5*time.Second),
    luft.WithResultLimit(20_000),
    luft.WithConfirmation(confirm),
)

result, err := client.Loop(ctx, system, history, toolset, 10)
~~~

`MustJoin` is for static composition where a duplicate tool name is a programmer error. `Join` returns an error for dynamic composition.

Available middleware: `WithTimeout`, `WithResultLimit`, `WithLogging`, `WithPanicRecovery`, `WithConfirmation`. Metadata is advisory; your application decides policy.

### Context management

`ContextManager` trims history explicitly before a call.

~~~go
cm := luft.ContextManager{MaxTokens: 8000, KeepFirst: 1, KeepRecent: 20}
trimmed, err := cm.Trim(ctx, history)
~~~

The original history is not mutated. Tool-use/tool-result integrity is preserved. Summarization happens only if you configure a summarizer.

### Agent

`Agent` is the blessed middle path: a thin block over a client, prompt, toolset, context manager, iteration limit, and hooks.

~~~go
a := luft.Agent{
    Client:  client,
    System:  "You are a helpful assistant.",
    Tools:   toolset,
    Context: luft.ContextManager{MaxTokens: 8000, KeepRecent: 20},
    MaxIter: 10,
}

// One-shot autonomous task: pass the goal as a single user message.
result, err := a.Step(ctx, []luft.Message{luft.NewUserMessage("do the thing")})

// Multi-turn: call Step once per human turn, threading history.
result, err = a.Step(ctx, history)
~~~

`Step` trims history once up front and again before every model call inside the loop (when a `ContextManager` is configured), so long autonomous runs don't silently blow the context window. `Hooks.OnIteration` observes each iteration; the underlying `Loop` and `ContextManager.Trim` primitives stay available if you want a different policy. No persistence, scheduler, runner, or hidden lifecycle.

## Built-in tools

| Package | Tools |
|---|---|
| `tools/clock` | current UTC time |
| `tools/math` | safe calculator |
| `tools/workspace` | sandboxed list, find, search, read, file info, exact-string edit |

~~~go
clockTool := clock.New()
ws, err := workspace.NewReadOnly(workspace.Config{Root: "."})
toolset := luft.MustJoin(clockTool.Toolset(), ws.Toolset())
~~~

`workspace.NewReadOnly` is read-only. `workspace.New` includes `edit_file` — wrap it with `WithConfirmation` before letting writes run.

## Provider tools

Some tools live on the provider side: Anthropic and OpenAI ship a set of tools the model is already trained to use. They split into two shapes.

**Server-executed (category 1):** the provider runs the tool and returns the result inline. There is no Go function to write. Attach via `ProviderTools`:

~~~go
import (
    "github.com/lukemuz/luft"
    "github.com/lukemuz/luft/providers/anthropic"
    "github.com/lukemuz/luft/providers/openai"
)

// Anthropic — works against the standard Messages API.
toolset := luft.Tools(myLocalBinding).
    WithProviderTools(
        anthropic.WebSearch(anthropic.WebSearchOpts{MaxUses: 3}),
        anthropic.CodeExecution(),
    )

// OpenAI Responses — needs openai.NewResponsesProvider.
toolset := luft.Tools(myLocalBinding).
    WithProviderTools(
        openai.WebSearch(),
        openai.CodeInterpreter(openai.CodeInterpreterOpts{}),
        openai.FileSearch(openai.FileSearchOpts{VectorStoreIDs: []string{"vs_..."}}),
        openai.ImageGeneration(),
    )
~~~

The agent loop never dispatches these — the response carries provider-specific result items (`server_tool_use`, `web_search_call`, `code_interpreter_call`, …) that round-trip verbatim via `ContentBlock.Raw`.

**Provider-defined schema, you execute (category 2):** the model has been trained on the tool's name and arguments, but you supply the runtime — `bash`, `text_editor`, `computer`. The wire declaration is `{type, name}` instead of `{name, description, input_schema}`, and the dispatch flow is identical to a normal tool. Constructors return ordinary `luft.ToolBinding`s:

~~~go
bash := anthropic.BashTool(func(ctx context.Context, in json.RawMessage) (string, error) {
    // run the model's command in your sandbox of choice
})
toolset := luft.Tools(bash).Wrap(luft.WithConfirmation(promptUser))
~~~

Tools and `ProviderTool`s are tagged for one provider; passing them to a different one fails at request build with a clear error.

**OpenAI: Chat Completions vs. Responses.** Hosted tools (`web_search`, `file_search`, `code_interpreter`, `image_generation`) live on `/v1/responses`, not `/v1/chat/completions`. Use `openai.NewResponsesClientFromEnv(model)` (or build one from `openai.NewResponsesProvider`) when you want them. Plain function calling works on both endpoints; OpenAI has signaled Responses as the path forward, so prefer it for new code.

## Prompt caching

Long, stable prompts (system instructions, tool definitions, big context blocks) can be cached so subsequent turns pay a fraction of the input-token cost. Caching is provider-specific in mechanism but exposed uniformly via `luft.CacheControl`:

~~~go
// The most common pattern: cache the system prompt and the tool prefix
// for any subsequent turn within the cache window.
client, _ := luft.New(luft.Config{
    Provider:    provider,
    Model:       luft.ModelSonnet,
    SystemCache: luft.Ephemeral(),       // 5-minute TTL
})
toolset := luft.Tools(...).CacheLast(luft.Ephemeral())
~~~

Per-provider behavior:

| Provider | Caching mechanism | Honors markers? |
|---|---|---|
| `anthropic.Provider` | Explicit `cache_control` blocks (cumulative; up to 4 breakpoints) | Yes — system, tools, message blocks |
| `openrouter.Provider` | Translates markers to OpenAI-compatible typed-parts content; routed through to Anthropic backends | Yes — system, tools, message blocks |
| `openai.Provider` | Automatic for prefixes ≥1024 tokens; no field needed | Markers ignored (dropped before send) |
| `openai.ResponsesProvider` | Automatic, same as Chat Completions | Markers ignored |

Use `luft.EphemeralExtended()` for the 1-hour TTL when a prefix will be reused across long sessions.

`Usage` reports cache stats when the provider returns them — `CacheCreationTokens` (Anthropic only — tokens written to cache this turn) and `CacheReadTokens` (Anthropic and OpenAI/OpenRouter — tokens served from cache at a discount).

## MCP

`mcp` adapts Model Context Protocol tools into ordinary toolsets.

~~~go
srv, err := mcp.Connect(ctx, mcp.Config{Command: "my-mcp-server"})
defer srv.Close()
mcpTools, err := srv.Toolset(ctx)
result, err := client.Loop(ctx, system, history, mcpTools, 10)
~~~

You choose the server, inspect the tools, and pass them in.

## Streaming

~~~go
_, _, err := client.AskStream(ctx, system, history, func(delta luft.ContentBlock) {
    if delta.Type == luft.TypeText {
        fmt.Print(delta.Text)
    }
})
~~~

Use `LoopStream` or `Agent.StepStream` for streamed tool loops.

Retries can restart a stream after partial output, so callbacks may see partial text from failed attempts. Use `StreamBuffer` with `RetryConfig.OnRetry` to react and clear:

~~~go
sb := luft.NewStreamBuffer(
    func(b luft.ContentBlock) { fmt.Print(b.Text) },
    func() { fmt.Print("\n[retrying…]\n") },
)
client, _ := luft.New(luft.Config{..., Retry: luft.RetryConfig{OnRetry: sb.OnRetry}})
msg, _, err := client.AskStream(ctx, system, history, sb.OnToken)
~~~

## Observability

Every model request, response, retry, and tool dispatch goes through `Recorder` — a single-method interface that receives structured `Event`s (`turn_start`, `model_request`, `model_response`, `tool_call_start`, `tool_call_end`, `retry_attempt`, `turn_end`, `turn_error`).

~~~go
client, _ := luft.New(luft.Config{
    Provider: provider,
    Model:    luft.ModelSonnet,
    Recorder: luft.NewJSONLRecorder(os.Stdout),
})
~~~

Built-ins: `JSONLRecorder` (one JSON object per line, append-safe), `MemoryRecorder` (collect for tests or replay), `MultiRecorder` (fan out), `RecorderToSession` (attach events to a `Session` for persistence). Implement the one-method interface to bridge OpenTelemetry, Datadog, Honeycomb, or anything else.

Events carry a stable `TurnID`, a 0-based `Iter` per model call, and a monotonic `Seq`. Tool events from one parallel batch share `Iter` and order by `Seq`, so reconstruction is unambiguous. The coding CLI exposes this via `luft -log auto`, which writes a JSONL trace of the entire session — main agent and subagents — for debugging.

## Sessions

`Session` is plain data. You load it, pass `History` to a model call, and persist the result yourself.

~~~go
sess, err := store.Get(ctx, sessionID)
if errors.Is(err, luft.ErrSessionNotFound) {
    sess = &luft.Session{ID: sessionID}
} else if err != nil {
    return err
}

sess.History = append(sess.History, luft.NewUserMessage(input))
result, err := assistant.Step(ctx, sess.History)
if err != nil {
    return err
}
sess.History = result.Messages

if len(sess.History) == 1 {
    err = store.Create(ctx, sess)
} else {
    err = store.Update(ctx, sess)
}
~~~

Two built-in stores: `MemoryStore` (in-memory, concurrent-safe) and `FileStore` (one JSON file per session, atomic writes). Both implement:

~~~go
type Store interface {
    Create(ctx context.Context, session *Session) error
    Get(ctx context.Context, id string) (*Session, error)
    Update(ctx context.Context, session *Session) error
    Delete(ctx context.Context, id string) error
    List(ctx context.Context, prefix string, limit int) ([]*Session, error)
}
~~~

`Create` returns `ErrSessionExists`; `Update` returns `ErrSessionNotFound`. Both work with `errors.Is`.

## Errors and retries

~~~go
client, err := luft.New(luft.Config{
    Provider:  provider,
    Model:     luft.ModelSonnet,
    MaxTokens: 4096,
    Retry: luft.RetryConfig{
        MaxRetries:  5,
        InitialWait: time.Second,
        MaxWait:     30 * time.Second,
        OnRetry: func(attempt int, wait time.Duration) {
            log.Printf("retry %d, waiting %s", attempt, wait)
        },
    },
})
~~~

Errors are typed and work with `errors.Is` / `errors.As`: `APIError`, `ToolError`, `LoopError`, `RetryExhaustedError`, `ErrMissingTool`, `ErrMaxIter`.

Tool execution errors are soft by default: the error returns to the model as a tool result with `IsError: true`. Missing tools are configuration errors.

## Testing

The `Provider` interface is the main testing seam. You can test calls, loops, streaming, tool execution, history shape, usage, and errors without real API calls.

~~~bash
go test ./...
~~~

Good tests assert contracts (message order, tool calls, error types, callback order, usage accumulation), not exact LLM prose.

## Examples

Smaller examples in `examples/`:

~~~bash
go run ./examples/ask        # one model call
go run ./examples/pipeline   # parallel + sequential composition
go run ./examples/agent      # tool-using loop
go run ./examples/stream     # streaming
~~~

Larger runnable patterns in `examples/recipes/`:

- `02-agent-with-tools` — curated toolset, middleware, context management
- `03-repo-explainer` — sandboxed workspace tools, streaming, file-backed sessions
- `04-router-subagents` — orchestrator delegates to specialist subagents
- `05-persistent-chat` — long-running conversation with `FileStore`
- `06-parallel-pipeline` — parallel fan-out then sequential fan-in
- `07-web-service` — deploy-shaped HTTP server (JSON + SSE) with a Dockerfile

Set the relevant API key first.

See [`VISION.md`](VISION.md) for design philosophy and [`ROADMAP.md`](ROADMAP.md) for forward-looking work and what stays out of core.
