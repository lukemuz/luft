# Roadmap

Forward-looking work only. For what already ships, see [`README.md`](README.md). For the philosophy, see [`VISION.md`](VISION.md).

## North star

> Easy things easy. Hard things possible. Nothing hidden.
>
> A library, not a framework. Read the code, not the manual.

Future work should make the common path shorter without introducing hidden model calls, hidden tool execution, hidden persistence, global registries, or framework-owned control flow.

## Planned

- **Durable tool execution.** Opt-in middleware that uses tool-use IDs as idempotency keys against a small `ToolResultStore`, so side-effectful tools survive crash-resume. Composes with existing middleware.
- **Observability hooks.** Expand `Hooks` toward request/response/tool/error events, plus an `agent/otel` subpackage so the core stays free of OpenTelemetry.
- **Testing helpers.** Tiny mock and scripted providers, plus assertions for history shape, tool calls, and usage. The `Provider` interface stays the main testing seam.
- **HTTP/SSE example.** A `net/http` handler that loads history, calls `Agent.StepStream`, writes SSE events, and saves. No web framework, no runner.
- **Evaluation helpers.** Small offline regression helpers. No hosted dashboards, no required databases.

## Maybe, maybe not

- **Lightweight multi-agent helpers.** Routing, critique, fan-out — only if the shape really is just functions over existing primitives.
- **Cross-session memory.** Embeddings and vector stores belong in a separate package, not the core loop.

## Shelved: the CLI as an integration seam

The CLI ships a non-interactive / pipe mode (`-p`, positional args, piped stdin) with `--json` output and documented exit codes. It works and is tested; it stays. But the broader framing it was built toward — "luft as a scriptable shell / the integration seam for non-Go consumers (IDE plugins, review bots, other agents calling luft as a subprocess)" — is **not being pursued actively.** It's shelved until a concrete consumer asks for it.

What's there today (kept, at risk of some bloat):
- `cmd/luft/print.go` — `isNonInteractive`, `assemblePrompt`, `runPrint`; one `agent.StepStream` call, streamed text or one JSON object on stdout, exit codes 0/1/2/130, auto-approves confirmations, persists the session so `-resume` chains work.
- `--json` shape: `{session_id, result, usage, tool_calls, error, exit_code}`.

What's deliberately **not** built (the gap between "one-shot filter" and "true headless"):
- No resident / server mode. Each invocation pays full cold start (provider init, MCP connect, memory load, tool assembly). A `luft serve` over `net/http` — the shape `examples/recipes/08-http-sse` already proves the library supports — would fill this, but it's out of scope until a real integrator needs a warm, multi-prompt process rather than a stateless subprocess.
- No stable versioned integration contract documented as such. The `--json` shape is an output, not a promise; treat it as unstable until someone builds against it and we commit.

Resume this thread only when a concrete non-Go consumer (IDE extension, bot, orchestrating agent) shows up. Until then the interactive REPL remains the main event.

## Non-goals

The core will not become a graph executor, visual workflow builder, no-code configurator, managed deployment platform, hidden scheduler, vector database, global tool registry, autonomous background runtime, or framework-owned runner. Higher-level systems can be built on top.
