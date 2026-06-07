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
- **Extended generation controls.** Thread temperature, top-p, and stop sequences through `ProviderRequest`. Zero values keep provider defaults.
- **Testing helpers.** Tiny mock and scripted providers, plus assertions for history shape, tool calls, and usage. The `Provider` interface stays the main testing seam.
- **HTTP/SSE example.** A `net/http` handler that loads history, calls `Agent.StepStream`, writes SSE events, and saves. No web framework, no runner.
- **Evaluation helpers.** Small offline regression helpers. No hosted dashboards, no required databases.

## Maybe, maybe not

- **Lightweight multi-agent helpers.** Routing, critique, fan-out — only if the shape really is just functions over existing primitives.
- **Cross-session memory.** Embeddings and vector stores belong in a separate package, not the core loop.

## Non-goals

The core will not become a graph executor, visual workflow builder, no-code configurator, managed deployment platform, hidden scheduler, vector database, global tool registry, autonomous background runtime, or framework-owned runner. Higher-level systems can be built on top.
