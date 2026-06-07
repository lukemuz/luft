# Vision

`luft` is a Go library for LLM calls, tools, and agent loops. It scales from a one-off model call to practical tool-using assistants without forcing the assistant shape onto the one-off call.

> Easy things easy. Hard things possible. Nothing hidden.
>
> A library, not a framework. Read the code, not the manual.

## The problem

Most agent frameworks optimize for the fully-loaded case — agents, runners, sessions, callbacks, memory services, graph runtimes. Once you accept the application model, hard things get easier. The price is that simple things pay the same setup cost as complex ones.

`luft` aims for a smoother complexity curve: a one-off call should not require an agent object, a useful assistant should not require rewriting the same glue, and a production system should not require fighting hidden ownership.

## What the promise means in code

- conversation history is plain `[]Message` data
- tools are normal Go functions
- providers implement small interfaces
- model calls, loops, persistence, and context management are explicit
- every helper is built from primitives you can inspect, bypass, or replace

The library should feel like Go: plain structs, plain functions, explicit errors, easy testing, minimal magic.

## Layers

1. **Primitives** — `Client`, `Provider`, `Message`, `Tool`, `Ask`, `Loop`, streaming, retries, typed errors.
2. **Assembly** — typed tools, schema helpers, toolsets, middleware, context managers, the `Agent` block, hooks.
3. **Recipes** — runnable patterns under `examples/recipes/`.

Each layer should be useful on its own. No layer should force concepts from a later one.

## Good convenience vs bad abstraction

`luft` is not allergic to convenience — it is allergic to loss of control.

A good helper compresses repetitive glue, has obvious behavior, exposes the primitives underneath, fails visibly, and could be rewritten by the user in ordinary Go. A bad one hides model calls, tool execution, memory mutation, or persistence, requires global registration, or owns application lifecycle.

The enemy is not the word "framework." The enemy is forcing every user into the same application model.

## The Agent block

A practical tool-using agent typically wants a client, a system prompt, a toolset, a context budget, optional summarization, an iteration cap, optional hooks, and caller-owned history. Hand-wiring that every time is tedious.

`Agent` does exactly that wiring and nothing else: trim history before each model call if a `ContextManager` is configured, run the tool-use loop, return the result. It does not own persistence, scheduling, deployment, or background work.

> An assembled primitive, not an application runner.

A one-shot autonomous task is a single `Agent.Step` call with the goal as the user message; a multi-turn conversation is one `Step` per human turn, threading history. Same struct, same method, no extra vocabulary.

## Context, tools, and subagents

**Context management** is explicit. The application owns history; the context manager returns a new `[]Message` without mutating the original; summarization happens only when configured; tool-use/tool-result integrity is preserved. The model does not decide when memory is compacted — the application does.

**Tools** are bounded, inspectable Go functions registered explicitly. **MCP** is an adapter: remote tools become ordinary `ToolBinding` values. A **subagent** is just a `ToolFunc` that calls `Loop` with its own prompt and tools — a word, not a type.

## Agent-legible by design

A good API test:

> Could a coding agent understand this from names, types, and nearby examples without reading a long manual?

That implies obvious names, local configuration, explicit data flow, no hidden state, no implicit registration, and helpers that reveal their primitives.

## Principles

1. **Start tiny.** The smallest useful program should stay small.
2. **Add power progressively.** Each capability should be adoptable independently.
3. **Keep primitives visible.** Every helper should map clearly to lower-level pieces.
4. **Make the common path short.** Boilerplate is not a virtue.
5. **Don't own the application.** Process, server, scheduler, persistence policy, deployment model, and memory policy belong to the application.
6. **Prefer boring, reliable pieces.** The win is a correct Go library for real LLM apps, not more agent vocabulary.

> Ordinary Go code, visible data flow, explicit control.
