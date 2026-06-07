# Recipe 04: router with subagents-as-tools

Demonstrates `luft`'s position that **a subagent is a `ToolFunc` that
happens to call `Loop`**. There is no `SubAgent` type. The parent's
dispatch map is the routing mechanism.

## What this shows

- An orchestrator agent with two specialists exposed as tools:
  - `research(task)` — inspects the project directory using sandboxed
    workspace tools and a clock
  - `write(task)` — turns notes into a polished answer with no tools
- Cost-tiering across roles: orchestrator uses Sonnet, specialists use Haiku
- Recursion / parallelism for free: the parent issuing two subagent calls
  in one turn would run them concurrently via `runTools` (no special API)
- The parent's history stays clean: it sees only the subagents' final
  outputs, not their intermediate tool calls

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
go run ./examples/recipes/04-router-subagents -dir . "What does this project do, and what's the testing story?"
```

## Library features exercised

- `luft.New`, `luft.Config`, `luft.Client`
- `luft.Agent` (the blessed middle path)
- `luft.NewTypedTool` with a `{task: string}` schema
- `luft.Join` to compose toolsets
- `luft.Toolset` with `ToolMetadata.Source` annotations
- Built-in tools: `tools/clock`, `tools/workspace` (read-only)

## Notes

The `subagentTool` helper at the bottom of `main.go` is intentionally local
to this recipe. If the same pattern recurs across several recipes unchanged,
it will earn a place in the library; until then, it stays as ordinary Go in
the example. That's the discipline test — recipes promote into APIs only
after they've justified themselves three times.
