# Recipe 06: parallel-then-sequential pipeline

Demonstrates `luft`'s position that **a workflow is just Go**. Two steps
run concurrently via `luft.Parallel`; a third step consumes their outputs
in a single follow-up call. There is no `Pipeline`, `ParallelAgent`, or
`SequentialAgent` type — fan-out is a goroutine helper, fan-in is a
function call, and the data flow is visible top-to-bottom in `main.go`.

## What this shows

- `luft.Parallel` running two `StepFunc[string]` concurrently and
  returning index-aligned `[]Result[string]`.
- Per-step error handling: each `Result` carries its own `Err`, so the
  caller decides whether one failure aborts the pipeline or degrades it.
- A sequential step (the comparison call) reading both parallel outputs
  via ordinary local variables — no shared blackboard, no event bus.
- The same `*luft.Client` reused across all three calls; cost-tiering or
  per-step models would be a one-line change.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
go run ./examples/recipes/06-parallel-pipeline
```

## Library features exercised

- `luft.Parallel`, `luft.StepFunc`, `luft.Result`
- `luft.Client.Ask` for single-shot calls (no tool loop)
- `luft.NewUserMessage`, `luft.TextContent`

## Notes

- **No cancellation policy is imposed.** `Parallel` waits for every step
  to finish; if you want fail-fast, derive a context with `cancel` and
  call it after the first error result.
- **The step type is generic.** `Parallel[T]` works for any `T` —
  `string` here, but it could be a struct, a `[]Message`, or whatever the
  next stage needs. The only constraint is that all steps in a single
  `Parallel` call return the same type.
- **No promotion to a `Pipeline` type yet.** Two parallel calls plus a
  follow-up is small enough that the control flow tells the story.
  If a recurring pattern justifies it, it would land as a recipe helper
  before earning a place in the library.
