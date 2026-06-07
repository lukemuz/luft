# Recipe 05: persistent chat with an event log

Demonstrates `luft`'s position that **the session is plain data, not a
framework-owned `SessionService`**. Session state is plain Go data, persistence is a five-method
interface, and a `Recorder` captures intermediate turn activity into
`Session.Events` alongside the model-facing `History`.

## What this shows

- **Open-or-create** with `luft.Load` — first call creates, later calls load.
- **Read-modify-write** turn loop:
  ```go
  sess.History = append(sess.History, luft.NewUserMessage(input))
  result, err := assistant.Step(ctx, sess.History)
  // failed turn → sess.History unchanged → next attempt starts from the same place
  sess.History = result.Messages
  luft.Save(ctx, store, sess)
  ```
- **`FileStore`** as one Store implementation; the same code works against
  `NewMemoryStore()` for tests and would work against a Postgres or Redis
  store you write yourself in ~80 lines.
- **`luft.RecorderToSession(sess)`** — a `Recorder` that appends every
  model request/response, retry, and tool call to `sess.Events`. After
  `Save`, the full intra-turn activity log is on disk next to `History`.
- **`-dump`** flag formats the recorded events into a per-turn timeline so
  you can see exactly what happened inside each turn, not just the final
  assistant message.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
go run ./examples/recipes/05-persistent-chat -id alice "what's 17 * 23?"
go run ./examples/recipes/05-persistent-chat -id alice "and minus 100?"
go run ./examples/recipes/05-persistent-chat -id alice -dump
```

The session lives at `$TMPDIR/luft-chat/alice.json` — it's a plain JSON
document you can `cat`, `jq`, diff, or hand-edit.

## Library features exercised

- `luft.Session`, `luft.Store`, `luft.FileStore`
- `luft.Load`, `luft.Save` (open-or-create / upsert convenience)
- `luft.Recorder` interface + `luft.RecorderToSession`
- `luft.Event` and the `EventType` constants
- `luft.Agent` driving the turn
- Built-in tools: `tools/math`

## Notes

- **Failure atomicity by default.** A failed `Step` returns an error and
  leaves `sess` untouched, because nothing was assigned. Retrying the turn
  is just calling `handleTurn` again. This is a property of the code you
  can see, not of a service implementation.
- **Recording is best-effort.** Recorders must be fast and non-blocking;
  errors from `JSONLRecorder.Write` are silently dropped so a flaky log
  destination cannot fail a turn. If you need stronger guarantees, wrap
  your own writer.
- **`Session.Events` does not feed back to the model.** It's an audit
  trail. The model only ever sees `Session.History`. This keeps the
  contract with the LLM unambiguous: history is what the model sees,
  events are what you saw the model do.
