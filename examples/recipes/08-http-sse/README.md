# Recipe 08: HTTP / SSE streaming chat

A `net/http` server that streams a `luft` `Agent`'s replies to the client as
Server-Sent Events, keeping per-session conversation history in an
in-memory store. No web framework, no external database.

The point is to see the luft streaming pattern end to end: load history →
append the user message → `Agent.StepStream` → write each text delta as an
SSE `data:` line → save history. The `Agent` and store live on a `Server`
struct so they're injected, not constructed in the handler — that's what
makes the handler testable with a mock provider.

## API

```
POST /chat
content-type: application/json
{ "session": "alice", "message": "say hi" }

→ 200 text/event-stream
data: Hello

data:  there!

event: error
data: <message>   ← only if the agent fails mid-stream
```

Each assistant text delta is written as `data: <text>\n\n` and flushed. If
the agent errors *before* any token is sent, the response is a plain `500`
JSON error; if it errors *after* some tokens, the status is already
committed and the failure is reported as an SSE `event: error` line.

```
GET /healthz → 200 ok
```

## Run locally

```bash
export OPENROUTER_API_KEY=...
go run ./examples/recipes/08-http-sse
```

`curl` with `-N` (no buffering) so the typewriter effect is visible:

```bash
curl -N -X POST localhost:8080/chat \
    -H 'content-type: application/json' \
    -d '{"session":"alice","message":"say hi"}'

# a second turn on the same session keeps the first turn's context:
curl -N -X POST localhost:8080/chat \
    -H 'content-type: application/json' \
    -d '{"session":"alice","message":"what did I just ask?"}'
```

## Configuration

| Env var              | Default                       | Notes                          |
|----------------------|-------------------------------|--------------------------------|
| `OPENROUTER_API_KEY` | required                      | OpenRouter API key.            |
| `MODEL`              | `anthropic/claude-sonnet-4-6` | Any OpenRouter model id.       |
| `PORT`               | `8080`                        | Listen port.                   |

## How it works

- **Read-modify-write.** `Store.Get` returns a copy of the session's
  history; the handler appends the user message, runs `Agent.StepStream`,
  and replaces the stored history with `LoopResult.Messages` — the full
  updated conversation, which is exactly the next turn's input.
- **Lazy SSE headers.** The `text/event-stream` status line is written on
  the first token, not up front. A pre-stream agent error (bad config,
  exhausted mock, immediate failure) can therefore still return `500`; a
  mid-stream error can't change the status, so it's reported as an SSE
  `event: error` line.
- **Only text deltas are streamed.** The `onToken` callback filters to
  `TypeText` blocks; tool-use and tool-result blocks are not surfaced over
  SSE.
- **No history saved on error.** If `StepStream` fails, the prior turn's
  history is left intact so the next request retries cleanly.

## Library features exercised

- `luft.Agent.StepStream` — streaming tool-aware agent loop
- `luft.ContentBlock` / `luft.TypeText` — token-delta filtering
- `luft.LoopResult.Messages` / `FinalText` — full updated history
- `luft.NewUserMessage`, `luft.TextContent` — message helpers
- `providers/openrouter.NewProviderFromEnv` — real provider wired from env
- `net/http` + `http.Flusher` — SSE without a framework

## Production notes

The in-memory store is per-process: history is lost on restart and not
shared across replicas. A single `sync.Mutex` serializes all sessions
(acceptable for a demo; for real concurrency use per-session locking or a
`luft.Store` backed by a database). Add auth, rate limiting,
`http.MaxBytesReader`, and graceful shutdown before exposing this publicly.
Recipe [`07-web-service`](../07-web-service) is the non-streaming,
`FileStore`-backed counterpart and covers the same graduation checklist.
