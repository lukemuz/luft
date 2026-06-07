# Quickstart

`luft` is a small Go library for LLM calls, tools, and agent loops. This guide gets you from zero to a model call to a tool loop. For the full reference see [`README.md`](README.md).

## Install

~~~bash
go get github.com/lukemuz/luft
export ANTHROPIC_API_KEY=sk-ant-...
~~~

OpenAI and OpenRouter work the same way; see the README.

## 1. One model call

~~~go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/lukemuz/luft"
    "github.com/lukemuz/luft/providers/anthropic"
)

func main() {
    ctx := context.Background()

    client, err := anthropic.NewClientFromEnv(luft.ModelSonnet)
    if err != nil {
        log.Fatal(err)
    }

    history := []luft.Message{
        luft.NewUserMessage("Give me three practical ideas for using LLMs in a Go service."),
    }

    reply, usage, err := client.Ask(ctx, "You are concise.", history)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(luft.TextContent(reply))
    fmt.Printf("(%d in / %d out tokens)\n", usage.InputTokens, usage.OutputTokens)
}
~~~

`history` is just data. There is no hidden session.

## 2. Continue the conversation

`Ask` does not mutate history. Append the reply yourself:

~~~go
history = append(history, reply)
history = append(history, luft.NewUserMessage("Pick the most practical idea."))
reply, _, err = client.Ask(ctx, "You are concise.", history)
~~~

That is the core data model: `[]luft.Message` in, `luft.Message` out.

## 3. Add a tool

A tool is a model-facing definition plus a Go function. The built-in clock tool is the smallest example:

~~~go
import "github.com/lukemuz/luft/tools/clock"

clockTool := clock.New()
toolset := clockTool.Toolset()

history := []luft.Message{
    luft.NewUserMessage("What time is it? Then suggest one thing to work on next."),
}

result, err := client.Loop(ctx, "Use tools when they help.", history, toolset, 5)
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.FinalText())
~~~

`Loop` calls the model, runs requested tools, appends results, and repeats until a final answer or the iteration limit. You still own the resulting history via `result.Messages`.

## 4. Define your own typed tool

~~~go
type CalculatorInput struct {
    Operation string  `json:"operation"`
    A         float64 `json:"a"`
    B         float64 `json:"b"`
}

calc, calcFn := luft.NewTypedTool(
    "calculator",
    "Do basic arithmetic.",
    luft.Object(
        luft.String("operation", "add, subtract, multiply", luft.Required(), luft.Enum("add", "subtract", "multiply")),
        luft.Number("a", "First number", luft.Required()),
        luft.Number("b", "Second number", luft.Required()),
    ),
    func(ctx context.Context, in CalculatorInput) (string, error) {
        switch in.Operation {
        case "add":      return fmt.Sprintf("%g", in.A+in.B), nil
        case "subtract": return fmt.Sprintf("%g", in.A-in.B), nil
        case "multiply": return fmt.Sprintf("%g", in.A*in.B), nil
        }
        return "", fmt.Errorf("unknown op: %s", in.Operation)
    },
)
~~~

Use it in a `Toolset`:

~~~go
tools := luft.Tools(luft.Bind(calc, calcFn))
~~~

The model sees the schema; your handler receives a typed Go value; dispatch is a normal map. No reflection registry, no hidden runtime.

### Arrays and nested objects

Use `luft.Array` for list parameters and `luft.ObjectOf` for nested object shapes:

~~~go
schema := luft.Object(
    luft.String("reasoning", "Why these subtasks cover the question"),
    luft.Array("subtasks", "List of sub-questions",
        luft.ObjectOf(
            luft.String("question", "Sub-question to research", luft.Required()),
            luft.String("rationale", "Why this matters"),
        ),
        luft.Required()),
)
~~~

`Array` takes a `SchemaProperty` for its element type — pass `ObjectOf(...)` for arrays of objects, or a primitive like `SchemaProperty{Type: "string"}` for arrays of scalars.

### Hand-rolled schemas (escape hatch)

The builders cover ~95% of tool schemas. For the rest — `oneOf`, `$ref`, regex `pattern`, recursive types — `luft.Tool.InputSchema` is `json.RawMessage`, so any valid JSON Schema works:

~~~go
const mySchema = `{
  "type": "object",
  "properties": {
    "value": {"oneOf": [{"type": "string"}, {"type": "number"}]}
  },
  "required": ["value"]
}`

tool := luft.Tool{
    Name:        "store",
    Description: "Store a string or number",
    InputSchema: json.RawMessage(mySchema),
}
~~~

You can mix this with `luft.TypedToolFunc` for the dispatch side; only the schema needs to be hand-rolled.

## 5. Compose built-ins and middleware

~~~go
ws, err := workspace.NewReadOnly(workspace.Config{Root: "."})
if err != nil {
    log.Fatal(err)
}

tools := luft.MustJoin(clockTool.Toolset(), ws.Toolset()).Wrap(
    luft.WithTimeout(5*time.Second),
    luft.WithResultLimit(20_000),
)
~~~

Use `workspace.NewReadOnly` for safe filesystem reads. `workspace.New` includes `edit_file` — wrap it with `luft.WithConfirmation` before letting writes run.

## 6. Use Agent when the glue repeats

`Agent` bundles a client, system prompt, toolset, context manager, and iteration cap.

~~~go
a := luft.Agent{
    Client:  client,
    System:  "You are a helpful assistant.",
    Tools:   tools,
    Context: luft.ContextManager{MaxTokens: 8000, KeepRecent: 20},
    MaxIter: 10,
}

// One-shot autonomous task: pass a single user message with the goal.
result, err := a.Step(ctx, []luft.Message{luft.NewUserMessage("do the thing")})

// Multi-turn: call Step once per human turn and thread history.
result, err = a.Step(ctx, history)
history = result.Messages
~~~

`Agent.Step` trims history once up front and again before every model call inside the loop (when a `ContextManager` is configured), so long autonomous runs do not silently blow the context window. The primitives `Loop` and `ContextManager.Trim` remain available if you want a different policy.

No persistence, runner, scheduler, or hidden lifecycle.

## Ask vs Loop vs Extract vs Agent

| Need | Use |
|---|---|
| One text response | `Ask` |
| Streaming response | `AskStream` |
| Model can call tools | `Loop` |
| Streaming tool loop | `LoopStream` |
| Typed value back (with or without tools) | `Extract[T]` |
| Repeated agent glue | `Agent.Step` / `Agent.StepStream` |
| Independent fan-out | `Parallel` |

## Next steps

- [`README.md`](README.md) for providers, streaming, retries, errors, sessions, MCP, and testing.
- [`VISION.md`](VISION.md) for the design philosophy.
- [`cmd/luft/README.md`](cmd/luft/README.md) for the CLI coding agent built on the library.
- [`examples/recipes/`](examples/recipes/) for runnable patterns.
