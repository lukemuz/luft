// Package batch provides a single tool, "batch", that runs N other tool
// calls concurrently in one model turn.
//
// Why: each round trip to the model is ~hundreds of milliseconds plus
// token cost. When the model wants to grep for a symbol AND read three
// files AND list a directory, doing that as four sequential turns is
// dominated by network latency. With batch, all four run in one turn:
//
//	{"calls": [
//	   {"name": "Grep",   "input": {"pattern": "FooBar"}},
//	   {"name": "read_file",     "input": {"path": "a.go"}},
//	   {"name": "read_file",     "input": {"path": "b.go"}},
//	   {"name": "list_directory","input": {"path": "internal/"}}
//	]}
//
// The handler dispatches all four concurrently via luft.Parallel and
// returns a single result that segments each sub-call's output with
// clearly delimited headers so the model can read them apart.
//
// The batch tool refuses to call itself (no nesting) and refuses to call
// tools whose RequiresConfirmation flag is set — those should be invoked
// directly so the human approval prompt fires. Confirmation-required
// tools are flagged in the tool description so the model knows.
package batch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/lukemuz/luft"
)

// Name is the tool name advertised to the model.
const Name = "batch"

// Config controls Tool behaviour.
type Config struct {
	// Bindings is the set of underlying tools the batch may dispatch to.
	// Pass the wrapped toolset's Bindings so middleware (timeouts, output
	// caps) applies to each sub-call.
	Bindings []luft.ToolBinding

	// MaxParallel caps how many sub-calls can run concurrently in one
	// batch invocation. 0 disables the cap (run all at once).
	MaxParallel int
}

// New returns a ToolBinding wired to dispatch to the supplied bindings.
func New(cfg Config) luft.ToolBinding {
	dispatch := make(map[string]luft.ToolFunc, len(cfg.Bindings))
	allowed := make([]string, 0, len(cfg.Bindings))
	for _, b := range cfg.Bindings {
		if b.Tool.Name == Name {
			continue // never include batch itself
		}
		if b.Meta.RequiresConfirmation {
			continue // direct invocation only
		}
		dispatch[b.Tool.Name] = b.Func
		allowed = append(allowed, b.Tool.Name)
	}

	desc := fmt.Sprintf(
		"Run 2+ independent read-only tool calls concurrently in one turn. Each direct tool call is a full LLM round trip; batch collapses N round trips into one and runs the underlying work in parallel. Default to batch whenever you have 2+ independent reads, searches, or inspections — e.g. one Grep + several read_file + a list_directory together. Don't batch when a later call's input depends on an earlier call's output (e.g. grep first, then read only the files it found). Allowed tools: %s. Confirmation-gated tools (edits, shell) are NOT allowed and must be invoked directly.",
		strings.Join(allowed, ", "),
	)

	t, fn := luft.NewTypedTool(
		Name,
		desc,
		luft.InputSchema{
			Type: "object",
			Properties: map[string]luft.SchemaProperty{
				"calls": {
					Type:        "array",
					Description: "List of tool invocations to run concurrently.",
					Items: &luft.SchemaProperty{
						Type: "object",
						Properties: map[string]luft.SchemaProperty{
							"name":  {Type: "string", Description: "Name of the underlying tool to call."},
							"input": {Type: "object", Description: "Input object for that tool, matching its schema."},
						},
						Required: []string{"name", "input"},
					},
				},
			},
			Required: []string{"calls"},
		},
		makeHandler(dispatch, cfg.MaxParallel),
	)
	return luft.ToolBinding{Tool: t, Func: fn}
}

type batchCall struct {
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type batchInput struct {
	Calls []batchCall `json:"calls"`
}

// UnmarshalJSON tolerates models that double-encode the arguments as JSON
// strings — "calls" sent as a quoted string of the array, and/or each "input"
// sent as a quoted string of the object — instead of nested JSON. It unwraps
// one layer of string-encoding wherever it finds it so a common model quirk
// doesn't fail the entire batch. Well-formed nested JSON is passed through
// untouched.
func (b *batchInput) UnmarshalJSON(data []byte) error {
	var raw struct {
		Calls json.RawMessage `json:"calls"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if len(bytes.TrimSpace(raw.Calls)) == 0 {
		return nil // leave Calls nil; the handler reports the empty case
	}
	var calls []batchCall
	if err := json.Unmarshal(unwrapJSONString(raw.Calls), &calls); err != nil {
		return err
	}
	for i := range calls {
		calls[i].Input = unwrapJSONString(calls[i].Input)
	}
	b.Calls = calls
	return nil
}

// unwrapJSONString unwraps one layer of JSON string-encoding: if raw is a JSON
// string whose decoded contents are themselves a JSON object or array, the
// inner JSON is returned; otherwise raw is returned unchanged. This rescues an
// argument a model double-encoded as a string without disturbing valid input.
func unwrapJSONString(raw json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '"' {
		return raw
	}
	var s string
	if err := json.Unmarshal(trimmed, &s); err != nil {
		return raw
	}
	if inner := bytes.TrimSpace([]byte(s)); len(inner) > 0 && (inner[0] == '{' || inner[0] == '[') {
		return json.RawMessage(inner)
	}
	return raw
}

type subResult struct {
	Name   string
	Output string
	Err    string
}

func makeHandler(dispatch map[string]luft.ToolFunc, maxParallel int) func(context.Context, batchInput) (string, error) {
	return func(ctx context.Context, in batchInput) (string, error) {
		if len(in.Calls) == 0 {
			return "", fmt.Errorf("batch: calls is empty")
		}
		results := make([]subResult, len(in.Calls))

		// Optional concurrency cap via a buffered semaphore.
		var sem chan struct{}
		if maxParallel > 0 {
			sem = make(chan struct{}, maxParallel)
		}

		var wg sync.WaitGroup
		for i, call := range in.Calls {
			i, call := i, call
			fn, ok := dispatch[call.Name]
			if !ok {
				results[i] = subResult{Name: call.Name, Err: fmt.Sprintf("tool %q not available in batch", call.Name)}
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				if sem != nil {
					sem <- struct{}{}
					defer func() { <-sem }()
				}
				out, err := fn(ctx, call.Input)
				r := subResult{Name: call.Name, Output: out}
				if err != nil {
					r.Err = err.Error()
				}
				results[i] = r
			}()
		}
		wg.Wait()

		return formatResults(results), nil
	}
}

func formatResults(results []subResult) string {
	var b strings.Builder
	for i, r := range results {
		fmt.Fprintf(&b, "=== call %d: %s", i+1, r.Name)
		if r.Err != "" {
			fmt.Fprintf(&b, " (error) ===\n%s\n", r.Err)
			continue
		}
		fmt.Fprintf(&b, " ===\n%s\n", r.Output)
	}
	return strings.TrimRight(b.String(), "\n")
}
