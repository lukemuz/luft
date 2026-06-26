package luft

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// newCounterTool returns a ToolFunc whose result is its call count, plus a
// pointer to that count so tests can assert how many times the underlying tool
// actually executed.
func newCounterTool() (ToolFunc, *int) {
	var n int
	fn := func(ctx context.Context, input json.RawMessage) (string, error) {
		n++
		return "call-" + inputStr(input), nil
	}
	return fn, &n
}

func inputStr(input json.RawMessage) string {
	if len(input) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(input, &s); err == nil {
		return s
	}
	return string(input)
}

// binding named "dur" wrapping fn; used as the Middleware input.
func durBinding(fn ToolFunc) ToolBinding {
	return ToolBinding{Tool: Tool{Name: "dur"}, Func: fn}
}

// call runs a wrapped ToolFunc on raw JSON input.
func call(t *testing.T, fn ToolFunc, input string) (string, error) {
	t.Helper()
	return fn(context.Background(), json.RawMessage(input))
}

func TestDurability_IdenticalCallRunsOnce(t *testing.T) {
	fn, count := newCounterTool()
	store := NewMemoryToolResultStore()
	wrapped := WithDurability(store)(durBinding(fn))

	in := `"x"`
	out1, err := call(t, wrapped, in)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	out2, err := call(t, wrapped, in)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}

	if out1 != out2 {
		t.Fatalf("cached output mismatch: %q vs %q", out1, out2)
	}
	if *count != 1 {
		t.Fatalf("underlying tool executed %d times, want 1", *count)
	}
}

func TestDurability_DifferentInputsExecuteIndependently(t *testing.T) {
	fn, count := newCounterTool()
	store := NewMemoryToolResultStore()
	wrapped := WithDurability(store)(durBinding(fn))

	out1, err := call(t, wrapped, `"a"`)
	if err != nil {
		t.Fatalf("call a: %v", err)
	}
	out2, err := call(t, wrapped, `"b"`)
	if err != nil {
		t.Fatalf("call b: %v", err)
	}

	if out1 == out2 {
		t.Fatalf("expected distinct outputs, both %q", out1)
	}
	if *count != 2 {
		t.Fatalf("underlying tool executed %d times, want 2", *count)
	}
}

func TestDurability_ErrorNotCached(t *testing.T) {
	var n int
	failing := func(ctx context.Context, input json.RawMessage) (string, error) {
		n++
		if n == 1 {
			return "", errors.New("transient boom")
		}
		return "recovered", nil
	}
	store := NewMemoryToolResultStore()
	wrapped := WithDurability(store)(durBinding(failing))

	in := `"err"`
	_, err := call(t, wrapped, in)
	if err == nil {
		t.Fatal("first call: want error, got nil")
	}
	out2, err := call(t, wrapped, in)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}

	if out2 != "recovered" {
		t.Fatalf("second call output = %q, want %q", out2, "recovered")
	}
	if n != 2 {
		t.Fatalf("underlying tool executed %d times, want 2 (error must not be cached)", n)
	}
}

// recordingMiddleware counts how many times the INNER (wrapped) stack runs,
// so tests can verify WithDurability short-circuits before it on a hit.
func recordingMiddleware(count *int) Middleware {
	return func(b ToolBinding) ToolFunc {
		return func(ctx context.Context, input json.RawMessage) (string, error) {
			*count++
			return b.Func(ctx, input)
		}
	}
}

func TestDurability_ComposesWithAnotherMiddleware(t *testing.T) {
	fn, execCount := newCounterTool()
	store := NewMemoryToolResultStore()
	binding := durBinding(fn)

	// Compose manually with durability outermost: the recording middleware
	// wraps the raw func, then WithDurability wraps that.
	var innerRuns int
	innerFunc := recordingMiddleware(&innerRuns)(ToolBinding{Tool: binding.Tool, Func: binding.Func})
	outerFunc := WithDurability(store)(ToolBinding{Tool: binding.Tool, Func: innerFunc})

	in := `"stack"`
	out1, err := call(t, outerFunc, in)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	out2, err := call(t, outerFunc, in)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}

	if out1 != out2 {
		t.Fatalf("cached output mismatch: %q vs %q", out1, out2)
	}
	// Underlying tool ran once; inner middleware saw exactly one call (the
	// cache hit on the second call bypassed it entirely).
	if *execCount != 1 {
		t.Fatalf("underlying tool executed %d times, want 1", *execCount)
	}
	if innerRuns != 1 {
		t.Fatalf("inner middleware ran %d times, want 1 (durability must short-circuit on hit)", innerRuns)
	}
}

// Verify the ordering rule holds through the real Wrap path: listing
// WithDurability first makes it outermost, so a second identical call is served
// from the store without touching the inner middleware or the tool.
func TestDurability_WrapOrderOutermost(t *testing.T) {
	fn, execCount := newCounterTool()
	binding := durBinding(fn)
	ts := Tools(binding)

	var innerRuns int
	wrapped := ts.Wrap(WithDurability(NewMemoryToolResultStore()), recordingMiddleware(&innerRuns))
	dispatch := wrapped.Dispatch()

	in := `"wrap"`
	out1, err := dispatch["dur"](context.Background(), json.RawMessage(in))
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	out2, err := dispatch["dur"](context.Background(), json.RawMessage(in))
	if err != nil {
		t.Fatalf("second call: %v", err)
	}

	if out1 != out2 {
		t.Fatalf("cached output mismatch: %q vs %q", out1, out2)
	}
	if *execCount != 1 {
		t.Fatalf("underlying tool executed %d times, want 1", *execCount)
	}
	if innerRuns != 1 {
		t.Fatalf("inner middleware ran %d times, want 1", innerRuns)
	}
	// Sanity: the recorded output carries the call marker so we know it came
	// from the real func on the first call and the store on the second.
	if !strings.HasPrefix(out1, "call-") {
		t.Fatalf("unexpected output %q", out1)
	}
}
