package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
)

func bindFunc(name string, fn luft.ToolFunc) luft.ToolBinding {
	return luft.ToolBinding{Tool: luft.Tool{Name: name}, Func: fn}
}

func TestDedupReadsSuppressesIdenticalRepeat(t *testing.T) {
	big := strings.Repeat("x", 100)
	calls := 0
	wrapped := dedupReads(10)(bindFunc("read_file", func(context.Context, json.RawMessage) (string, error) {
		calls++
		return big, nil
	}))

	in := json.RawMessage(`{"path":"a.go"}`)
	out1, _ := wrapped(context.Background(), in)
	if out1 != big {
		t.Fatalf("first read should return full content, got %q", out1)
	}
	out2, _ := wrapped(context.Background(), in)
	if !strings.Contains(out2, "identical to your earlier") {
		t.Fatalf("second identical read should be a dedup notice, got %q", out2)
	}
	if calls != 2 {
		t.Fatalf("tool should still run both times (to detect changes), ran %d", calls)
	}
}

func TestDedupReadsReturnsFreshOnChange(t *testing.T) {
	outputs := []string{strings.Repeat("a", 50), strings.Repeat("b", 50)}
	i := 0
	wrapped := dedupReads(10)(bindFunc("read_file", func(context.Context, json.RawMessage) (string, error) {
		o := outputs[i]
		i++
		return o, nil
	}))

	in := json.RawMessage(`{"path":"a.go"}`)
	if out, _ := wrapped(context.Background(), in); out != outputs[0] {
		t.Fatalf("first read: %q", out)
	}
	// Same input, but the file changed → different output → full content, no notice.
	out, _ := wrapped(context.Background(), in)
	if out != outputs[1] {
		t.Fatalf("changed file should return fresh content, got %q", out)
	}
}

func TestDedupReadsIgnoresSmallAndDistinctInputs(t *testing.T) {
	wrapped := dedupReads(1000)(bindFunc("read_file", func(_ context.Context, in json.RawMessage) (string, error) {
		return "short", nil // below minBytes, never deduped
	}))
	in := json.RawMessage(`{"path":"a.go"}`)
	if out, _ := wrapped(context.Background(), in); out != "short" {
		t.Fatalf("small output should pass through, got %q", out)
	}
	if out, _ := wrapped(context.Background(), in); out != "short" {
		t.Fatalf("small output should never become a notice, got %q", out)
	}
}

func TestTruncateMiddleKeepsHeadAndTail(t *testing.T) {
	s := "HEAD" + strings.Repeat("m", 1000) + "TAIL"
	got := truncateMiddle(s, 20)
	if !strings.HasPrefix(got, "HEAD") {
		t.Fatalf("should keep head: %q", got)
	}
	if !strings.HasSuffix(got, "TAIL") {
		t.Fatalf("should keep tail: %q", got)
	}
	if !strings.Contains(got, "elided") {
		t.Fatalf("should note elision: %q", got)
	}
	if len(got) >= len(s) {
		t.Fatalf("should be shorter than input")
	}
}

func TestTruncateMiddlePassesThroughSmall(t *testing.T) {
	if got := truncateMiddle("tiny", 100); got != "tiny" {
		t.Fatalf("small input should pass through unchanged, got %q", got)
	}
}
