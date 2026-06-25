package subagent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
)

// fakeProvider returns canned responses in order, repeating the last one.
// It records the history it was called with so tests can assert on trimming.
type fakeProvider struct {
	responses []luft.ProviderResponse
	calls     int
	lastSeen  []luft.Message
}

func (f *fakeProvider) Call(_ context.Context, req luft.ProviderRequest) (luft.ProviderResponse, error) {
	f.lastSeen = req.Messages
	i := f.calls
	if i >= len(f.responses) {
		i = len(f.responses) - 1
	}
	f.calls++
	return f.responses[i], nil
}

func (f *fakeProvider) Stream(ctx context.Context, req luft.ProviderRequest, _ func(luft.ContentBlock)) (luft.ProviderResponse, error) {
	return f.Call(ctx, req)
}

func newClient(t *testing.T, p luft.Provider) *luft.Client {
	t.Helper()
	c, err := luft.New(luft.Config{Provider: p, Model: "test", Retry: luft.RetryConfig{Disabled: true}})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func endTurn(text string) luft.ProviderResponse {
	return luft.ProviderResponse{
		Content:    []luft.ContentBlock{{Type: luft.TypeText, Text: text}},
		StopReason: "end_turn",
	}
}

func TestSubagentReturnsFinalText(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("all done")}}
	b, err := New(Config{
		Name:        "worker",
		Description: "does work",
		Client:      newClient(t, p),
		System:      "you are a worker",
		MaxIter:     5,
	})
	if err != nil {
		t.Fatal(err)
	}

	out, err := b.Func(context.Background(), json.RawMessage(`{"task":"do the thing"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "all done" {
		t.Errorf("result = %q, want %q", out, "all done")
	}
	// The task must reach the model as the opening user message.
	if got := luft.TextContent(p.lastSeen[0]); !strings.Contains(got, "do the thing") {
		t.Errorf("task not passed to model; first message = %q", got)
	}
}

func TestSubagentEmptyTaskRejected(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("unused")}}
	b, _ := New(Config{
		Name: "worker", Description: "d", Client: newClient(t, p), System: "s", MaxIter: 5,
	})
	if _, err := b.Func(context.Background(), json.RawMessage(`{"task":""}`)); err == nil {
		t.Error("expected error for empty task")
	}
	if p.calls != 0 {
		t.Errorf("model should not be called for an empty task; calls = %d", p.calls)
	}
}

// TestSubagentHonorsContext proves the subagent runs through the Agent block:
// a tiny token budget plus a Summarizer must invoke that summarizer (raw Loop,
// the old path, had no context management and would never call it).
func TestSubagentHonorsContext(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("done")}}
	summarized := 0
	b, err := New(Config{
		Name:        "worker",
		Description: "d",
		Client:      newClient(t, p),
		System:      "s",
		MaxIter:     5,
		Context: luft.ContextManager{
			MaxTokens: 1, // anything real exceeds this, forcing a trim
			Summarizer: func(_ context.Context, trimmed []luft.Message) (string, error) {
				summarized++
				return "SUMMARY", nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := b.Func(context.Background(), json.RawMessage(`{"task":"some long enough task"}`)); err != nil {
		t.Fatal(err)
	}
	if summarized == 0 {
		t.Error("expected the context summarizer to run — subagent is not using the Agent block")
	}
}

// TestSubagentNoContextSkipsSummarizer is the control: with a zero-value
// Context, the trim step is a no-op and nothing is summarized.
func TestSubagentNoContextSkipsSummarizer(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("done")}}
	summarized := 0
	b, _ := New(Config{
		Name: "worker", Description: "d", Client: newClient(t, p), System: "s", MaxIter: 5,
		Context: luft.ContextManager{
			// MaxTokens 0 → trimming disabled entirely.
			Summarizer: func(_ context.Context, _ []luft.Message) (string, error) {
				summarized++
				return "SUMMARY", nil
			},
		},
	})
	if _, err := b.Func(context.Background(), json.RawMessage(`{"task":"x"}`)); err != nil {
		t.Fatal(err)
	}
	if summarized != 0 {
		t.Errorf("summarizer ran %d times with MaxTokens=0; want 0", summarized)
	}
}
