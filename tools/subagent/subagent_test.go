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
	lastModel string // model on the most recent request — lets tier tests assert routing
}

func (f *fakeProvider) Call(_ context.Context, req luft.ProviderRequest) (luft.ProviderResponse, error) {
	f.lastSeen = req.Messages
	f.lastModel = req.Model
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
	return newClientModel(t, p, "test")
}

func newClientModel(t *testing.T, p luft.Provider, model string) *luft.Client {
	t.Helper()
	c, err := luft.New(luft.Config{Provider: p, Model: model, Retry: luft.RetryConfig{Disabled: true}})
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

// tierTiers builds a three-tier set over one provider, each tier carrying a
// distinct model string so the provider can report which tier ran.
func tierTiers(t *testing.T, p luft.Provider) []Tier {
	t.Helper()
	return []Tier{
		{Name: "powerful", Description: "strong", Client: newClientModel(t, p, "model-powerful")},
		{Name: "medium", Description: "balanced", Client: newClientModel(t, p, "model-medium")},
		{Name: "ultrafast", Description: "fast", Client: newClientModel(t, p, "model-ultrafast")},
	}
}

func TestSubagentTierDefault(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("done")}}
	b, err := New(Config{
		Name: "explore", Description: "d", System: "s", MaxIter: 5,
		Tiers: tierTiers(t, p), DefaultTier: "ultrafast",
	})
	if err != nil {
		t.Fatal(err)
	}
	// No tier passed → the default tier's model runs.
	if _, err := b.Func(context.Background(), json.RawMessage(`{"task":"look something up"}`)); err != nil {
		t.Fatal(err)
	}
	if p.lastModel != "model-ultrafast" {
		t.Errorf("default tier model = %q, want model-ultrafast", p.lastModel)
	}
}

func TestSubagentTierOverride(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("done")}}
	b, _ := New(Config{
		Name: "explore", Description: "d", System: "s", MaxIter: 5,
		Tiers: tierTiers(t, p), DefaultTier: "ultrafast",
	})
	// Caller escalates to powerful.
	if _, err := b.Func(context.Background(), json.RawMessage(`{"task":"hard","tier":"powerful"}`)); err != nil {
		t.Fatal(err)
	}
	if p.lastModel != "model-powerful" {
		t.Errorf("overridden tier model = %q, want model-powerful", p.lastModel)
	}
}

func TestSubagentUnknownTierFallsBackToDefault(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("done")}}
	b, _ := New(Config{
		Name: "explore", Description: "d", System: "s", MaxIter: 5,
		Tiers: tierTiers(t, p), DefaultTier: "medium",
	})
	if _, err := b.Func(context.Background(), json.RawMessage(`{"task":"x","tier":"nonsense"}`)); err != nil {
		t.Fatal(err)
	}
	if p.lastModel != "model-medium" {
		t.Errorf("unknown tier should fall back to default model-medium, got %q", p.lastModel)
	}
}

// schemaProps unmarshals a tool's InputSchema and returns its properties map.
func schemaProps(t *testing.T, b luft.ToolBinding) map[string]struct {
	Enum []any `json:"enum"`
} {
	t.Helper()
	var s struct {
		Properties map[string]struct {
			Enum []any `json:"enum"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(b.Tool.InputSchema, &s); err != nil {
		t.Fatalf("unmarshal schema: %v", err)
	}
	return s.Properties
}

func TestSubagentTierAdvertisedInSchema(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("done")}}
	b, _ := New(Config{
		Name: "explore", Description: "d", System: "s", MaxIter: 5,
		Tiers: tierTiers(t, p), DefaultTier: "ultrafast",
	})
	prop, ok := schemaProps(t, b)["tier"]
	if !ok {
		t.Fatal("tier property not advertised in schema")
	}
	if len(prop.Enum) != 3 {
		t.Errorf("tier enum has %d values, want 3", len(prop.Enum))
	}
}

func TestSubagentNoTiersOmitsTierProperty(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("done")}}
	b, _ := New(Config{
		Name: "worker", Description: "d", Client: newClient(t, p), System: "s", MaxIter: 5,
	})
	if _, ok := schemaProps(t, b)["tier"]; ok {
		t.Error("tier property should be absent when no Tiers are configured")
	}
}

func TestSubagentTierValidation(t *testing.T) {
	p := &fakeProvider{responses: []luft.ProviderResponse{endTurn("done")}}
	// Tiers set but DefaultTier missing.
	if _, err := New(Config{
		Name: "x", Description: "d", System: "s",
		Tiers: tierTiers(t, p),
	}); err == nil {
		t.Error("expected error when Tiers set without DefaultTier")
	}
	// DefaultTier not among the tiers.
	if _, err := New(Config{
		Name: "x", Description: "d", System: "s",
		Tiers: tierTiers(t, p), DefaultTier: "bogus",
	}); err == nil {
		t.Error("expected error when DefaultTier is not among Tiers")
	}
}
