package luft

import (
	"context"
	"testing"
)

func TestPtr(t *testing.T) {
	if p := Ptr(0.0); p == nil || *p != 0.0 {
		t.Fatalf("Ptr(0.0) = %v, want pointer to 0.0", p)
	}
	if p := Ptr(42); p == nil || *p != 42 {
		t.Fatalf("Ptr(42) = %v, want pointer to 42", p)
	}
	if p := Ptr("x"); p == nil || *p != "x" {
		t.Fatalf("Ptr(\"x\") = %v, want pointer to \"x\"", p)
	}
}

// TestGenerationParamsThreadToProvider verifies that Config.Generation reaches
// the ProviderRequest on both the Ask path and the Loop path. spyProvider only
// overrides Call, which both Ask and (non-streaming) Loop use.
func TestGenerationParamsThreadToProvider(t *testing.T) {
	gen := GenerationParams{
		Temperature:   Ptr(0.0),
		TopP:          Ptr(0.9),
		TopK:          Ptr(40),
		StopSequences: []string{"STOP"},
		Seed:          Ptr(7),
	}

	var got []ProviderRequest
	spy := &spyProvider{
		inner:  newTestProvider(),
		onCall: func(req ProviderRequest) { got = append(got, req) },
	}
	client, err := New(Config{Provider: spy, Model: ModelSonnet, Generation: gen})
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	history := []Message{NewUserMessage("hi")}
	if _, _, err := client.Ask(ctx, "sys", history); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if _, err := client.Loop(ctx, "sys", history, Tools(), 1); err != nil {
		t.Fatalf("Loop: %v", err)
	}

	if len(got) != 2 {
		t.Fatalf("provider Call count = %d, want 2 (Ask + Loop)", len(got))
	}
	for i, req := range got {
		g := req.Generation
		if g.Temperature == nil || *g.Temperature != 0.0 {
			t.Errorf("call %d Temperature = %v, want 0.0", i, g.Temperature)
		}
		if g.TopP == nil || *g.TopP != 0.9 {
			t.Errorf("call %d TopP = %v, want 0.9", i, g.TopP)
		}
		if g.TopK == nil || *g.TopK != 40 {
			t.Errorf("call %d TopK = %v, want 40", i, g.TopK)
		}
		if len(g.StopSequences) != 1 || g.StopSequences[0] != "STOP" {
			t.Errorf("call %d StopSequences = %v, want [STOP]", i, g.StopSequences)
		}
		if g.Seed == nil || *g.Seed != 7 {
			t.Errorf("call %d Seed = %v, want 7", i, g.Seed)
		}
	}
}

func TestGenerationParamsZeroByDefault(t *testing.T) {
	var got ProviderRequest
	spy := &spyProvider{
		inner:  newTestProvider(),
		onCall: func(req ProviderRequest) { got = req },
	}
	client, err := New(Config{Provider: spy, Model: ModelSonnet})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := client.Ask(context.Background(), "", []Message{NewUserMessage("hi")}); err != nil {
		t.Fatal(err)
	}
	g := got.Generation
	if g.Temperature != nil || g.TopP != nil || g.TopK != nil || g.Seed != nil || g.StopSequences != nil {
		t.Errorf("default Generation should be all-unset, got %+v", g)
	}
}
