package openrouter

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/lukemuz/luft"
)

func TestSplitProviderRouting(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		wantModel string
		wantOpt   bool // whether a routing option is expected
	}{
		{"no suffix", "openai/gpt-oss-120b", "openai/gpt-oss-120b", false},
		{"single provider", "openai/gpt-oss-120b@cerebras", "openai/gpt-oss-120b", true},
		{"multiple providers", "openai/gpt-oss-120b@cerebras,groq", "openai/gpt-oss-120b", true},
		{"spaces trimmed", "openai/gpt-oss-120b@ cerebras , groq ", "openai/gpt-oss-120b", true},
		{"keeps :variant before @", "x/y:nitro@cerebras", "x/y:nitro", true},
		{"empty provider list is a no-op", "openai/gpt-oss-120b@", "openai/gpt-oss-120b@", false},
		{"empty base is a no-op", "@cerebras", "@cerebras", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotModel, opts := splitProviderRouting(tt.model)
			if gotModel != tt.wantModel {
				t.Errorf("model = %q, want %q", gotModel, tt.wantModel)
			}
			if (len(opts) > 0) != tt.wantOpt {
				t.Errorf("routing option present = %v, want %v", len(opts) > 0, tt.wantOpt)
			}
		})
	}
}

// TestProviderRoutingReachesWire is the end-to-end check: a model slug with an
// @provider suffix must send the bare model and a provider.order body field
// listing exactly those providers, in order.
func TestProviderRoutingReachesWire(t *testing.T) {
	tests := []struct {
		model     string
		wantModel string
		wantOrder []string
	}{
		{"openai/gpt-oss-120b@cerebras", "openai/gpt-oss-120b", []string{"cerebras"}},
		{"openai/gpt-oss-120b@cerebras,groq", "openai/gpt-oss-120b", []string{"cerebras", "groq"}},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			srv, cap := newCaptureServer(t)
			p := newProviderForTest(t, srv.URL)

			if _, err := p.Call(context.Background(), luft.ProviderRequest{
				Model:    tt.model,
				Messages: []luft.Message{luft.NewUserMessage("hi")},
			}); err != nil {
				t.Fatalf("Call: %v", err)
			}

			var sent map[string]any
			if err := json.Unmarshal(cap.body, &sent); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if sent["model"] != tt.wantModel {
				t.Errorf("model on wire = %v, want %q", sent["model"], tt.wantModel)
			}
			prov, ok := sent["provider"].(map[string]any)
			if !ok {
				t.Fatalf("provider field missing or wrong type: %v", sent["provider"])
			}
			var order []string
			for _, v := range prov["order"].([]any) {
				order = append(order, v.(string))
			}
			if !reflect.DeepEqual(order, tt.wantOrder) {
				t.Errorf("provider.order = %v, want %v", order, tt.wantOrder)
			}
		})
	}
}

// TestNoRoutingOmitsProviderField confirms a plain slug sends no provider field.
func TestNoRoutingOmitsProviderField(t *testing.T) {
	srv, cap := newCaptureServer(t)
	p := newProviderForTest(t, srv.URL)

	if _, err := p.Call(context.Background(), luft.ProviderRequest{
		Model:    "anthropic/claude-test",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
	}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	var sent map[string]any
	json.Unmarshal(cap.body, &sent)
	if _, present := sent["provider"]; present {
		t.Errorf("provider field should be omitted for an unpinned model, got %v", sent["provider"])
	}
}
