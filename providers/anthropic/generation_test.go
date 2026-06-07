package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/lukemuz/luft"
)

func TestProvider_GenerationParams(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_1","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()

	p := &Provider{cfg: Config{APIKey: "k", HTTPClient: srv.Client(), BaseURL: srv.URL}}
	req := luft.ProviderRequest{
		Model:    luft.ModelSonnet,
		Messages: []luft.Message{luft.NewUserMessage("hi")},
		Generation: luft.GenerationParams{
			Temperature:   luft.Ptr(0.0),
			TopP:          luft.Ptr(0.5),
			TopK:          luft.Ptr(10),
			StopSequences: []string{"END"},
			Seed:          luft.Ptr(9), // unsupported by anthropic — must be dropped
		},
	}
	if _, err := p.Call(context.Background(), req); err != nil {
		t.Fatal(err)
	}

	var wire map[string]any
	if err := json.Unmarshal(gotBody, &wire); err != nil {
		t.Fatalf("request body not JSON: %v", err)
	}
	num := func(k string) (float64, bool) { v, ok := wire[k].(float64); return v, ok }
	if v, ok := num("temperature"); !ok || v != 0 {
		t.Errorf("temperature = %v (present=%v), want 0", wire["temperature"], ok)
	}
	if v, ok := num("top_p"); !ok || v != 0.5 {
		t.Errorf("top_p = %v, want 0.5", wire["top_p"])
	}
	if v, ok := num("top_k"); !ok || v != 10 {
		t.Errorf("top_k = %v, want 10", wire["top_k"])
	}
	if _, ok := wire["seed"]; ok {
		t.Errorf("seed must not be sent to anthropic, got %v", wire["seed"])
	}
	stops, ok := wire["stop_sequences"].([]any)
	if !ok || len(stops) != 1 || stops[0] != "END" {
		t.Errorf("stop_sequences = %v, want [END]", wire["stop_sequences"])
	}
}

// TestProvider_OmitsUnsetGenerationParams guards the "zero value = provider
// default" contract: an unset GenerationParams must not emit any sampling field.
func TestProvider_OmitsUnsetGenerationParams(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`))
	}))
	defer srv.Close()

	p := &Provider{cfg: Config{APIKey: "k", HTTPClient: srv.Client(), BaseURL: srv.URL}}
	if _, err := p.Call(context.Background(), luft.ProviderRequest{Model: "m", Messages: []luft.Message{luft.NewUserMessage("hi")}}); err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(gotBody, &wire); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"temperature", "top_p", "top_k", "stop_sequences", "seed"} {
		if _, ok := wire[k]; ok {
			t.Errorf("unset %q should be omitted from request, got %v", k, wire[k])
		}
	}
}

func TestProvider_ParsesRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "12")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit_error"}}`))
	}))
	defer srv.Close()

	p := &Provider{cfg: Config{APIKey: "k", HTTPClient: srv.Client(), BaseURL: srv.URL}}
	_, err := p.Call(context.Background(), luft.ProviderRequest{Model: "m"})
	var apiErr *luft.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *luft.APIError, got %T: %v", err, err)
	}
	if apiErr.RetryAfter != 12*time.Second {
		t.Errorf("RetryAfter = %v, want 12s", apiErr.RetryAfter)
	}
}
