package openai

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

func TestCompatibleCall_GenerationParams(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer srv.Close()

	p := &Provider{cfg: Config{APIKey: "k", HTTPClient: srv.Client(), BaseURL: srv.URL}}
	req := luft.ProviderRequest{
		Model:    "gpt-4o",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
		Generation: luft.GenerationParams{
			Temperature:   luft.Ptr(0.0),
			TopP:          luft.Ptr(0.5),
			TopK:          luft.Ptr(10), // unsupported by chat completions — dropped
			StopSequences: []string{"END"},
			Seed:          luft.Ptr(9),
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
	if v, ok := num("seed"); !ok || v != 9 {
		t.Errorf("seed = %v, want 9", wire["seed"])
	}
	if _, ok := wire["top_k"]; ok {
		t.Errorf("top_k must not be sent to openai, got %v", wire["top_k"])
	}
	stops, ok := wire["stop"].([]any)
	if !ok || len(stops) != 1 || stops[0] != "END" {
		t.Errorf("stop = %v, want [END]", wire["stop"])
	}
}

func TestCompatibleCall_ParsesRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"message":"unavailable","type":"server_error"}}`))
	}))
	defer srv.Close()

	p := &Provider{cfg: Config{APIKey: "k", HTTPClient: srv.Client(), BaseURL: srv.URL}}
	_, err := p.Call(context.Background(), luft.ProviderRequest{Model: "m"})
	var apiErr *luft.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *luft.APIError, got %T: %v", err, err)
	}
	if apiErr.RetryAfter != 7*time.Second {
		t.Errorf("RetryAfter = %v, want 7s", apiErr.RetryAfter)
	}
}

func TestResponses_GenerationParams(t *testing.T) {
	srv, capt := stubResponsesServer(t, `{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`)
	defer srv.Close()

	p := &ResponsesProvider{cfg: ResponsesConfig{APIKey: "k", HTTPClient: srv.Client(), BaseURL: srv.URL}}
	req := luft.ProviderRequest{
		Model:    "gpt-4o",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
		Generation: luft.GenerationParams{
			Temperature:   luft.Ptr(0.3),
			TopP:          luft.Ptr(0.8),
			StopSequences: []string{"END"}, // unsupported on responses — dropped
			Seed:          luft.Ptr(5),     // unsupported on responses — dropped
		},
	}
	if _, err := p.Call(context.Background(), req); err != nil {
		t.Fatal(err)
	}

	var wire map[string]any
	if err := json.Unmarshal(capt.body, &wire); err != nil {
		t.Fatalf("request body not JSON: %v", err)
	}
	num := func(k string) (float64, bool) { v, ok := wire[k].(float64); return v, ok }
	if v, ok := num("temperature"); !ok || v != 0.3 {
		t.Errorf("temperature = %v, want 0.3", wire["temperature"])
	}
	if v, ok := num("top_p"); !ok || v != 0.8 {
		t.Errorf("top_p = %v, want 0.8", wire["top_p"])
	}
	for _, k := range []string{"stop", "stop_sequences", "seed", "top_k"} {
		if _, ok := wire[k]; ok {
			t.Errorf("responses request must not contain %q, got %v", k, wire[k])
		}
	}
}

func TestResponses_ParsesRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "9")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow","type":"rate_limit_error"}}`))
	}))
	defer srv.Close()

	p := &ResponsesProvider{cfg: ResponsesConfig{APIKey: "k", HTTPClient: srv.Client(), BaseURL: srv.URL}}
	_, err := p.Call(context.Background(), luft.ProviderRequest{Model: "m"})
	var apiErr *luft.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *luft.APIError, got %T: %v", err, err)
	}
	if apiErr.RetryAfter != 9*time.Second {
		t.Errorf("RetryAfter = %v, want 9s", apiErr.RetryAfter)
	}
}
