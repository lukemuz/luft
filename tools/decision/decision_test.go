package decision

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
)

func TestRequest(t *testing.T) {
	prompt := "Choose A or B.\nConstraints: \"fast\", cost < $10."
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer test-key" || r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected request: %s %s, headers %v", r.Method, r.URL.Path, r.Header)
		}
		var body struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string
				Content string
			} `json:"messages"`
			Tools  json.RawMessage `json:"tools"`
			Stream bool            `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body.Model != "test-model" || len(body.Messages) != 1 || body.Messages[0].Role != "user" || body.Messages[0].Content != prompt || len(body.Tools) != 0 || body.Stream {
			t.Errorf("unexpected body: %+v", body)
		}
		w.Write([]byte(`{"choices":[{"message":{"content":"Choose B."},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()
	b, err := New(Config{Endpoint: server.URL + "/v1/chat/completions", Model: "test-model", APIKey: "test-key"})
	if err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal(map[string]string{"prompt": prompt})
	got, err := b.Func(context.Background(), input)
	if err != nil || got != "Choose B." {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestFailures(t *testing.T) {
	for _, tc := range []struct {
		name       string
		status     int
		body, want string
	}{
		{"unauthorized", 401, `{"error":{"message":"invalid key","type":"auth"}}`, "invalid key"},
		{"rate limit", 429, `{"error":{"message":"slow down"}}`, "slow down"},
		{"invalid JSON", 200, `not JSON`, "decode"},
		{"empty", 200, `{"choices":[]}`, "no text"},
		{"truncated", 200, `{"choices":[{"message":{"content":"Partial"},"finish_reason":"length"}]}`, "output limit"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(tc.status); w.Write([]byte(tc.body)) }))
			defer s.Close()
			b, err := New(Config{Endpoint: s.URL, Model: "m", APIKey: "k"})
			if err != nil {
				t.Fatal(err)
			}
			_, err = b.Func(context.Background(), json.RawMessage(`{"prompt":"decide"}`))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want %q", err, tc.want)
			}
		})
	}
}

func TestValidationAndCancellation(t *testing.T) {
	for _, cfg := range []Config{
		{}, {Model: "m"}, {APIKey: "k"},
		{Model: "m", APIKey: "k", Endpoint: ":bad"},
		{Model: "m", APIKey: "k", Endpoint: "http://example.com"},
		{Model: "m", APIKey: "k", Endpoint: "https://user:password@example.com"},
	} {
		if _, err := New(cfg); err == nil {
			t.Errorf("expected config error for %+v", cfg)
		}
	}
	b, err := New(Config{Model: "m", APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{`{}`, `null`, `{"prompt":"  "}`, `{"prompt":42}`} {
		if _, err := b.Func(context.Background(), json.RawMessage(raw)); err == nil {
			t.Errorf("accepted %s", raw)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = b.Func(ctx, json.RawMessage(`{"prompt":"decide"}`))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want cancellation", err)
	}
	if !b.Meta.ReadOnly || b.Tool.Name != "decision_model" {
		t.Fatalf("unexpected binding: %+v", b)
	}
	var schema luft.InputSchema
	if err := json.Unmarshal(b.Tool.InputSchema, &schema); err != nil || len(schema.Required) != 1 || schema.Required[0] != "prompt" {
		t.Fatalf("unexpected schema: %s", b.Tool.InputSchema)
	}
}
