package vertex

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/lukemuz/luft"
)

func TestBuildGenerateRequest_GenerationParams(t *testing.T) {
	req := luft.ProviderRequest{
		Model:     "gemini-1.5-pro",
		MaxTokens: 100,
		Messages:  []luft.Message{luft.NewUserMessage("hi")},
		Generation: luft.GenerationParams{
			Temperature:   luft.Ptr(0.0),
			TopP:          luft.Ptr(0.5),
			TopK:          luft.Ptr(10),
			StopSequences: []string{"END"},
			Seed:          luft.Ptr(7),
		},
	}
	wire, err := buildGenerateRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	gc := wire.GenerationConfig
	if gc == nil {
		t.Fatal("generationConfig is nil")
	}
	if gc.Temperature == nil || *gc.Temperature != 0.0 {
		t.Errorf("temperature = %v, want 0", gc.Temperature)
	}
	if gc.TopP == nil || *gc.TopP != 0.5 {
		t.Errorf("topP = %v, want 0.5", gc.TopP)
	}
	if gc.TopK == nil || *gc.TopK != 10 {
		t.Errorf("topK = %v, want 10", gc.TopK)
	}
	if gc.Seed == nil || *gc.Seed != 7 {
		t.Errorf("seed = %v, want 7", gc.Seed)
	}
	if len(gc.StopSequences) != 1 || gc.StopSequences[0] != "END" {
		t.Errorf("stopSequences = %v, want [END]", gc.StopSequences)
	}
	if gc.MaxOutputTokens != 100 {
		t.Errorf("maxOutputTokens = %d, want 100", gc.MaxOutputTokens)
	}
}

// TestBuildGenerateRequest_NoConfigWhenEmpty guards the "zero value = provider
// default" contract: with nothing set, generationConfig is omitted.
func TestBuildGenerateRequest_NoConfigWhenEmpty(t *testing.T) {
	wire, err := buildGenerateRequest(luft.ProviderRequest{
		Model:    "gemini-1.5-pro",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if wire.GenerationConfig != nil {
		t.Errorf("generationConfig should be nil when nothing is set, got %+v", wire.GenerationConfig)
	}
}

func TestProvider_ParsesRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "8")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":429,"message":"slow","status":"RESOURCE_EXHAUSTED"}}`))
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	_, err := p.Call(context.Background(), luft.ProviderRequest{
		Model:    "gemini-1.5-pro",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
	})
	var apiErr *luft.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *luft.APIError, got %T: %v", err, err)
	}
	if apiErr.RetryAfter != 8*time.Second {
		t.Errorf("RetryAfter = %v, want 8s", apiErr.RetryAfter)
	}
}
