package bedrock

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/lukemuz/luft"
)

func TestBuildConverseRequest_GenerationParams(t *testing.T) {
	req := luft.ProviderRequest{
		Model:     "anthropic.claude",
		MaxTokens: 100,
		Messages:  []luft.Message{luft.NewUserMessage("hi")},
		Generation: luft.GenerationParams{
			Temperature:   luft.Ptr(0.0),
			TopP:          luft.Ptr(0.5),
			StopSequences: []string{"END"},
			TopK:          luft.Ptr(10), // unsupported by Converse inferenceConfig — dropped
			Seed:          luft.Ptr(7),  // unsupported by Converse inferenceConfig — dropped
		},
	}
	wire, err := buildConverseRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	ic := wire.InferenceConfig
	if ic == nil {
		t.Fatal("inferenceConfig is nil")
	}
	if ic.Temperature == nil || *ic.Temperature != 0.0 {
		t.Errorf("temperature = %v, want 0", ic.Temperature)
	}
	if ic.TopP == nil || *ic.TopP != 0.5 {
		t.Errorf("topP = %v, want 0.5", ic.TopP)
	}
	if len(ic.StopSequences) != 1 || ic.StopSequences[0] != "END" {
		t.Errorf("stopSequences = %v, want [END]", ic.StopSequences)
	}
	if ic.MaxTokens != 100 {
		t.Errorf("maxTokens = %d, want 100", ic.MaxTokens)
	}
}

// TestBuildConverseRequest_NoInferenceConfigWhenEmpty guards the "zero value =
// provider default" contract: with nothing set, inferenceConfig is omitted.
func TestBuildConverseRequest_NoInferenceConfigWhenEmpty(t *testing.T) {
	wire, err := buildConverseRequest(luft.ProviderRequest{
		Model:    "anthropic.claude",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if wire.InferenceConfig != nil {
		t.Errorf("inferenceConfig should be nil when nothing is set, got %+v", wire.InferenceConfig)
	}
}

func TestProvider_ParsesRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "8")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"__type":"ThrottlingException","message":"slow down"}`))
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	_, err := p.Call(context.Background(), luft.ProviderRequest{
		Model:    "anthropic.claude",
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
