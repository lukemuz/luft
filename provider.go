package luft

import "context"

// Provider is the abstraction over an LLM API backend.
// Implementations translate between the canonical ProviderRequest /
// ProviderResponse types and their own wire formats.
type Provider interface {
	// Call sends a single request and returns a normalised response.
	Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error)

	// Stream sends the request and invokes onDelta for every incremental
	// ContentBlock (typically text deltas; also partial tool_use blocks
	// when supported by the backend). It returns the final aggregated
	// ProviderResponse (complete Content, StopReason, and Usage) once
	// the stream ends. The callback is invoked synchronously as data
	// arrives. Retries (if configured) may cause multiple callback
	// invocations across attempts.
	Stream(ctx context.Context, req ProviderRequest, onDelta func(ContentBlock)) (ProviderResponse, error)
}

// ProviderRequest is the canonical request passed to every Provider.
//
// Tools are local (category-2 included) tool declarations the model may call;
// the provider translates them to its native wire format and the agent loop
// dispatches via ToolFunc when the model emits a tool_use block.
//
// ProviderTools are server-executed (category-1) tools — the provider runs
// them and returns inline result blocks. The loop never inspects them; they
// are passed straight through and spliced verbatim into the provider's tools
// array. Each entry is tagged with a Provider string so providers can reject
// mismatched entries at request build time.
type ProviderRequest struct {
	Model         string
	MaxTokens     int
	System        string
	Messages      []Message
	Tools         []Tool
	ProviderTools []ProviderTool

	// SystemCache, if set, marks the system prompt as a cache breakpoint.
	// This is the most useful single cache placement when the system text
	// is large and stable across turns. Honored by AnthropicProvider and
	// OpenRouterProvider; ignored elsewhere (OpenAI caches automatically).
	SystemCache *CacheControl

	// Generation carries optional sampling controls (temperature, top_p,
	// stop sequences, …). The zero value leaves every provider default in
	// place. Providers silently ignore controls they do not support; see
	// GenerationParams for the per-provider support matrix.
	Generation GenerationParams
}

// ProviderResponse is the normalised response every Provider must return.
type ProviderResponse struct {
	Content    []ContentBlock
	StopReason string
	Usage      Usage
}

// GenerationParams holds optional sampling controls. Every field is a pointer
// or slice so its zero value means "unset — use the provider's default."
//
// Pointers matter most for Temperature: temperature 0 (deterministic) is a
// meaningful value that a bare float64 could not distinguish from "unset."
// Construct pointers with Ptr:
//
//	gen := luft.GenerationParams{Temperature: luft.Ptr(0.0)}
//
// Provider support varies; unsupported fields are silently dropped:
//
//	                          Temperature  TopP  TopK  StopSequences  Seed
//	anthropic.Provider             ✓        ✓     ✓         ✓          —
//	openai.Provider                ✓        ✓     —         ✓          ✓
//	openai.ResponsesProvider       ✓        ✓     —         —          —
//	openrouter.Provider            ✓        ✓     —         ✓          ✓
//	bedrock.Provider               ✓        ✓     —         ✓          —
//	vertex.Provider                ✓        ✓     ✓         ✓          ✓
type GenerationParams struct {
	// Temperature controls sampling randomness; nil uses the provider default.
	Temperature *float64
	// TopP is the nucleus-sampling probability mass; nil uses the default.
	TopP *float64
	// TopK limits sampling to the top-K tokens. Honored only by
	// anthropic.Provider; other providers leave it unset.
	TopK *int
	// StopSequences are strings that halt generation when produced. Empty
	// means none. Honored by every provider except openai.ResponsesProvider.
	StopSequences []string
	// Seed requests reproducible sampling where the backend supports it
	// (OpenAI / OpenRouter). nil leaves it unset; anthropic ignores it.
	Seed *int
}

// Ptr returns a pointer to v. It is a convenience for the optional pointer
// fields of GenerationParams (and any other *T config field):
//
//	luft.GenerationParams{Temperature: luft.Ptr(0.2), Seed: luft.Ptr(42)}
func Ptr[T any](v T) *T { return &v }
