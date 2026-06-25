package openrouter

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/providers/openai"
)

const (
	openRouterDefaultBaseURL = "https://openrouter.ai"
	defaultHTTPTimeout       = 60 * time.Second
)

// Config holds configuration for the OpenRouter provider.
type Config struct {
	APIKey     string       // required
	BaseURL    string       // defaults to https://openrouter.ai
	HTTPClient *http.Client // defaults to a 60-second timeout client
}

// Provider implements Provider for the OpenRouter API (OpenAI-compatible).
type Provider struct {
	cfg Config
}

// NewProvider creates an Provider, filling in defaults.
func NewProvider(cfg Config) (*Provider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("luft: Config.APIKey is required")
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = openRouterDefaultBaseURL
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: defaultHTTPTimeout}
	}
	return &Provider{cfg: cfg}, nil
}

// NewProviderFromEnv creates an Provider using the
// OPENROUTER_API_KEY environment variable. Returns an error if the variable
// is unset or empty.
func NewProviderFromEnv() (*Provider, error) {
	key := os.Getenv("OPENROUTER_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("luft: OPENROUTER_API_KEY environment variable is not set")
	}
	return NewProvider(Config{APIKey: key})
}

// NewClientFromEnv creates a Client backed by the OpenRouter provider,
// reading the API key from OPENROUTER_API_KEY. model is any model string
// supported by OpenRouter (e.g. "anthropic/claude-sonnet-4-6").
func NewClientFromEnv(model string) (*luft.Client, error) {
	provider, err := NewProviderFromEnv()
	if err != nil {
		return nil, err
	}
	return luft.New(luft.Config{Provider: provider, Model: model})
}

// Call implements luft.Provider for OpenRouter using the OpenAI-compatible
// chat completions endpoint.
//
// cacheCompatible=true: OpenRouter accepts cache_control markers (passed
// through to Anthropic backends, ignored by OpenAI backends), so we emit
// the typed-parts content shape and tool cache_control field whenever the
// canonical types carry them.
//
// allowProviderTools=true: OpenRouter hosts tools server-side at this same
// endpoint (e.g. WebSearch — "openrouter:web_search"); ProviderTool entries
// are spliced verbatim into the wire tools array.
func (p *Provider) Call(ctx context.Context, req luft.ProviderRequest) (luft.ProviderResponse, error) {
	if err := validateProviderTools(req.ProviderTools); err != nil {
		return luft.ProviderResponse{}, err
	}
	return openai.CompatibleCall(
		ctx,
		p.cfg.HTTPClient,
		p.cfg.APIKey,
		p.cfg.BaseURL+"/api/v1/chat/completions",
		req,
		true,
		true,
	)
}

// Stream implements luft.Provider.Stream for OpenRouter by delegating to
// the shared streaming helper (mirrors the Call pattern).
func (p *Provider) Stream(ctx context.Context, req luft.ProviderRequest, onDelta func(luft.ContentBlock)) (luft.ProviderResponse, error) {
	if err := validateProviderTools(req.ProviderTools); err != nil {
		return luft.ProviderResponse{}, err
	}
	return openai.CompatibleStream(
		ctx,
		p.cfg.HTTPClient,
		p.cfg.APIKey,
		p.cfg.BaseURL+"/api/v1/chat/completions",
		req,
		onDelta,
		true,
		true,
	)
}

// validateProviderTools rejects ProviderTool entries tagged for a different
// provider. Mirrors what AnthropicProvider does for its own tools — surfaces
// misuse loudly at request build time rather than dispatching a malformed
// request.
func validateProviderTools(pts []luft.ProviderTool) error {
	for _, pt := range pts {
		if pt.Provider != ProviderTag {
			return fmt.Errorf("luft: openrouter: provider tool tagged %q cannot be used with this provider", pt.Provider)
		}
	}
	return nil
}
