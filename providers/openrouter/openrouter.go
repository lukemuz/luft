package openrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
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
	var opt []openai.CompatibleOption
	req.Model, opt = splitProviderRouting(req.Model)
	return openai.CompatibleCall(
		ctx,
		p.cfg.HTTPClient,
		p.cfg.APIKey,
		p.cfg.BaseURL+"/api/v1/chat/completions",
		req,
		true,
		true,
		opt...,
	)
}

// Stream implements luft.Provider.Stream for OpenRouter by delegating to
// the shared streaming helper (mirrors the Call pattern).
func (p *Provider) Stream(ctx context.Context, req luft.ProviderRequest, onDelta func(luft.ContentBlock)) (luft.ProviderResponse, error) {
	if err := validateProviderTools(req.ProviderTools); err != nil {
		return luft.ProviderResponse{}, err
	}
	var opt []openai.CompatibleOption
	req.Model, opt = splitProviderRouting(req.Model)
	return openai.CompatibleStream(
		ctx,
		p.cfg.HTTPClient,
		p.cfg.APIKey,
		p.cfg.BaseURL+"/api/v1/chat/completions",
		req,
		onDelta,
		true,
		true,
		opt...,
	)
}

// splitProviderRouting parses an OpenRouter model string that optionally pins
// one or more upstream providers with an "@" suffix:
//
//	"openai/gpt-oss-120b@cerebras"        → prefer Cerebras
//	"openai/gpt-oss-120b@cerebras,groq"   → prefer Cerebras, then Groq
//
// It returns the bare model id plus a CompatibleOption that sets OpenRouter's
// provider-routing "order" so those providers are tried first. Fallbacks stay
// enabled, so an unavailable pinned provider degrades to OpenRouter's normal
// routing rather than failing the request. Without an "@", no option is
// returned and the model is unchanged. Provider names are passed through
// verbatim (OpenRouter matches them case-insensitively).
func splitProviderRouting(model string) (string, []openai.CompatibleOption) {
	at := strings.LastIndex(model, "@")
	if at < 0 {
		return model, nil
	}
	base := strings.TrimSpace(model[:at])
	var provs []string
	for _, p := range strings.Split(model[at+1:], ",") {
		if p = strings.TrimSpace(p); p != "" {
			provs = append(provs, p)
		}
	}
	if base == "" || len(provs) == 0 {
		return model, nil // malformed suffix — leave the slug untouched
	}
	routing, err := json.Marshal(struct {
		Order []string `json:"order"`
	}{Order: provs})
	if err != nil {
		return base, nil
	}
	return base, []openai.CompatibleOption{openai.WithProviderRouting(routing)}
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
