// Package decision exposes a single text-model request as an agent tool.
package decision

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/providers/openai"
)

// DefaultEndpoint is Baseten's shared OpenAI-compatible Model API.
const DefaultEndpoint = "https://inference.baseten.co/v1/chat/completions"

// Config selects an OpenAI-compatible chat completions endpoint. Credentials
// and routing are fixed by the operator, never supplied by the calling agent.
type Config struct {
	Endpoint   string
	Model      string
	APIKey     string
	HTTPClient *http.Client
}

// New creates a one-shot tool with no filesystem access or nested tool calls.
func New(cfg Config) (luft.ToolBinding, error) {
	if strings.TrimSpace(cfg.Model) == "" || strings.TrimSpace(cfg.APIKey) == "" {
		return luft.ToolBinding{}, fmt.Errorf("decision: Model and APIKey are required")
	}
	if cfg.Endpoint == "" {
		cfg.Endpoint = DefaultEndpoint
	}
	u, err := url.Parse(cfg.Endpoint)
	if err != nil || u.Host == "" || u.User != nil || u.Fragment != "" || (u.Scheme != "https" && u.Scheme != "http") {
		return luft.ToolBinding{}, fmt.Errorf("decision: Endpoint must be an http(s) URL without credentials or a fragment")
	}
	if u.Scheme == "http" && u.Hostname() != "localhost" && u.Hostname() != "127.0.0.1" && u.Hostname() != "::1" {
		return luft.ToolBinding{}, fmt.Errorf("decision: Endpoint must use HTTPS except on loopback")
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 120 * time.Second}
	}
	type input struct {
		Prompt string `json:"prompt"`
	}
	tool, fn := luft.NewTypedTool("decision_model",
		"Consult an external decision/reasoning model with one self-contained text prompt. Include the question, relevant facts, options, constraints, and desired answer format. It cannot see this conversation, read files, or execute tools. Returns advice for you to evaluate.",
		luft.Object(luft.String("prompt", "Complete text input for the decision model, including all necessary context.", luft.Required())),
		func(ctx context.Context, in input) (string, error) {
			if strings.TrimSpace(in.Prompt) == "" {
				return "", fmt.Errorf("decision: prompt is required")
			}
			ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
			defer cancel()
			resp, err := openai.CompatibleCall(ctx, cfg.HTTPClient, cfg.APIKey, cfg.Endpoint, luft.ProviderRequest{
				Model:     cfg.Model,
				MaxTokens: 8192,
				Messages:  []luft.Message{luft.NewUserMessage(in.Prompt)},
			}, false, false)
			if err != nil {
				return "", fmt.Errorf("decision: %w", err)
			}
			var text strings.Builder
			for _, block := range resp.Content {
				if block.Type == luft.TypeText {
					text.WriteString(block.Text)
				}
			}
			if strings.TrimSpace(text.String()) == "" {
				return "", fmt.Errorf("decision: model returned no text")
			}
			if resp.StopReason == "max_tokens" {
				return "", fmt.Errorf("decision: model exceeded output limit (partial answer: %s)", text.String())
			}
			return text.String(), nil
		})
	return luft.ToolBinding{Tool: tool, Func: fn, Meta: luft.ToolMetadata{Source: "tools/decision", ReadOnly: true}}, nil
}
