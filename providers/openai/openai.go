package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/lukemuz/luft"
)

const (
	openAIDefaultBaseURL = "https://api.openai.com"
	defaultHTTPTimeout   = 60 * time.Second
)

// ---------------------------------------------------------------------------
// OpenAI-compatible wire types (reused by openrouter.Provider as well)
// ---------------------------------------------------------------------------

// openAIChatRequest is the JSON body for OpenAI-compatible chat completions.
//
// Tools is []json.RawMessage rather than []openAIToolDef so backends that host
// non-function tools (OpenRouter's "openrouter:web_search", etc.) can splice
// opaque entries alongside function-call tool definitions. toOpenAITools
// renders each function tool to JSON itself and concatenates with any
// allowed ProviderTool.Raw bodies.
type openAIChatRequest struct {
	Model       string            `json:"model"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Messages    []openAIMessage   `json:"messages"`
	Tools       []json.RawMessage `json:"tools,omitempty"`
	Temperature *float64          `json:"temperature,omitempty"`
	TopP        *float64          `json:"top_p,omitempty"`
	Stop        []string          `json:"stop,omitempty"`
	Seed        *int              `json:"seed,omitempty"`
	Stream      bool              `json:"stream,omitempty"`
}

// openAIMessage is a wire message. Content is `any` so request paths can
// emit either a plain string or a typed-parts array (used to attach
// cache_control on cache-aware backends like OpenRouter); response paths
// receive a string and use messageContentString to coerce.
//
// Annotations is the raw annotations array some backends attach to assistant
// messages (e.g. OpenRouter url_citation entries returned alongside hosted
// web_search results). It is captured opaquely; fromOpenAIResponse turns
// each entry into an opaque luft.ContentBlock so callers can render
// citations without the library taking a dependency on a specific shape.
type openAIMessage struct {
	Role        string            `json:"role"`
	Content     any               `json:"content,omitempty"`
	ToolCallID  string            `json:"tool_call_id,omitempty"`
	ToolCalls   []openAIToolCall  `json:"tool_calls,omitempty"`
	Annotations []json.RawMessage `json:"annotations,omitempty"`
}

// openAIContentPart is one element in the array form of openAIMessage.Content.
// Two variants are supported: the text part (Type=="text", Text populated)
// and the image_url part (Type=="image_url", ImageURL populated). cache_control
// rides along on text parts when the canonical message marked them.
type openAIContentPart struct {
	Type         string             `json:"type"` // "text" or "image_url"
	Text         string             `json:"text,omitempty"`
	ImageURL     *openAIImageURL    `json:"image_url,omitempty"`
	CacheControl *luft.CacheControl `json:"cache_control,omitempty"`
}

// openAIImageURL is the value of an image_url content part. URL is either
// a base64 data URI or a remote http(s) URL; OpenAI Chat Completions and
// most OpenRouter backends accept both, with data URIs being the more
// portable choice.
type openAIImageURL struct {
	URL string `json:"url"`
}

// messageContentString coerces the polymorphic Content field back to a
// plain string when decoding responses (which always carry string content).
func messageContentString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

type openAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // always "function"
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openAIToolDef struct {
	Type     string `json:"type"` // always "function"
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
	// CacheControl, when set, becomes a sibling field on the tool definition.
	// Only emitted when the caller passes cacheCompatible=true (e.g. OpenRouter).
	CacheControl *luft.CacheControl `json:"cache_control,omitempty"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message      openAIMessage `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails struct {
			// CachedTokens is the OpenAI-compatible report of how many of
			// the prompt tokens were served from the prompt cache. OpenRouter
			// surfaces the same field for routes that support caching.
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

type openAIErrorBody struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

type openAIStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content,omitempty"`
			ToolCalls []struct {
				Index    int    `json:"index,omitempty"`
				ID       string `json:"id,omitempty"`
				Type     string `json:"type,omitempty"`
				Function struct {
					Name      string `json:"name,omitempty"`
					Arguments string `json:"arguments,omitempty"`
				} `json:"function,omitempty"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens,omitempty"`
		CompletionTokens    int `json:"completion_tokens,omitempty"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details,omitempty"`
	} `json:"usage,omitempty"`
}

// ---------------------------------------------------------------------------
// Shared conversion helpers
// ---------------------------------------------------------------------------

// toOpenAIMessages converts canonical Messages to OpenAI wire format.
// system is prepended as a system-role message if non-empty.
//
// cacheCompatible controls whether cache_control markers ride along on the
// wire. When true (e.g. OpenRouter), text content is emitted as a typed-parts
// array so cache_control can attach to the right spot. When false (e.g.
// OpenAI Chat Completions, which auto-caches and may reject unknown fields),
// markers are dropped silently.
func toOpenAIMessages(system string, systemCache *luft.CacheControl, messages []luft.Message, cacheCompatible bool) []openAIMessage {
	var out []openAIMessage

	if system != "" {
		out = append(out, openAIMessage{Role: "system", Content: wireTextContent(system, systemCache, cacheCompatible)})
	}

	for _, msg := range messages {
		switch msg.Role {
		case luft.RoleAssistant:
			m := openAIMessage{Role: luft.RoleAssistant}
			text, cache := joinTextBlocks(msg.Content)
			for _, block := range msg.Content {
				if block.Type != luft.TypeToolUse {
					continue
				}
				args := string(block.Input)
				if args == "" {
					args = "{}"
				}
				tc := openAIToolCall{
					ID:   block.ID,
					Type: "function",
				}
				tc.Function.Name = block.Name
				tc.Function.Arguments = args
				m.ToolCalls = append(m.ToolCalls, tc)
			}
			if text != "" {
				m.Content = wireTextContent(text, cache, cacheCompatible)
			}
			out = append(out, m)

		case luft.RoleUser:
			// Separate tool_result blocks into individual tool-role messages;
			// collect remaining text + image blocks into a single user message.
			for _, block := range msg.Content {
				if block.Type == luft.TypeToolResult {
					out = append(out, openAIMessage{
						Role:       "tool",
						ToolCallID: block.ToolUseID,
						Content:    block.Content,
					})
				}
			}
			if userContent, ok := buildUserContent(msg.Content, cacheCompatible); ok {
				out = append(out, openAIMessage{
					Role:    luft.RoleUser,
					Content: userContent,
				})
			}
		}
	}

	return out
}

// buildUserContent renders the non-tool_result blocks of a user-role message
// into the wire shape. When images are present it emits a typed-parts array
// (text part(s) followed by image_url parts); otherwise it falls through to
// wireTextContent so backward-compat with the plain-string path is exact.
// Returns (content, true) when a user message should be emitted, or
// (nil, false) when there is nothing to send (e.g. tool_result-only turn).
func buildUserContent(blocks []luft.ContentBlock, cacheCompatible bool) (any, bool) {
	hasImage := false
	for _, b := range blocks {
		if b.Type == luft.TypeImage {
			hasImage = true
			break
		}
	}
	if !hasImage {
		text, cache := joinTextBlocks(blocks)
		if text == "" {
			return nil, false
		}
		return wireTextContent(text, cache, cacheCompatible), true
	}

	text, cache := joinTextBlocks(blocks)
	parts := make([]openAIContentPart, 0, 1+len(blocks))
	if text != "" {
		part := openAIContentPart{Type: "text", Text: text}
		if cacheCompatible {
			part.CacheControl = cache
		}
		parts = append(parts, part)
	}
	for _, b := range blocks {
		if b.Type != luft.TypeImage {
			continue
		}
		parts = append(parts, openAIContentPart{
			Type:     "image_url",
			ImageURL: &openAIImageURL{URL: b.Source},
		})
	}
	if len(parts) == 0 {
		return nil, false
	}
	return parts, true
}

// joinTextBlocks concatenates the text blocks within a Message and returns
// the highest-precedence cache marker among them. If multiple text blocks
// carry markers we keep the last one — for cumulative cache semantics the
// later marker subsumes earlier ones at the same level.
func joinTextBlocks(blocks []luft.ContentBlock) (string, *luft.CacheControl) {
	var b strings.Builder
	var cache *luft.CacheControl
	for _, block := range blocks {
		if block.Type != luft.TypeText {
			continue
		}
		b.WriteString(block.Text)
		if block.CacheControl != nil {
			cache = block.CacheControl
		}
	}
	return b.String(), cache
}

// wireTextContent picks the wire shape: plain string when no cache marker
// applies (or the backend can't carry it), or a single-element typed-parts
// array when we need to attach cache_control.
func wireTextContent(text string, cache *luft.CacheControl, cacheCompatible bool) any {
	if cache == nil || !cacheCompatible {
		return text
	}
	return []openAIContentPart{{Type: "text", Text: text, CacheControl: cache}}
}

// toOpenAITools converts canonical Tools to OpenAI function-calling format
// and (when allowProviderTools=true) splices any ProviderTool.Raw bodies as
// opaque entries in the resulting tools array.
//
// Provider-tagged tools (category 2 — Anthropic-only declaration shapes) are
// rejected unconditionally: those wire shapes are not translatable to Chat
// Completions.
//
// ProviderTools (category 1 — server-executed) are rejected when
// allowProviderTools=false. Stock OpenAI Chat Completions does not expose
// hosted tools — callers who want OpenAI-hosted tools (web_search,
// file_search, code_interpreter) need a Responses-API provider. OpenRouter,
// despite speaking the same wire format, *does* host tools at this endpoint
// (e.g. "openrouter:web_search"), so its Provider passes allowProviderTools
// =true.
//
// cacheCompatible controls whether Tool.CacheControl markers are emitted as
// a sibling cache_control field on the wire. False for stock OpenAI; true
// for OpenRouter, which routes the marker to Anthropic backends.
func toOpenAITools(tools []luft.Tool, providerTools []luft.ProviderTool, cacheCompatible, allowProviderTools bool) ([]json.RawMessage, error) {
	if len(providerTools) > 0 && !allowProviderTools {
		return nil, fmt.Errorf("luft: openai (chat completions) does not support ProviderTools; use a Responses-API provider")
	}
	if len(tools) == 0 && len(providerTools) == 0 {
		return nil, nil
	}
	out := make([]json.RawMessage, 0, len(tools)+len(providerTools))
	for _, t := range tools {
		if t.Provider != "" && t.Provider != "openai" {
			return nil, fmt.Errorf("luft: openai: tool %q is tagged for provider %q", t.Name, t.Provider)
		}
		var td openAIToolDef
		td.Type = "function"
		td.Function.Name = t.Name
		td.Function.Description = t.Description
		td.Function.Parameters = t.InputSchema
		if cacheCompatible {
			td.CacheControl = t.CacheControl
		}
		raw, err := json.Marshal(td)
		if err != nil {
			return nil, fmt.Errorf("luft: marshal openai tool %q: %w", t.Name, err)
		}
		out = append(out, raw)
	}
	for _, pt := range providerTools {
		if len(pt.Raw) == 0 {
			return nil, fmt.Errorf("luft: openai: provider tool tagged %q has empty Raw body", pt.Provider)
		}
		out = append(out, pt.Raw)
	}
	return out, nil
}

// applyChatGeneration copies supported sampling controls onto the chat-
// completions wire request. OpenAI (and OpenRouter, via this shared path)
// support temperature, top_p, stop, and seed. TopK has no equivalent in this
// wire format and is dropped.
func applyChatGeneration(r *openAIChatRequest, g luft.GenerationParams) {
	r.Temperature = g.Temperature
	r.TopP = g.TopP
	r.Stop = g.StopSequences
	r.Seed = g.Seed
}

// openAIFinishReason maps an OpenAI finish_reason to canonical StopReason.
func openAIFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	default:
		return reason
	}
}

// fromOpenAIResponse converts an openAIChatResponse to a luft.ProviderResponse.
func fromOpenAIResponse(r openAIChatResponse) luft.ProviderResponse {
	if len(r.Choices) == 0 {
		return luft.ProviderResponse{}
	}

	choice := r.Choices[0]
	var content []luft.ContentBlock

	if text := messageContentString(choice.Message.Content); text != "" {
		content = append(content, luft.ContentBlock{
			Type: luft.TypeText,
			Text: text,
		})
	}

	for _, ann := range choice.Message.Annotations {
		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(ann, &head); err != nil || head.Type == "" {
			continue
		}
		content = append(content, luft.ContentBlock{
			Type: head.Type,
			Raw:  append(json.RawMessage(nil), ann...),
		})
	}

	for _, tc := range choice.Message.ToolCalls {
		content = append(content, luft.ContentBlock{
			Type:  luft.TypeToolUse,
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		})
	}

	return luft.ProviderResponse{
		Content:    content,
		StopReason: openAIFinishReason(choice.FinishReason),
		Usage: luft.Usage{
			InputTokens:     r.Usage.PromptTokens,
			OutputTokens:    r.Usage.CompletionTokens,
			CacheReadTokens: r.Usage.PromptTokensDetails.CachedTokens,
		},
	}
}

// ---------------------------------------------------------------------------
// Config and Provider
// ---------------------------------------------------------------------------

// Config holds configuration for the OpenAI provider.
type Config struct {
	APIKey     string       // required
	BaseURL    string       // defaults to https://api.openai.com
	HTTPClient *http.Client // defaults to a 60-second timeout client
}

// Provider implements luft.Provider for the OpenAI Chat Completions API.
type Provider struct {
	cfg Config
}

// NewProvider creates a Provider, filling in defaults.
func NewProvider(cfg Config) (*Provider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("luft: Config.APIKey is required")
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = openAIDefaultBaseURL
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: defaultHTTPTimeout}
	}
	return &Provider{cfg: cfg}, nil
}

// NewProviderFromEnv creates a Provider using the OPENAI_API_KEY environment
// variable. Returns an error if the variable is unset or empty.
func NewProviderFromEnv() (*Provider, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("luft: OPENAI_API_KEY environment variable is not set")
	}
	return NewProvider(Config{APIKey: key})
}

// NewClientFromEnv creates a Client backed by the OpenAI provider, reading
// the API key from OPENAI_API_KEY. model is the model identifier (e.g.
// "gpt-4o", "gpt-4o-mini").
func NewClientFromEnv(model string) (*luft.Client, error) {
	provider, err := NewProviderFromEnv()
	if err != nil {
		return nil, err
	}
	return luft.New(luft.Config{Provider: provider, Model: model})
}

// Call implements luft.Provider for OpenAI Chat Completions. Cache markers
// in canonical types are dropped because OpenAI auto-caches and may validate
// strictly against unknown fields. ProviderTools are rejected: stock OpenAI
// hosts tools only via the Responses API.
func (p *Provider) Call(ctx context.Context, req luft.ProviderRequest) (luft.ProviderResponse, error) {
	return CompatibleCall(ctx, p.cfg.HTTPClient, p.cfg.APIKey, p.cfg.BaseURL+"/v1/chat/completions", req, false, false)
}

// Stream implements luft.Provider.Stream by delegating to the shared
// streaming helper (mirrors the Call pattern).
func (p *Provider) Stream(ctx context.Context, req luft.ProviderRequest, onDelta func(luft.ContentBlock)) (luft.ProviderResponse, error) {
	return CompatibleStream(ctx, p.cfg.HTTPClient, p.cfg.APIKey, p.cfg.BaseURL+"/v1/chat/completions", req, onDelta, false, false)
}

// ---------------------------------------------------------------------------
// Shared HTTP call helper for OpenAI-compatible endpoints
// ---------------------------------------------------------------------------

// CompatibleCall marshals req into an OpenAI chat completions body, POSTs it
// to url with the given Bearer key, and decodes the response. cacheCompatible
// controls whether cache_control markers are emitted (OpenRouter) or dropped
// (stock OpenAI). allowProviderTools=true permits ProviderTool entries to be
// spliced into the wire tools array (used by OpenRouter for hosted tools
// like "openrouter:web_search"); stock OpenAI passes false.
func CompatibleCall(
	ctx context.Context,
	httpClient *http.Client,
	apiKey string,
	url string,
	req luft.ProviderRequest,
	cacheCompatible bool,
	allowProviderTools bool,
) (luft.ProviderResponse, error) {
	openaiTools, err := toOpenAITools(req.Tools, req.ProviderTools, cacheCompatible, allowProviderTools)
	if err != nil {
		return luft.ProviderResponse{}, err
	}
	chatReq := openAIChatRequest{
		Model:     req.Model,
		MaxTokens: req.MaxTokens,
		Messages:  toOpenAIMessages(req.System, req.SystemCache, req.Messages, cacheCompatible),
		Tools:     openaiTools,
	}
	applyChatGeneration(&chatReq, req.Generation)

	body, err := json.Marshal(chatReq)
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: marshal openai request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: build openai request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: openai http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errBody openAIErrorBody
		json.NewDecoder(resp.Body).Decode(&errBody) //nolint:errcheck
		return luft.ProviderResponse{}, &luft.APIError{
			StatusCode: resp.StatusCode,
			Type:       errBody.Error.Type,
			Message:    errBody.Error.Message,
			RetryAfter: luft.ParseRetryAfter(resp.Header),
		}
	}

	var chatResp openAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: decode openai response: %w", err)
	}

	return fromOpenAIResponse(chatResp), nil
}

// CompatibleStream marshals req (with Stream=true), POSTs to the OpenAI-
// compatible endpoint with Accept: text/event-stream header, handles non-200
// errors identically to the non-stream version, then uses bufio.Scanner to
// parse "data: " JSON chunks (skipping "[DONE]"). It calls onDelta for each
// text delta and each tool_call delta (accumulating arguments per index from
// the deltas), accumulates full text and tool calls for the final
// luft.ProviderResponse.Content (mirroring fromOpenAIResponse logic), maps
// finish_reason via openAIFinishReason and captures usage, then returns the
// aggregated response (or a stream read error).
func CompatibleStream(
	ctx context.Context,
	httpClient *http.Client,
	apiKey string,
	url string,
	req luft.ProviderRequest,
	onDelta func(luft.ContentBlock),
	cacheCompatible bool,
	allowProviderTools bool,
) (luft.ProviderResponse, error) {
	openaiTools, err := toOpenAITools(req.Tools, req.ProviderTools, cacheCompatible, allowProviderTools)
	if err != nil {
		return luft.ProviderResponse{}, err
	}
	chatReq := openAIChatRequest{
		Model:     req.Model,
		MaxTokens: req.MaxTokens,
		Messages:  toOpenAIMessages(req.System, req.SystemCache, req.Messages, cacheCompatible),
		Tools:     openaiTools,
		Stream:    true,
	}
	applyChatGeneration(&chatReq, req.Generation)

	body, err := json.Marshal(chatReq)
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: marshal openai stream request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: build openai stream request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: openai stream http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errBody openAIErrorBody
		json.NewDecoder(resp.Body).Decode(&errBody) //nolint:errcheck
		return luft.ProviderResponse{}, &luft.APIError{
			StatusCode: resp.StatusCode,
			Type:       errBody.Error.Type,
			Message:    errBody.Error.Message,
			RetryAfter: luft.ParseRetryAfter(resp.Header),
		}
	}

	var fullText strings.Builder
	toolCallAccum := make(map[int]openAIToolCall)
	var stopReason string
	var usage luft.Usage

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if strings.TrimSpace(data) == "[DONE]" {
			break
		}

		// Some providers embed error objects in the stream body even on HTTP 200.
		var errCheck openAIErrorBody
		if err := json.Unmarshal([]byte(data), &errCheck); err == nil && errCheck.Error.Message != "" {
			return luft.ProviderResponse{}, &luft.APIError{Type: errCheck.Error.Type, Message: errCheck.Error.Message}
		}

		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) == 0 {
			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				usage.InputTokens = chunk.Usage.PromptTokens
				usage.OutputTokens = chunk.Usage.CompletionTokens
				usage.CacheReadTokens = chunk.Usage.PromptTokensDetails.CachedTokens
			}
			continue
		}

		choice := chunk.Choices[0]

		if choice.Delta.Content != "" {
			delta := choice.Delta.Content
			fullText.WriteString(delta)
			onDelta(luft.ContentBlock{
				Type: luft.TypeText,
				Text: delta,
			})
		}

		for _, tcd := range choice.Delta.ToolCalls {
			idx := tcd.Index
			tc := toolCallAccum[idx]
			if tcd.ID != "" {
				tc.ID = tcd.ID
			}
			if tcd.Type != "" {
				tc.Type = tcd.Type
			}
			if tcd.Function.Name != "" {
				tc.Function.Name = tcd.Function.Name
			}
			if tcd.Function.Arguments != "" {
				tc.Function.Arguments += tcd.Function.Arguments
			}
			toolCallAccum[idx] = tc

			cb := luft.ContentBlock{
				Type: luft.TypeToolUse,
				ID:   tc.ID,
				Name: tc.Function.Name,
			}
			if tc.Function.Arguments != "" {
				cb.Input = json.RawMessage(tc.Function.Arguments)
			}
			onDelta(cb)
		}

		if choice.FinishReason != nil && *choice.FinishReason != "" {
			stopReason = openAIFinishReason(*choice.FinishReason)
		}

		if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
			usage.InputTokens = chunk.Usage.PromptTokens
			usage.OutputTokens = chunk.Usage.CompletionTokens
			usage.CacheReadTokens = chunk.Usage.PromptTokensDetails.CachedTokens
		}
	}

	if err := scanner.Err(); err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: openai stream read: %w", err)
	}

	// Build final content (mirrors fromOpenAIResponse).
	var content []luft.ContentBlock
	if fullText.Len() > 0 {
		content = append(content, luft.ContentBlock{
			Type: luft.TypeText,
			Text: fullText.String(),
		})
	}
	for _, tc := range toolCallAccum {
		var input json.RawMessage
		if tc.Function.Arguments != "" {
			input = json.RawMessage(tc.Function.Arguments)
		}
		content = append(content, luft.ContentBlock{
			Type:  luft.TypeToolUse,
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		})
	}

	if stopReason == "" {
		if len(toolCallAccum) > 0 {
			stopReason = "tool_use"
		} else {
			stopReason = "end_turn"
		}
	}

	return luft.ProviderResponse{
		Content:    content,
		StopReason: stopReason,
		Usage:      usage,
	}, nil
}
