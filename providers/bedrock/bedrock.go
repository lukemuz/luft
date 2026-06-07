// Package bedrock is a luft provider for AWS Bedrock using the Converse and
// ConverseStream APIs.
//
// The Converse API is the unified Bedrock entry point: one request shape works
// across Anthropic, Meta, Mistral, Cohere, AI21, and Amazon model families,
// and the response shape is normalized. luft's bedrock provider speaks only
// Converse (and ConverseStream for streaming) — older per-family
// InvokeModel routes are intentionally not exposed.
//
// Auth is AWS Signature Version 4, written from scratch in sigv4.go so the
// provider keeps luft's zero-dependency promise. Streaming responses arrive
// over AWS's binary event-stream framing, decoded in eventstream.go.
package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/lukemuz/luft"
)

const (
	bedrockService     = "bedrock"
	defaultHTTPTimeout = 60 * time.Second

	// ProviderTag is the value luft.Tool.Provider and luft.ProviderTool.Provider
	// must carry for an entry to be accepted by this provider.
	ProviderTag = "bedrock"
)

// Config holds configuration for the Bedrock provider.
type Config struct {
	// Credentials, if non-zero, override credentials resolved from the
	// environment. SessionToken is optional (set for STS / IAM-role / SSO).
	Credentials Credentials

	// Region is the AWS region for the Bedrock runtime endpoint (e.g.
	// "us-east-1"). Required; if empty, NewProvider reads AWS_REGION then
	// AWS_DEFAULT_REGION.
	Region string

	// BaseURL overrides the default
	// https://bedrock-runtime.<region>.amazonaws.com endpoint. Set to the
	// URL of an httptest.Server in tests.
	BaseURL string

	// HTTPClient lets you supply a custom client (timeouts, proxy, transport).
	// Defaults to a 60-second-timeout client.
	HTTPClient *http.Client

	// Now, if set, supplies the signing timestamp. Used by tests for
	// deterministic signatures. Defaults to time.Now().UTC().
	Now func() time.Time
}

// Credentials carries the IAM identity used to sign every Bedrock request.
// Populate from the environment or pass explicitly via Config.
type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string // optional; required only for STS / IAM-role / SSO creds
}

// Provider implements luft.Provider for Bedrock Converse / ConverseStream.
type Provider struct {
	cfg Config
}

// NewProvider creates a Bedrock provider, filling in defaults. Returns an
// error if Region cannot be resolved or AWS credentials are missing.
func NewProvider(cfg Config) (*Provider, error) {
	if cfg.Region == "" {
		cfg.Region = firstNonEmpty(os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION"))
		if cfg.Region == "" {
			return nil, errors.New("luft: bedrock: region not set (pass Config.Region or set AWS_REGION)")
		}
	}
	if cfg.Credentials.AccessKeyID == "" {
		id := os.Getenv("AWS_ACCESS_KEY_ID")
		secret := os.Getenv("AWS_SECRET_ACCESS_KEY")
		if id == "" || secret == "" {
			return nil, errors.New("luft: bedrock: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required (or pass Config.Credentials)")
		}
		cfg.Credentials = Credentials{
			AccessKeyID:     id,
			SecretAccessKey: secret,
			SessionToken:    os.Getenv("AWS_SESSION_TOKEN"),
		}
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", cfg.Region)
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: defaultHTTPTimeout}
	}
	if cfg.Now == nil {
		cfg.Now = func() time.Time { return time.Now().UTC() }
	}
	return &Provider{cfg: cfg}, nil
}

// NewProviderFromEnv reads region and credentials entirely from the
// environment (AWS_REGION / AWS_DEFAULT_REGION, AWS_ACCESS_KEY_ID,
// AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN). Use this when you need a custom
// luft.Config; otherwise prefer NewClientFromEnv.
func NewProviderFromEnv() (*Provider, error) {
	return NewProvider(Config{})
}

// NewClientFromEnv builds a luft.Client backed by a Bedrock provider
// configured from the environment. model is the Bedrock model identifier
// (e.g. "anthropic.claude-sonnet-4-5-20250929-v1:0" or a cross-region
// inference profile id like "us.anthropic.claude-sonnet-4-5-20250929-v1:0").
func NewClientFromEnv(model string) (*luft.Client, error) {
	p, err := NewProviderFromEnv()
	if err != nil {
		return nil, err
	}
	return luft.New(luft.Config{Provider: p, Model: model})
}

// ---- wire types ----

type converseRequest struct {
	Messages        []converseMessage     `json:"messages"`
	System          []converseSystemBlock `json:"system,omitempty"`
	InferenceConfig *inferenceConfig      `json:"inferenceConfig,omitempty"`
	ToolConfig      *converseToolConfig   `json:"toolConfig,omitempty"`
}

type converseSystemBlock struct {
	Text string `json:"text"`
}

type inferenceConfig struct {
	MaxTokens     int      `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

type converseToolConfig struct {
	Tools []converseToolEntry `json:"tools"`
}

// converseToolEntry wraps either a {toolSpec: ...} or a raw entry. We always
// emit toolSpec for ordinary luft.Tool declarations.
type converseToolEntry struct {
	ToolSpec *converseToolSpec `json:"toolSpec,omitempty"`
}

type converseToolSpec struct {
	Name        string             `json:"name"`
	Description string             `json:"description,omitempty"`
	InputSchema converseInputShape `json:"inputSchema"`
}

type converseInputShape struct {
	JSON json.RawMessage `json:"json"`
}

type converseMessage struct {
	Role    string                 `json:"role"`
	Content []converseContentBlock `json:"content"`
}

// converseContentBlock is exactly one of the named child fields populated at
// a time. Bedrock's wire format is a discriminated union by presence of key.
type converseContentBlock struct {
	Text       string              `json:"text,omitempty"`
	Image      *converseImage      `json:"image,omitempty"`
	ToolUse    *converseToolUse    `json:"toolUse,omitempty"`
	ToolResult *converseToolResult `json:"toolResult,omitempty"`
}

type converseImage struct {
	Format string              `json:"format"` // "png" | "jpeg" | "gif" | "webp"
	Source converseImageSource `json:"source"`
}

type converseImageSource struct {
	Bytes string `json:"bytes"` // base64-encoded
}

type converseToolUse struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

type converseToolResult struct {
	ToolUseID string                    `json:"toolUseId"`
	Content   []converseToolResultBlock `json:"content"`
	Status    string                    `json:"status,omitempty"` // "success" | "error"
}

type converseToolResultBlock struct {
	Text string `json:"text,omitempty"`
}

type converseResponse struct {
	Output struct {
		Message converseMessage `json:"message"`
	} `json:"output"`
	StopReason string `json:"stopReason"`
	Usage      struct {
		InputTokens  int `json:"inputTokens"`
		OutputTokens int `json:"outputTokens"`
	} `json:"usage"`
}

type errorBody struct {
	Message string `json:"message"`
	Type    string `json:"__type"`
}

// ---- request build / response parse ----

// buildConverseRequest converts a luft.ProviderRequest into the Converse wire
// shape. SystemCache is ignored — Bedrock does not currently honor caller-side
// cache markers across all model families.
func buildConverseRequest(req luft.ProviderRequest) (converseRequest, error) {
	out := converseRequest{}
	if req.System != "" {
		out.System = []converseSystemBlock{{Text: req.System}}
	}
	// Bedrock Converse exposes temperature, topP, and stopSequences in
	// inferenceConfig. TopK and Seed have no portable Converse field (they are
	// model-specific additionalModelRequestFields) and are dropped. The config
	// is attached only when at least one field is set.
	ic := inferenceConfig{
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Generation.Temperature,
		TopP:          req.Generation.TopP,
		StopSequences: req.Generation.StopSequences,
	}
	if ic.MaxTokens > 0 || ic.Temperature != nil || ic.TopP != nil || len(ic.StopSequences) > 0 {
		out.InferenceConfig = &ic
	}

	msgs, err := messagesToConverse(req.Messages)
	if err != nil {
		return converseRequest{}, err
	}
	out.Messages = msgs

	if len(req.ProviderTools) > 0 {
		return converseRequest{}, errors.New("luft: bedrock: ProviderTools (category-1 server tools) are not supported")
	}
	if len(req.Tools) > 0 {
		tools, err := buildToolConfig(req.Tools)
		if err != nil {
			return converseRequest{}, err
		}
		out.ToolConfig = tools
	}
	return out, nil
}

func buildToolConfig(tools []luft.Tool) (*converseToolConfig, error) {
	entries := make([]converseToolEntry, 0, len(tools))
	for _, t := range tools {
		if t.Provider != "" && t.Provider != ProviderTag {
			return nil, fmt.Errorf("luft: bedrock: tool %q is tagged for provider %q", t.Name, t.Provider)
		}
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		entries = append(entries, converseToolEntry{
			ToolSpec: &converseToolSpec{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: converseInputShape{JSON: schema},
			},
		})
	}
	return &converseToolConfig{Tools: entries}, nil
}

// messagesToConverse maps luft Messages to Converse messages. Tool results
// inside a luft user message become toolResult blocks; image attachments
// become image blocks. Unknown block types are dropped.
func messagesToConverse(in []luft.Message) ([]converseMessage, error) {
	out := make([]converseMessage, 0, len(in))
	for _, m := range in {
		blocks := make([]converseContentBlock, 0, len(m.Content))
		for _, c := range m.Content {
			switch c.Type {
			case luft.TypeText:
				if c.Text == "" {
					continue
				}
				blocks = append(blocks, converseContentBlock{Text: c.Text})
			case luft.TypeImage:
				img, err := luftImageToConverse(c)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, converseContentBlock{Image: img})
			case luft.TypeToolUse:
				input := c.Input
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				blocks = append(blocks, converseContentBlock{
					ToolUse: &converseToolUse{
						ToolUseID: c.ID,
						Name:      c.Name,
						Input:     input,
					},
				})
			case luft.TypeToolResult:
				status := ""
				if c.IsError {
					status = "error"
				}
				blocks = append(blocks, converseContentBlock{
					ToolResult: &converseToolResult{
						ToolUseID: c.ToolUseID,
						Content:   []converseToolResultBlock{{Text: c.Content}},
						Status:    status,
					},
				})
			}
		}
		if len(blocks) == 0 {
			continue
		}
		out = append(out, converseMessage{Role: m.Role, Content: blocks})
	}
	return out, nil
}

func luftImageToConverse(c luft.ContentBlock) (*converseImage, error) {
	format := imageFormatFromMediaType(c.MediaType)
	if format == "" {
		return nil, fmt.Errorf("luft: bedrock: unsupported image media type %q", c.MediaType)
	}
	if strings.HasPrefix(c.Source, "data:") {
		idx := strings.Index(c.Source, ";base64,")
		if idx < 0 {
			return nil, errors.New("luft: bedrock: image data URI must be base64-encoded")
		}
		return &converseImage{
			Format: format,
			Source: converseImageSource{Bytes: c.Source[idx+len(";base64,"):]},
		}, nil
	}
	return nil, errors.New("luft: bedrock: only base64 data-URI images are supported (got remote URL)")
}

func imageFormatFromMediaType(mt string) string {
	switch strings.ToLower(mt) {
	case "image/png":
		return "png"
	case "image/jpeg", "image/jpg":
		return "jpeg"
	case "image/gif":
		return "gif"
	case "image/webp":
		return "webp"
	}
	return ""
}

// converseToLuft converts the message in a Converse response into luft
// ContentBlocks.
func converseToLuft(m converseMessage) []luft.ContentBlock {
	out := make([]luft.ContentBlock, 0, len(m.Content))
	for _, c := range m.Content {
		switch {
		case c.Text != "":
			out = append(out, luft.ContentBlock{Type: luft.TypeText, Text: c.Text})
		case c.ToolUse != nil:
			input := c.ToolUse.Input
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			out = append(out, luft.ContentBlock{
				Type:  luft.TypeToolUse,
				ID:    c.ToolUse.ToolUseID,
				Name:  c.ToolUse.Name,
				Input: input,
			})
		}
	}
	return out
}

// ---- Call / Stream ----

// Call implements luft.Provider.Call.
func (p *Provider) Call(ctx context.Context, req luft.ProviderRequest) (luft.ProviderResponse, error) {
	body, httpReq, err := p.buildHTTP(ctx, req, "converse")
	if err != nil {
		return luft.ProviderResponse{}, err
	}
	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: bedrock: http: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return luft.ProviderResponse{}, decodeError(resp)
	}
	_ = body // retained for symmetry; signing already used it

	var parsed converseResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: bedrock: decode response: %w", err)
	}
	return luft.ProviderResponse{
		Content:    converseToLuft(parsed.Output.Message),
		StopReason: parsed.StopReason,
		Usage:      luft.Usage{InputTokens: parsed.Usage.InputTokens, OutputTokens: parsed.Usage.OutputTokens},
	}, nil
}

// Stream implements luft.Provider.Stream over Bedrock ConverseStream. Events
// arrive as AWS binary event-stream frames whose payloads carry JSON deltas.
func (p *Provider) Stream(ctx context.Context, req luft.ProviderRequest, onDelta func(luft.ContentBlock)) (luft.ProviderResponse, error) {
	_, httpReq, err := p.buildHTTP(ctx, req, "converse-stream")
	if err != nil {
		return luft.ProviderResponse{}, err
	}
	httpReq.Header.Set("Accept", "application/vnd.amazon.eventstream")
	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: bedrock: stream http: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return luft.ProviderResponse{}, decodeError(resp)
	}

	reader := newEventStreamReader(resp.Body)

	// Per-block accumulation, keyed by contentBlock index.
	type blockState struct {
		text      strings.Builder
		toolID    string
		toolName  string
		toolInput strings.Builder
		isToolUse bool
	}
	blocks := map[int]*blockState{}
	get := func(i int) *blockState {
		if b, ok := blocks[i]; ok {
			return b
		}
		b := &blockState{}
		blocks[i] = b
		return b
	}

	var stopReason string
	var usage luft.Usage

	for {
		msg, err := reader.Read()
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			break
		}
		if err != nil {
			return luft.ProviderResponse{}, fmt.Errorf("luft: bedrock: stream decode: %w", err)
		}
		eventType := msg.Headers[":event-type"]
		messageType := msg.Headers[":message-type"]
		if messageType == "exception" {
			var eb errorBody
			_ = json.Unmarshal(msg.Payload, &eb)
			t := msg.Headers[":exception-type"]
			if t == "" {
				t = eb.Type
			}
			return luft.ProviderResponse{}, &luft.APIError{Type: t, Message: eb.Message}
		}
		if messageType != "" && messageType != "event" {
			continue
		}

		switch eventType {
		case "contentBlockStart":
			var ev struct {
				ContentBlockIndex int `json:"contentBlockIndex"`
				Start             struct {
					ToolUse *converseToolUse `json:"toolUse,omitempty"`
				} `json:"start"`
			}
			if err := json.Unmarshal(msg.Payload, &ev); err != nil {
				continue
			}
			b := get(ev.ContentBlockIndex)
			if ev.Start.ToolUse != nil {
				b.isToolUse = true
				b.toolID = ev.Start.ToolUse.ToolUseID
				b.toolName = ev.Start.ToolUse.Name
				onDelta(luft.ContentBlock{Type: luft.TypeToolUse, ID: b.toolID, Name: b.toolName})
			}
		case "contentBlockDelta":
			var ev struct {
				ContentBlockIndex int `json:"contentBlockIndex"`
				Delta             struct {
					Text    string `json:"text,omitempty"`
					ToolUse *struct {
						Input string `json:"input"`
					} `json:"toolUse,omitempty"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(msg.Payload, &ev); err != nil {
				continue
			}
			b := get(ev.ContentBlockIndex)
			if ev.Delta.Text != "" {
				b.text.WriteString(ev.Delta.Text)
				onDelta(luft.ContentBlock{Type: luft.TypeText, Text: ev.Delta.Text})
			}
			if ev.Delta.ToolUse != nil && ev.Delta.ToolUse.Input != "" {
				b.toolInput.WriteString(ev.Delta.ToolUse.Input)
				if b.toolID != "" {
					onDelta(luft.ContentBlock{
						Type:  luft.TypeToolUse,
						ID:    b.toolID,
						Name:  b.toolName,
						Input: json.RawMessage(b.toolInput.String()),
					})
				}
			}
		case "messageStop":
			var ev struct {
				StopReason string `json:"stopReason"`
			}
			_ = json.Unmarshal(msg.Payload, &ev)
			stopReason = ev.StopReason
		case "metadata":
			var ev struct {
				Usage struct {
					InputTokens  int `json:"inputTokens"`
					OutputTokens int `json:"outputTokens"`
				} `json:"usage"`
			}
			_ = json.Unmarshal(msg.Payload, &ev)
			usage.InputTokens = ev.Usage.InputTokens
			usage.OutputTokens = ev.Usage.OutputTokens
		}
	}

	// Assemble final content in index order.
	final := make([]luft.ContentBlock, 0, len(blocks))
	for i := 0; i < len(blocks); i++ {
		b, ok := blocks[i]
		if !ok {
			continue
		}
		if b.isToolUse {
			input := json.RawMessage(`{}`)
			if s := b.toolInput.String(); s != "" {
				input = json.RawMessage(s)
			}
			final = append(final, luft.ContentBlock{
				Type:  luft.TypeToolUse,
				ID:    b.toolID,
				Name:  b.toolName,
				Input: input,
			})
		} else if b.text.Len() > 0 {
			final = append(final, luft.ContentBlock{Type: luft.TypeText, Text: b.text.String()})
		}
	}
	if stopReason == "" {
		stopReason = "end_turn"
	}
	return luft.ProviderResponse{Content: final, StopReason: stopReason, Usage: usage}, nil
}

// buildHTTP constructs and signs the HTTP request for a Converse call.
// action is "converse" or "converse-stream".
func (p *Provider) buildHTTP(ctx context.Context, req luft.ProviderRequest, action string) ([]byte, *http.Request, error) {
	if req.Model == "" {
		return nil, nil, errors.New("luft: bedrock: model is required")
	}
	wire, err := buildConverseRequest(req)
	if err != nil {
		return nil, nil, err
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, nil, fmt.Errorf("luft: bedrock: marshal request: %w", err)
	}

	// Model IDs contain ':' (e.g. "anthropic.claude-sonnet-4-5-20250929-v1:0")
	// which must be percent-encoded in the wire URL. net/url's PathEscape
	// leaves ':' alone in path-segment mode, so use awsURIEncode and force
	// the encoded form via RawPath.
	u, err := url.Parse(p.cfg.BaseURL)
	if err != nil {
		return nil, nil, fmt.Errorf("luft: bedrock: parse base URL: %w", err)
	}
	u.Path = "/model/" + req.Model + "/" + action
	u.RawPath = "/model/" + awsURIEncode(req.Model, true) + "/" + action

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(body))
	if err != nil {
		return nil, nil, fmt.Errorf("luft: bedrock: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if action == "converse" {
		httpReq.Header.Set("Accept", "application/json")
	}

	signRequest(httpReq,
		awsCredentials{
			AccessKeyID:     p.cfg.Credentials.AccessKeyID,
			SecretAccessKey: p.cfg.Credentials.SecretAccessKey,
			SessionToken:    p.cfg.Credentials.SessionToken,
		},
		bedrockService, p.cfg.Region, body, p.cfg.Now())
	return body, httpReq, nil
}

func decodeError(resp *http.Response) error {
	data, _ := io.ReadAll(resp.Body)
	var eb errorBody
	_ = json.Unmarshal(data, &eb)
	msg := eb.Message
	if msg == "" {
		msg = strings.TrimSpace(string(data))
	}
	return &luft.APIError{
		StatusCode: resp.StatusCode,
		Type:       eb.Type,
		Message:    msg,
		RetryAfter: luft.ParseRetryAfter(resp.Header),
	}
}

func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}
