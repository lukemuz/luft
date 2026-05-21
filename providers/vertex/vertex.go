// Package vertex is a luft provider for Vertex AI Gemini.
//
// Auth uses a Google service-account JSON (RS256-signed JWT → OAuth2 access
// token exchange). Wire format is Gemini's generateContent / streamGenerateContent
// against the regional Vertex AI endpoint. Streaming uses SSE.
//
// Gemini's wire format differs from Anthropic / OpenAI in two notable ways:
// (1) the assistant role is "model" rather than "assistant"; (2) function
// calls have no IDs of their own — luft synthesizes them as "<name>::<seq>"
// so tool_result blocks coming back through the loop can be mapped to a
// functionResponse by name.
package vertex

import (
	"bufio"
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
	defaultHTTPTimeout = 60 * time.Second
	defaultLocation    = "us-central1"

	// ProviderTag is the value luft.Tool.Provider and luft.ProviderTool.Provider
	// must carry for an entry to be accepted by this provider.
	ProviderTag = "vertex"

	// toolIDSep is the separator used in synthesized tool_use IDs
	// ("<name><sep><seq>"). The separator must not appear in Gemini tool
	// names (which match the function-naming pattern [A-Za-z_][A-Za-z0-9_-]*).
	toolIDSep = "::"
)

// Config holds configuration for the Vertex AI Gemini provider.
type Config struct {
	// Project is the GCP project ID. Required; if empty, NewProvider reads
	// GOOGLE_CLOUD_PROJECT, then the project_id field of the service-account
	// JSON.
	Project string

	// Location is the Vertex AI region (e.g. "us-central1"). Pass "global"
	// for the multi-region endpoint. Defaults to "us-central1".
	Location string

	// ServiceAccount is the parsed service-account credentials. If zero,
	// NewProvider loads from the path in GOOGLE_APPLICATION_CREDENTIALS.
	ServiceAccount ServiceAccount

	// TokenSource overrides the default service-account flow. Supply this
	// to use a caller-managed access token (gcloud auth print-access-token,
	// workload identity, ADC handled externally). When set, ServiceAccount
	// is ignored.
	TokenSource TokenSource

	// BaseURL overrides the default <location>-aiplatform.googleapis.com
	// endpoint. Set to an httptest.Server URL in tests.
	BaseURL string

	// HTTPClient lets you supply a custom client. Defaults to a
	// 60-second-timeout client.
	HTTPClient *http.Client

	// Now, if set, supplies the time used for JWT exp/iat and token
	// cache expiry. Defaults to time.Now.
	Now func() time.Time
}

// Provider implements luft.Provider for Vertex AI Gemini.
type Provider struct {
	cfg    Config
	tokens TokenSource
}

// NewProvider creates a Vertex provider, filling in defaults. Returns an
// error if Project cannot be resolved or no credentials are available.
func NewProvider(cfg Config) (*Provider, error) {
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: defaultHTTPTimeout}
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Location == "" {
		cfg.Location = defaultLocation
	}

	var tokens TokenSource
	switch {
	case cfg.TokenSource != nil:
		tokens = cfg.TokenSource
	case cfg.ServiceAccount.ClientEmail != "":
		ts, err := newServiceAccountTokenSource(cfg.ServiceAccount, cfg.HTTPClient, cfg.Now)
		if err != nil {
			return nil, err
		}
		tokens = ts
	default:
		path := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
		if path == "" {
			return nil, errors.New("luft: vertex: no credentials (set GOOGLE_APPLICATION_CREDENTIALS, pass Config.ServiceAccount, or pass Config.TokenSource)")
		}
		sa, err := LoadServiceAccountFile(path)
		if err != nil {
			return nil, err
		}
		cfg.ServiceAccount = sa
		ts, err := newServiceAccountTokenSource(sa, cfg.HTTPClient, cfg.Now)
		if err != nil {
			return nil, err
		}
		tokens = ts
	}

	if cfg.Project == "" {
		cfg.Project = os.Getenv("GOOGLE_CLOUD_PROJECT")
		if cfg.Project == "" {
			cfg.Project = cfg.ServiceAccount.ProjectID
		}
		if cfg.Project == "" {
			return nil, errors.New("luft: vertex: project not set (pass Config.Project or set GOOGLE_CLOUD_PROJECT)")
		}
	}

	if cfg.BaseURL == "" {
		if cfg.Location == "global" {
			cfg.BaseURL = "https://aiplatform.googleapis.com"
		} else {
			cfg.BaseURL = fmt.Sprintf("https://%s-aiplatform.googleapis.com", cfg.Location)
		}
	}

	return &Provider{cfg: cfg, tokens: tokens}, nil
}

// NewProviderFromEnv builds a provider from GOOGLE_APPLICATION_CREDENTIALS
// and GOOGLE_CLOUD_PROJECT (the latter is optional if the service account
// JSON carries project_id).
func NewProviderFromEnv() (*Provider, error) {
	return NewProvider(Config{})
}

// NewClientFromEnv builds a luft.Client backed by a Vertex provider
// configured from the environment. model is the Gemini model id
// (e.g. "gemini-2.5-pro", "gemini-2.0-flash-001").
func NewClientFromEnv(model string) (*luft.Client, error) {
	p, err := NewProviderFromEnv()
	if err != nil {
		return nil, err
	}
	return luft.New(luft.Config{Provider: p, Model: model})
}

// NewProviderWithToken builds a provider using a static access token. Use
// this when token lifecycle is managed externally (gcloud, workload
// identity, ADC). project is required; location defaults to "us-central1".
func NewProviderWithToken(project, location, token string) (*Provider, error) {
	return NewProvider(Config{
		Project:     project,
		Location:    location,
		TokenSource: staticTokenSource{tok: token},
	})
}

// ---- wire types ----

type generateRequest struct {
	Contents          []geminiContent      `json:"contents"`
	SystemInstruction *geminiContent       `json:"systemInstruction,omitempty"`
	GenerationConfig  *generationConfig    `json:"generationConfig,omitempty"`
	Tools             []geminiToolDecl     `json:"tools,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

// geminiPart is a discriminated union by field presence: exactly one of Text,
// InlineData, FunctionCall, or FunctionResponse is populated.
type geminiPart struct {
	Text             string              `json:"text,omitempty"`
	InlineData       *geminiInlineData   `json:"inlineData,omitempty"`
	FunctionCall     *geminiFunctionCall `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResp `json:"functionResponse,omitempty"`
}

type geminiInlineData struct {
	MIMEType string `json:"mimeType"`
	Data     string `json:"data"` // base64
}

type geminiFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type geminiFunctionResp struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type generationConfig struct {
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
}

type geminiToolDecl struct {
	FunctionDeclarations []geminiFunctionDecl `json:"functionDeclarations"`
}

type geminiFunctionDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type generateResponse struct {
	Candidates    []geminiCandidate `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
	Index        int           `json:"index"`
}

type geminiAPIError struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

// ---- request build / response parse ----

func buildGenerateRequest(req luft.ProviderRequest) (generateRequest, error) {
	if len(req.ProviderTools) > 0 {
		return generateRequest{}, errors.New("luft: vertex: ProviderTools are not supported")
	}

	out := generateRequest{}
	if req.System != "" {
		out.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: req.System}}}
	}
	if req.MaxTokens > 0 {
		out.GenerationConfig = &generationConfig{MaxOutputTokens: req.MaxTokens}
	}

	contents, err := messagesToGemini(req.Messages)
	if err != nil {
		return generateRequest{}, err
	}
	out.Contents = contents

	if len(req.Tools) > 0 {
		decls, err := buildFunctionDecls(req.Tools)
		if err != nil {
			return generateRequest{}, err
		}
		out.Tools = []geminiToolDecl{{FunctionDeclarations: decls}}
	}
	return out, nil
}

func buildFunctionDecls(tools []luft.Tool) ([]geminiFunctionDecl, error) {
	out := make([]geminiFunctionDecl, 0, len(tools))
	for _, t := range tools {
		if t.Provider != "" && t.Provider != ProviderTag {
			return nil, fmt.Errorf("luft: vertex: tool %q is tagged for provider %q", t.Name, t.Provider)
		}
		out = append(out, geminiFunctionDecl{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}
	return out, nil
}

// messagesToGemini maps luft Messages to Gemini contents. Tool results in
// luft user messages become functionResponse parts; image content blocks
// become inlineData parts. The luft "assistant" role maps to Gemini "model".
func messagesToGemini(msgs []luft.Message) ([]geminiContent, error) {
	out := make([]geminiContent, 0, len(msgs))
	for _, m := range msgs {
		role := m.Role
		if role == luft.RoleAssistant {
			role = "model"
		}
		parts := make([]geminiPart, 0, len(m.Content))
		for _, c := range m.Content {
			switch c.Type {
			case luft.TypeText:
				if c.Text == "" {
					continue
				}
				parts = append(parts, geminiPart{Text: c.Text})
			case luft.TypeImage:
				inline, err := imageToInlineData(c)
				if err != nil {
					return nil, err
				}
				parts = append(parts, geminiPart{InlineData: inline})
			case luft.TypeToolUse:
				args := c.Input
				if len(args) == 0 {
					args = json.RawMessage(`{}`)
				}
				parts = append(parts, geminiPart{FunctionCall: &geminiFunctionCall{
					Name: c.Name,
					Args: args,
				}})
			case luft.TypeToolResult:
				name := nameFromToolUseID(c.ToolUseID)
				respJSON := toolResultResponse(c.Content, c.IsError)
				parts = append(parts, geminiPart{FunctionResponse: &geminiFunctionResp{
					Name:     name,
					Response: respJSON,
				}})
			}
		}
		if len(parts) == 0 {
			continue
		}
		out = append(out, geminiContent{Role: role, Parts: parts})
	}
	return out, nil
}

// nameFromToolUseID strips the "::N" suffix synthesized in candidateToLuft.
// If the ID has no separator (an older or external ID), the whole string is
// returned and used directly.
func nameFromToolUseID(id string) string {
	if idx := strings.LastIndex(id, toolIDSep); idx >= 0 {
		return id[:idx]
	}
	return id
}

// toolResultResponse wraps a luft tool-result string into the JSON object
// Gemini's functionResponse.response expects.
func toolResultResponse(content string, isError bool) json.RawMessage {
	if isError {
		b, _ := json.Marshal(map[string]any{"error": content})
		return b
	}
	b, _ := json.Marshal(map[string]string{"output": content})
	return b
}

func imageToInlineData(c luft.ContentBlock) (*geminiInlineData, error) {
	if c.MediaType == "" {
		return nil, errors.New("luft: vertex: image block missing MediaType")
	}
	if strings.HasPrefix(c.Source, "data:") {
		idx := strings.Index(c.Source, ";base64,")
		if idx < 0 {
			return nil, errors.New("luft: vertex: image data URI must be base64-encoded")
		}
		return &geminiInlineData{MIMEType: c.MediaType, Data: c.Source[idx+len(";base64,"):]}, nil
	}
	return nil, errors.New("luft: vertex: only base64 data-URI images are supported")
}

// candidateToLuft converts the first candidate's content into luft
// ContentBlocks, synthesizing IDs for any functionCall parts so they can be
// matched on the way back.
func candidateToLuft(c geminiContent) []luft.ContentBlock {
	out := make([]luft.ContentBlock, 0, len(c.Parts))
	for i, p := range c.Parts {
		switch {
		case p.Text != "":
			out = append(out, luft.ContentBlock{Type: luft.TypeText, Text: p.Text})
		case p.FunctionCall != nil:
			args := p.FunctionCall.Args
			if len(args) == 0 {
				args = json.RawMessage(`{}`)
			}
			id := fmt.Sprintf("%s%s%d", p.FunctionCall.Name, toolIDSep, i)
			out = append(out, luft.ContentBlock{
				Type:  luft.TypeToolUse,
				ID:    id,
				Name:  p.FunctionCall.Name,
				Input: args,
			})
		}
	}
	return out
}

// mapFinishReason translates Gemini's finishReason strings to luft's
// stop-reason vocabulary. The agent loop only cares whether a turn is final;
// an empty string falls through to "end_turn" in the caller.
func mapFinishReason(reason string, hasToolUse bool) string {
	if hasToolUse {
		return "tool_use"
	}
	switch reason {
	case "STOP":
		return "end_turn"
	case "MAX_TOKENS":
		return "max_tokens"
	}
	return strings.ToLower(reason)
}

// ---- Call / Stream ----

// Call implements luft.Provider.Call.
func (p *Provider) Call(ctx context.Context, req luft.ProviderRequest) (luft.ProviderResponse, error) {
	body, httpReq, err := p.buildHTTP(ctx, req, "generateContent", "")
	if err != nil {
		return luft.ProviderResponse{}, err
	}
	_ = body

	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: vertex: http: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return luft.ProviderResponse{}, decodeError(resp)
	}

	var parsed generateResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: vertex: decode response: %w", err)
	}
	if len(parsed.Candidates) == 0 {
		return luft.ProviderResponse{}, errors.New("luft: vertex: response had no candidates")
	}

	cand := parsed.Candidates[0]
	content := candidateToLuft(cand.Content)
	hasToolUse := false
	for _, b := range content {
		if b.Type == luft.TypeToolUse {
			hasToolUse = true
			break
		}
	}
	stop := mapFinishReason(cand.FinishReason, hasToolUse)
	if stop == "" {
		stop = "end_turn"
	}

	return luft.ProviderResponse{
		Content:    content,
		StopReason: stop,
		Usage: luft.Usage{
			InputTokens:  parsed.UsageMetadata.PromptTokenCount,
			OutputTokens: parsed.UsageMetadata.CandidatesTokenCount,
		},
	}, nil
}

// Stream implements luft.Provider.Stream. Vertex SSE events are full
// generateContent response shapes, accumulated across events. Text parts are
// emitted as deltas as they arrive; functionCall parts arrive complete in
// one event and are emitted then.
func (p *Provider) Stream(ctx context.Context, req luft.ProviderRequest, onDelta func(luft.ContentBlock)) (luft.ProviderResponse, error) {
	_, httpReq, err := p.buildHTTP(ctx, req, "streamGenerateContent", "alt=sse")
	if err != nil {
		return luft.ProviderResponse{}, err
	}
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: vertex: stream http: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return luft.ProviderResponse{}, decodeError(resp)
	}

	var (
		textBuilder  strings.Builder
		toolBlocks   []luft.ContentBlock
		finishReason string
		usage        luft.Usage
		toolSeq      int
	)

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if strings.TrimSpace(data) == "" {
			continue
		}

		var ev generateResponse
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		if ev.UsageMetadata.TotalTokenCount > 0 {
			usage.InputTokens = ev.UsageMetadata.PromptTokenCount
			usage.OutputTokens = ev.UsageMetadata.CandidatesTokenCount
		}
		if len(ev.Candidates) == 0 {
			continue
		}
		cand := ev.Candidates[0]
		for _, part := range cand.Content.Parts {
			switch {
			case part.Text != "":
				textBuilder.WriteString(part.Text)
				onDelta(luft.ContentBlock{Type: luft.TypeText, Text: part.Text})
			case part.FunctionCall != nil:
				args := part.FunctionCall.Args
				if len(args) == 0 {
					args = json.RawMessage(`{}`)
				}
				id := fmt.Sprintf("%s%s%d", part.FunctionCall.Name, toolIDSep, toolSeq)
				toolSeq++
				blk := luft.ContentBlock{
					Type:  luft.TypeToolUse,
					ID:    id,
					Name:  part.FunctionCall.Name,
					Input: args,
				}
				toolBlocks = append(toolBlocks, blk)
				onDelta(blk)
			}
		}
		if cand.FinishReason != "" {
			finishReason = cand.FinishReason
		}
	}
	if err := scanner.Err(); err != nil {
		return luft.ProviderResponse{}, fmt.Errorf("luft: vertex: stream read: %w", err)
	}

	content := make([]luft.ContentBlock, 0, len(toolBlocks)+1)
	if textBuilder.Len() > 0 {
		content = append(content, luft.ContentBlock{Type: luft.TypeText, Text: textBuilder.String()})
	}
	content = append(content, toolBlocks...)

	stop := mapFinishReason(finishReason, len(toolBlocks) > 0)
	if stop == "" {
		stop = "end_turn"
	}
	return luft.ProviderResponse{Content: content, StopReason: stop, Usage: usage}, nil
}

func (p *Provider) buildHTTP(ctx context.Context, req luft.ProviderRequest, action, query string) ([]byte, *http.Request, error) {
	if req.Model == "" {
		return nil, nil, errors.New("luft: vertex: model is required")
	}
	wire, err := buildGenerateRequest(req)
	if err != nil {
		return nil, nil, err
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, nil, fmt.Errorf("luft: vertex: marshal request: %w", err)
	}

	// Path: /v1/projects/<project>/locations/<location>/publishers/google/models/<model>:<action>
	// Gemini model IDs don't contain ':' so simple concatenation is safe; we
	// PathEscape the model defensively.
	path := fmt.Sprintf("/v1/projects/%s/locations/%s/publishers/google/models/%s:%s",
		url.PathEscape(p.cfg.Project),
		url.PathEscape(p.cfg.Location),
		url.PathEscape(req.Model),
		action,
	)
	endpoint := p.cfg.BaseURL + path
	if query != "" {
		endpoint += "?" + query
	}

	token, err := p.tokens.Token(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("luft: vertex: token: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, nil, fmt.Errorf("luft: vertex: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)
	return body, httpReq, nil
}

func decodeError(resp *http.Response) error {
	data, _ := io.ReadAll(resp.Body)
	var eb geminiAPIError
	_ = json.Unmarshal(data, &eb)
	msg := eb.Error.Message
	if msg == "" {
		msg = strings.TrimSpace(string(data))
	}
	return &luft.APIError{
		StatusCode: resp.StatusCode,
		Type:       eb.Error.Status,
		Message:    msg,
	}
}
