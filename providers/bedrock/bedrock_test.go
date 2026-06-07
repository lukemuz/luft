package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lukemuz/luft"
)

func testProvider(t *testing.T, baseURL string) *Provider {
	t.Helper()
	p, err := NewProvider(Config{
		Credentials: Credentials{AccessKeyID: "AKID", SecretAccessKey: "secret"},
		Region:      "us-east-1",
		BaseURL:     baseURL,
		Now:         func() time.Time { return time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	return p
}

func TestBuildConverseRequest_TextAndTools(t *testing.T) {
	tool := luft.Tool{
		Name:        "calc",
		Description: "Add two numbers",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}`),
	}

	req := luft.ProviderRequest{
		Model:     "anthropic.claude",
		MaxTokens: 256,
		System:    "be helpful",
		Messages:  []luft.Message{luft.NewUserMessage("hi")},
		Tools:     []luft.Tool{tool},
	}
	wire, err := buildConverseRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(wire.System) != 1 || wire.System[0].Text != "be helpful" {
		t.Errorf("system: %+v", wire.System)
	}
	if wire.InferenceConfig == nil || wire.InferenceConfig.MaxTokens != 256 {
		t.Errorf("inferenceConfig: %+v", wire.InferenceConfig)
	}
	if wire.ToolConfig == nil || len(wire.ToolConfig.Tools) != 1 {
		t.Fatalf("toolConfig: %+v", wire.ToolConfig)
	}
	spec := wire.ToolConfig.Tools[0].ToolSpec
	if spec.Name != "calc" || spec.Description != "Add two numbers" {
		t.Errorf("toolSpec: %+v", spec)
	}
	if !bytes.Equal(spec.InputSchema.JSON, tool.InputSchema) {
		t.Errorf("schema mismatch: %s", spec.InputSchema.JSON)
	}
}

func TestMessagesToConverse_ToolFlow(t *testing.T) {
	msgs := []luft.Message{
		luft.NewUserMessage("calculate"),
		{
			Role: luft.RoleAssistant,
			Content: []luft.ContentBlock{
				{Type: luft.TypeText, Text: "let me try"},
				{Type: luft.TypeToolUse, ID: "tu_1", Name: "calc", Input: json.RawMessage(`{"a":1,"b":2}`)},
			},
		},
		luft.NewToolResultMessage([]luft.ToolResult{{ToolUseID: "tu_1", Content: "3"}}),
	}
	out, err := messagesToConverse(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("messages: %d", len(out))
	}
	if out[1].Content[1].ToolUse.ToolUseID != "tu_1" {
		t.Errorf("toolUse mapping: %+v", out[1].Content[1])
	}
	if out[2].Content[0].ToolResult.ToolUseID != "tu_1" {
		t.Errorf("toolResult mapping: %+v", out[2].Content[0])
	}
	if out[2].Content[0].ToolResult.Content[0].Text != "3" {
		t.Errorf("toolResult body: %+v", out[2].Content[0].ToolResult)
	}
}

func TestMessagesToConverse_ToolError(t *testing.T) {
	msg := luft.NewToolResultMessage([]luft.ToolResult{{ToolUseID: "tu_1", Content: "boom", IsError: true}})
	out, err := messagesToConverse([]luft.Message{msg})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Content[0].ToolResult.Status != "error" {
		t.Errorf("status: %q", out[0].Content[0].ToolResult.Status)
	}
}

func TestConverseToLuft_TextAndToolUse(t *testing.T) {
	resp := converseMessage{
		Role: "assistant",
		Content: []converseContentBlock{
			{Text: "hello"},
			{ToolUse: &converseToolUse{ToolUseID: "tu_1", Name: "calc", Input: json.RawMessage(`{"a":1}`)}},
		},
	}
	blocks := converseToLuft(resp)
	if len(blocks) != 2 {
		t.Fatalf("blocks: %d", len(blocks))
	}
	if blocks[0].Text != "hello" {
		t.Errorf("text: %q", blocks[0].Text)
	}
	if blocks[1].Type != luft.TypeToolUse || blocks[1].ID != "tu_1" {
		t.Errorf("toolUse: %+v", blocks[1])
	}
}

func TestProvider_Call(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.EscapedPath()
		gotAuth = r.Header.Get("Authorization")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
            "output":{"message":{"role":"assistant","content":[{"text":"hello back"}]}},
            "stopReason":"end_turn",
            "usage":{"inputTokens":12,"outputTokens":5}
        }`))
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	resp, err := p.Call(context.Background(), luft.ProviderRequest{
		Model:    "anthropic.claude-sonnet-4-5-20250929-v1:0",
		System:   "be helpful",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	// The wire URL must percent-encode the colon in the Bedrock model ID.
	if !strings.Contains(gotPath, "/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse") {
		t.Errorf("wire path missing %%3A encoding: %s", gotPath)
	}
	if !strings.HasPrefix(gotAuth, "AWS4-HMAC-SHA256 ") {
		t.Errorf("auth: %s", gotAuth)
	}
	if !strings.Contains(string(gotBody), "be helpful") {
		t.Errorf("body missing system: %s", gotBody)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "hello back" {
		t.Errorf("content: %+v", resp.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stopReason: %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage: %+v", resp.Usage)
	}
}

func TestProvider_CallError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"__type":"ValidationException","message":"bad model"}`))
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	_, err := p.Call(context.Background(), luft.ProviderRequest{
		Model:    "nope",
		Messages: []luft.Message{luft.NewUserMessage("x")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *luft.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *luft.APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 400 || !strings.Contains(apiErr.Message, "bad model") {
		t.Errorf("apiErr: %+v", apiErr)
	}
}

func TestProvider_Stream(t *testing.T) {
	frames := []byte{}
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "messageStart"},
		[]byte(`{"role":"assistant"}`),
	)...)
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "contentBlockDelta"},
		[]byte(`{"contentBlockIndex":0,"delta":{"text":"Hello"}}`),
	)...)
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "contentBlockDelta"},
		[]byte(`{"contentBlockIndex":0,"delta":{"text":" world"}}`),
	)...)
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "messageStop"},
		[]byte(`{"stopReason":"end_turn"}`),
	)...)
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "metadata"},
		[]byte(`{"usage":{"inputTokens":10,"outputTokens":2}}`),
	)...)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.Write(frames)
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	var deltas []string
	resp, err := p.Stream(context.Background(),
		luft.ProviderRequest{Model: "m", Messages: []luft.Message{luft.NewUserMessage("hi")}},
		func(b luft.ContentBlock) {
			if b.Type == luft.TypeText {
				deltas = append(deltas, b.Text)
			}
		},
	)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if got := strings.Join(deltas, ""); got != "Hello world" {
		t.Errorf("deltas: %q", got)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "Hello world" {
		t.Errorf("final content: %+v", resp.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stopReason: %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 2 {
		t.Errorf("usage: %+v", resp.Usage)
	}
}

func TestProvider_StreamToolUse(t *testing.T) {
	frames := []byte{}
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "contentBlockStart"},
		[]byte(`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"tu_1","name":"calc"}}}`),
	)...)
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "contentBlockDelta"},
		[]byte(`{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"a\":"}}}`),
	)...)
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "contentBlockDelta"},
		[]byte(`{"contentBlockIndex":0,"delta":{"toolUse":{"input":"1}"}}}`),
	)...)
	frames = append(frames, encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "messageStop"},
		[]byte(`{"stopReason":"tool_use"}`),
	)...)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.Write(frames)
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	resp, err := p.Stream(context.Background(),
		luft.ProviderRequest{Model: "m", Messages: []luft.Message{luft.NewUserMessage("hi")}},
		func(b luft.ContentBlock) {},
	)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != luft.TypeToolUse {
		t.Fatalf("expected one tool_use block: %+v", resp.Content)
	}
	if resp.Content[0].ID != "tu_1" || resp.Content[0].Name != "calc" {
		t.Errorf("toolUse id/name: %+v", resp.Content[0])
	}
	if string(resp.Content[0].Input) != `{"a":1}` {
		t.Errorf("toolUse input: %s", resp.Content[0].Input)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stopReason: %q", resp.StopReason)
	}
}

func TestNewProvider_RequiresRegion(t *testing.T) {
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_DEFAULT_REGION", "")
	_, err := NewProvider(Config{Credentials: Credentials{AccessKeyID: "a", SecretAccessKey: "b"}})
	if err == nil || !strings.Contains(err.Error(), "region") {
		t.Errorf("expected region error, got %v", err)
	}
}

func TestNewProvider_RequiresCredentials(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "")
	_, err := NewProvider(Config{Region: "us-east-1"})
	if err == nil || !strings.Contains(err.Error(), "AWS_ACCESS_KEY_ID") {
		t.Errorf("expected credentials error, got %v", err)
	}
}
