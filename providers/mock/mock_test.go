package mock

import (
	"context"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
)

// textResponse is a small helper for building canned end_turn responses.
func textResponse(text string, usage luft.Usage) luft.ProviderResponse {
	return luft.ProviderResponse{
		Content:    []luft.ContentBlock{{Type: luft.TypeText, Text: text}},
		StopReason: "end_turn",
		Usage:      usage,
	}
}

func newAgent(p *Provider) luft.Agent {
	client, err := luft.New(luft.Config{Provider: p, Model: "test-model"})
	if err != nil {
		panic(err)
	}
	return luft.Agent{Client: client, System: "be helpful"}
}

func TestMock_DrivesAgentTwoTurnToolUse(t *testing.T) {
	ctx := context.Background()

	// Script: turn 1 the model calls a tool; turn 2 it finishes with text.
	toolUse := luft.ProviderResponse{
		Content: []luft.ContentBlock{{
			Type: luft.TypeToolUse,
			ID:   "tu_1", Name: "ping", Input: []byte(`{}`),
		}},
		StopReason: "tool_use",
		Usage:      luft.Usage{InputTokens: 10, OutputTokens: 1},
	}
	finish := textResponse("done", luft.Usage{InputTokens: 12, OutputTokens: 3})

	p := New(toolUse, finish)

	var pingCalled bool
	tool, fn := luft.NewTypedTool[struct{}](
		"ping", "returns pong",
		luft.Object(),
		func(_ context.Context, _ struct{}) (string, error) {
			pingCalled = true
			return "pong", nil
		},
	)
	a := newAgent(p)
	a.Tools = luft.Tools(luft.Bind(tool, fn))

	result, err := a.Step(ctx, []luft.Message{luft.NewUserMessage("run ping")})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if !pingCalled {
		t.Error("expected the ping tool to be dispatched")
	}
	if got := result.FinalText(); got != "done" {
		t.Fatalf("final text = %q, want %q", got, "done")
	}
	if got, want := result.Usage.InputTokens, 22; got != want {
		t.Errorf("input tokens = %d, want %d", got, want)
	}
	if got, want := result.Usage.OutputTokens, 4; got != want {
		t.Errorf("output tokens = %d, want %d", got, want)
	}

	// Two model calls were made: the tool_use turn and the end turn.
	reqs := p.Requests()
	if len(reqs) != 2 {
		t.Fatalf("Requests() captured %d calls, want 2", len(reqs))
	}
	if got := reqs[0].System; got != "be helpful" {
		t.Errorf("request 0 system = %q, want %q", got, "be helpful")
	}
	if got := reqs[0].Model; got != "test-model" {
		t.Errorf("request 0 model = %q, want test-model", got)
	}
	// First request carries just the user message; second carries the
	// user + assistant tool_use + tool_result.
	if len(reqs[0].Messages) != 1 {
		t.Errorf("request 0 had %d messages, want 1", len(reqs[0].Messages))
	}
	if len(reqs[1].Messages) != 3 {
		t.Errorf("request 1 had %d messages, want 3 (user, assistant tool_use, tool_result)", len(reqs[1].Messages))
	}
	if reqs[1].Messages[1].Content[0].Type != luft.TypeToolUse {
		t.Error("request 1 message 1 should be the assistant tool_use turn")
	}
	if reqs[1].Messages[2].Content[0].Type != luft.TypeToolResult {
		t.Error("request 1 message 2 should be the tool_result turn")
	}
	// The toolset must have been advertised to the model on every call.
	if len(reqs[0].Tools) != 1 || reqs[0].Tools[0].Name != "ping" {
		t.Errorf("request 0 tools = %+v, want one tool named ping", reqs[0].Tools)
	}
	if len(reqs[1].Tools) != 1 || reqs[1].Tools[0].Name != "ping" {
		t.Errorf("request 1 tools = %+v, want one tool named ping", reqs[1].Tools)
	}
}

func TestMock_StreamReplaysTextDeltas(t *testing.T) {
	ctx := context.Background()

	p := New(textResponse("hello world", luft.Usage{InputTokens: 1, OutputTokens: 2}))
	client, err := luft.New(luft.Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}

	var deltas []luft.ContentBlock
	msg, usage, err := client.AskStream(ctx, "sys", []luft.Message{luft.NewUserMessage("hi")},
		func(b luft.ContentBlock) { deltas = append(deltas, b) },
	)
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	if got := luft.TextContent(msg); got != "hello world" {
		t.Fatalf("streamed message text = %q, want %q", got, "hello world")
	}
	if usage.OutputTokens != 2 {
		t.Errorf("usage = %+v, want 2 output tokens", usage)
	}
	// A single text block is delivered as one text delta.
	if len(deltas) != 1 || deltas[0].Type != luft.TypeText || deltas[0].Text != "hello world" {
		t.Fatalf("deltas = %+v, want one text delta \"hello world\"", deltas)
	}
}

func TestMock_StreamingLoopMatchesNonStreaming(t *testing.T) {
	ctx := context.Background()

	toolUse := luft.ProviderResponse{
		Content: []luft.ContentBlock{{
			Type: luft.TypeToolUse, ID: "tu_1", Name: "ping", Input: []byte(`{}`),
		}},
		StopReason: "tool_use",
		Usage:      luft.Usage{InputTokens: 7},
	}
	finish := textResponse("streamed-done", luft.Usage{InputTokens: 9, OutputTokens: 4})

	p := New(toolUse, finish)

	var deltas []luft.ContentBlock
	a := newAgent(p)
	tool, fn := luft.NewTypedTool[struct{}]("ping", "returns pong", luft.Object(),
		func(context.Context, struct{}) (string, error) { return "pong", nil })
	a.Tools = luft.Tools(luft.Bind(tool, fn))

	result, err := a.StepStream(ctx,
		[]luft.Message{luft.NewUserMessage("go")},
		func(b luft.ContentBlock) { deltas = append(deltas, b) },
		nil,
	)
	if err != nil {
		t.Fatalf("StepStream: %v", err)
	}
	if got := result.FinalText(); got != "streamed-done" {
		t.Fatalf("final text = %q, want %q", got, "streamed-done")
	}
	// The finish turn's text block is delivered via the delta callback.
	if !containsText(deltas, "streamed-done") {
		t.Errorf("deltas = %+v, want one containing the final text", deltas)
	}
	if len(p.Requests()) != 2 {
		t.Errorf("captured %d requests, want 2", len(p.Requests()))
	}
}

func TestMock_ExhaustedReturnsError(t *testing.T) {
	ctx := context.Background()

	p := New(textResponse("only one", luft.Usage{}))
	a := newAgent(p)

	// First step consumes the single response.
	if _, err := a.Step(ctx, []luft.Message{luft.NewUserMessage("first")}); err != nil {
		t.Fatalf("first Step: %v", err)
	}
	// Second step has nothing to return; the loop's first model call errors.
	_, err := a.Step(ctx, []luft.Message{luft.NewUserMessage("second")})
	if err == nil {
		t.Fatal("expected error when scripted responses are exhausted, got nil")
	}
	if !strings.Contains(err.Error(), "exhausted") {
		t.Errorf("error = %q, want it to mention exhaustion", err.Error())
	}
	// The exhausted call was still captured.
	if len(p.Requests()) != 2 {
		t.Errorf("captured %d requests, want 2 (including the failed one)", len(p.Requests()))
	}
}

func TestMock_ResetClearsState(t *testing.T) {
	p := New(textResponse("a", luft.Usage{}), textResponse("b", luft.Usage{}))
	client, err := luft.New(luft.Config{Provider: p, Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	_, _, _ = client.Ask(ctx, "", []luft.Message{luft.NewUserMessage("x")})
	if len(p.Requests()) != 1 {
		t.Fatalf("after one call, want 1 request, got %d", len(p.Requests()))
	}
	p.Reset()
	if len(p.Requests()) != 0 {
		t.Fatalf("after Reset, want 0 requests, got %d", len(p.Requests()))
	}
	// Cursor is back at the start, so the first canned response ("a") returns.
	msg, _, err := client.Ask(ctx, "", []luft.Message{luft.NewUserMessage("y")})
	if err != nil {
		t.Fatal(err)
	}
	if luft.TextContent(msg) != "a" {
		t.Errorf("after Reset, first response = %q, want %q", luft.TextContent(msg), "a")
	}
}

func TestMock_CallErrPropagatesThroughStream(t *testing.T) {
	ctx := context.Background()
	p := New() // no responses
	_, err := p.Stream(ctx, luft.ProviderRequest{Model: "m"}, func(luft.ContentBlock) {})
	if err == nil {
		t.Fatal("expected error from empty Stream")
	}
	if !strings.Contains(err.Error(), "exhausted") {
		t.Errorf("error = %q, want exhaustion message", err.Error())
	}
}

func containsText(blocks []luft.ContentBlock, want string) bool {
	for _, b := range blocks {
		if b.Type == luft.TypeText && b.Text == want {
			return true
		}
	}
	return false
}
