package luft

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"
)

type mockProvider struct{}

var testResponse = ProviderResponse{
	Content:    []ContentBlock{{Type: TypeText, Text: "Hello from mock"}},
	StopReason: "end_turn",
	Usage:      Usage{InputTokens: 10, OutputTokens: 5},
}

// testProvider is a configurable mock implementing Provider. Fields control
// responses, errors, and spies for callbacks. Call/StreamCount and DeltaSpy
// allow verifying calls and deltas without side effects on real providers.
// Use newTestProvider() for defaults or set fields directly in tests.
//
// Responses, when non-nil, is consumed in order (FIFO). When exhausted,
// Resp is returned. This allows tests to script multi-turn sequences.
type testProvider struct {
	Resp            ProviderResponse
	Responses       []ProviderResponse // consumed in order; falls back to Resp
	Err             error
	Deltas          []ContentBlock
	DeltaSpy        func(ContentBlock)
	CallCount       int
	StreamCount     int
	OnToolResultSpy func([]ToolResult) // spy for LoopStream (not part of Provider)
}

func newTestProvider() *testProvider {
	return &testProvider{
		Resp: testResponse,
	}
}

func (p *testProvider) Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error) {
	p.CallCount++
	if p.Err != nil {
		return ProviderResponse{}, p.Err
	}
	if len(p.Responses) > 0 {
		resp := p.Responses[0]
		p.Responses = p.Responses[1:]
		return resp, nil
	}
	return p.Resp, nil
}

func (p *testProvider) Stream(ctx context.Context, req ProviderRequest, onDelta func(ContentBlock)) (ProviderResponse, error) {
	p.StreamCount++
	if p.DeltaSpy != nil {
		for _, d := range p.Deltas {
			p.DeltaSpy(d)
			onDelta(d)
		}
	} else {
		for _, d := range p.Deltas {
			onDelta(d)
		}
	}
	if p.Err != nil {
		return ProviderResponse{}, p.Err
	}
	return p.Resp, nil
}

// test helpers (package level so methods are allowed)
type temporaryError struct {
	error
}

func (temporaryError) Temporary() bool { return true }

type nonTemporary struct {
	error
}

func (nonTemporary) Temporary() bool { return false }

func TestNew(t *testing.T) {
	t.Run("requires provider", func(t *testing.T) {
		_, err := New(Config{Model: "claude-sonnet-4-6"})
		if err == nil {
			t.Fatal("expected error for missing Provider")
		}
		if !strings.Contains(err.Error(), "Config.Provider is required") {
			t.Errorf("unexpected error message: %v", err)
		}
	})

	t.Run("requires model", func(t *testing.T) {
		_, err := New(Config{Provider: newTestProvider()})
		if err == nil {
			t.Fatal("expected error for missing Model")
		}
		if !strings.Contains(err.Error(), "Config.Model is required") {
			t.Errorf("unexpected error message: %v", err)
		}
	})

	t.Run("sets default MaxTokens", func(t *testing.T) {
		c, err := New(Config{
			Provider: newTestProvider(),
			Model:    "test-model",
		})
		if err != nil {
			t.Fatal(err)
		}
		if c.cfg.MaxTokens != defaultMaxTokens {
			t.Errorf("expected defaultMaxTokens %d, got %d", defaultMaxTokens, c.cfg.MaxTokens)
		}
	})
}

func TestTextContent(t *testing.T) {
	msg := Message{
		Role: RoleUser,
		Content: []ContentBlock{
			{Type: TypeText, Text: "Hello "},
			{Type: TypeToolUse, ID: "tu_1", Name: "calc"},
			{Type: TypeText, Text: "world!"},
			{Type: TypeToolResult, Content: "ignored"},
		},
	}
	if got := TextContent(msg); got != "Hello world!" {
		t.Errorf("TextContent() = %q, want %q", got, "Hello world!")
	}

	msg2 := NewUserMessage("plain text message")
	if got := TextContent(msg2); got != "plain text message" {
		t.Errorf("TextContent(NewUserMessage) = %q, want %q", got, "plain text message")
	}
}

func TestNewUserMessage(t *testing.T) {
	msg := NewUserMessage("hello from test")
	if msg.Role != RoleUser {
		t.Errorf("expected RoleUser, got %q", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(msg.Content))
	}
	block := msg.Content[0]
	if block.Type != TypeText || block.Text != "hello from test" {
		t.Errorf("unexpected content block: %+v", block)
	}
}

func TestNewToolResultMessage(t *testing.T) {
	results := []ToolResult{
		{ToolUseID: "tu_1", Content: "success result"},
		{ToolUseID: "tu_2", Content: "something went wrong", IsError: true},
	}
	msg := NewToolResultMessage(results)

	if msg.Role != RoleUser {
		t.Errorf("expected RoleUser, got %q", msg.Role)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(msg.Content))
	}

	b1 := msg.Content[0]
	if b1.Type != TypeToolResult || b1.ToolUseID != "tu_1" || b1.Content != "success result" || b1.IsError {
		t.Errorf("bad first block: %+v", b1)
	}

	b2 := msg.Content[1]
	if b2.Type != TypeToolResult || b2.ToolUseID != "tu_2" || b2.Content != "something went wrong" || !b2.IsError {
		t.Errorf("bad second block: %+v", b2)
	}
}

func TestNewTool(t *testing.T) {
	schema := Object(
		String("a", "first param", Required()),
		Integer("b", ""),
	)

	tool := NewTool("test_tool", "A tool for testing", schema)
	if tool.Name != "test_tool" || tool.Description != "A tool for testing" {
		t.Errorf("metadata mismatch: %+v", tool)
	}
	if len(tool.InputSchema) == 0 {
		t.Error("InputSchema was not populated by json.Marshal")
	}

	// Verify it round-trips reasonably
	var parsed InputSchema
	if err := json.Unmarshal(tool.InputSchema, &parsed); err != nil {
		t.Errorf("schema did not contain valid JSON: %v", err)
	}
	if parsed.Type != "object" || len(parsed.Properties) != 2 {
		t.Errorf("parsed schema invalid: %+v", parsed)
	}
}

func TestSchema_ArrayOfObjects(t *testing.T) {
	// This is the planner-style schema shape that previously forced callers
	// out of the typed builders into hand-written JSON.
	schema := Object(
		String("reasoning", "Why these subtasks cover the question"),
		Array("subtasks", "List of sub-questions",
			ObjectOf(
				String("question", "Sub-question to research", Required()),
				String("rationale", "Why this matters"),
			),
			Required()),
	)

	raw, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("round-trip: %v", err)
	}

	props := got["properties"].(map[string]any)
	subtasks, ok := props["subtasks"].(map[string]any)
	if !ok {
		t.Fatalf("subtasks property missing: %v", got)
	}
	if subtasks["type"] != "array" {
		t.Errorf("subtasks type = %v, want array", subtasks["type"])
	}
	items, ok := subtasks["items"].(map[string]any)
	if !ok {
		t.Fatalf("items missing: %v", subtasks)
	}
	if items["type"] != "object" {
		t.Errorf("items type = %v, want object", items["type"])
	}
	itemProps := items["properties"].(map[string]any)
	if _, ok := itemProps["question"]; !ok {
		t.Errorf("items.properties.question missing")
	}
	itemReq := items["required"].([]any)
	if len(itemReq) != 1 || itemReq[0] != "question" {
		t.Errorf("items.required = %v, want [question]", itemReq)
	}
	topReq := got["required"].([]any)
	if len(topReq) != 1 || topReq[0] != "subtasks" {
		t.Errorf("top required = %v, want [subtasks]", topReq)
	}
}

func TestSchema_ArrayOfScalars(t *testing.T) {
	schema := Object(
		Array("tags", "Free-form tags", SchemaProperty{Type: "string"}),
	)
	raw, _ := json.Marshal(schema)
	var got map[string]any
	_ = json.Unmarshal(raw, &got)
	tags := got["properties"].(map[string]any)["tags"].(map[string]any)
	items := tags["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("items type = %v, want string", items["type"])
	}
}

func TestTypedToolFunc(t *testing.T) {
	type input struct {
		Op  string `json:"op"`
		Val int    `json:"val"`
	}
	fn := TypedToolFunc(func(_ context.Context, in input) (string, error) {
		if in.Op != "echo" {
			return "", fmt.Errorf("unknown op %s", in.Op)
		}
		return fmt.Sprintf("echo:%d", in.Val), nil
	})

	ctx := context.Background()
	t.Run("success", func(t *testing.T) {
		out, err := fn(ctx, json.RawMessage(`{"op":"echo","val":42}`))
		if err != nil {
			t.Fatal(err)
		}
		if out != "echo:42" {
			t.Errorf("got %q", out)
		}
	})
	t.Run("unmarshal fails", func(t *testing.T) {
		_, err := fn(ctx, json.RawMessage(`{"op":"echo","val":"bad"}`))
		if err == nil || !strings.Contains(err.Error(), "unmarshal tool input") {
			t.Errorf("expected unmarshal error, got %v", err)
		}
	})
	t.Run("handler fails", func(t *testing.T) {
		_, err := fn(ctx, json.RawMessage(`{"op":"bad","val":1}`))
		if err == nil || !strings.Contains(err.Error(), "unknown op") {
			t.Errorf("expected handler error, got %v", err)
		}
	})

	t.Run("NewTypedTool", func(t *testing.T) {
		schema := Object(
			String("op", "", Required()),
			Integer("val", "", Required()),
		)
		tool, fn2 := NewTypedTool[input](
			"echo",
			"echo a value",
			schema,
			func(_ context.Context, in input) (string, error) {
				if in.Op != "echo" {
					return "", fmt.Errorf("unknown op %s", in.Op)
				}
				return fmt.Sprintf("echo:%d", in.Val), nil
			},
		)
		if tool.Name != "echo" || tool.Description != "echo a value" {
			t.Errorf("bad tool metadata: %+v", tool)
		}
		if len(tool.InputSchema) == 0 {
			t.Error("InputSchema was not populated")
		}
		// verify fn works like the wrapper
		ctx := context.Background()
		out, err := fn2(ctx, json.RawMessage(`{"op":"echo","val":123}`))
		if err != nil {
			t.Fatal(err)
		}
		if out != "echo:123" {
			t.Errorf("got %q, want echo:123", out)
		}
	})
}

func TestJSONResult(t *testing.T) {
	type out struct {
		Sum int `json:"sum"`
	}
	got, err := JSONResult(out{Sum: 7})
	if err != nil {
		t.Fatal(err)
	}
	if got != `{"sum":7}` {
		t.Errorf("got %q", got)
	}
}

func TestErrorTypes(t *testing.T) {
	t.Run("APIError", func(t *testing.T) {
		err := &APIError{
			StatusCode: 429,
			Type:       "rate_limit_error",
			Message:    "too many requests",
			RetryAfter: 10 * time.Second,
		}
		want := `luft: API 429 (rate_limit_error): too many requests (retry after 10s)`
		if got := err.Error(); got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("ToolError wrapping", func(t *testing.T) {
		cause := errors.New("division by zero")
		err := &ToolError{
			ToolName:  "divide",
			ToolUseID: "tu_42",
			Cause:     cause,
		}
		if !strings.Contains(err.Error(), `tool "divide" (tu_42): division by zero`) {
			t.Errorf("unexpected error string: %v", err)
		}
		if !errors.Is(err, cause) {
			t.Error("ToolError should unwrap to its Cause")
		}
	})

	t.Run("LoopError wrapping", func(t *testing.T) {
		cause := ErrMaxIter
		err := &LoopError{Iter: 10, Cause: cause}
		if !strings.Contains(err.Error(), "loop aborted at iteration 10") {
			t.Errorf("unexpected error string: %v", err)
		}
		if !errors.Is(err, ErrMaxIter) {
			t.Error("LoopError should unwrap to ErrMaxIter")
		}
	})

	t.Run("RetryExhaustedError", func(t *testing.T) {
		cause := errors.New("connection reset")
		err := &RetryExhaustedError{Attempts: 4, Cause: cause}
		if !strings.Contains(err.Error(), "retry exhausted after 4 attempt(s)") {
			t.Errorf("unexpected error string: %v", err)
		}
		if !errors.Is(err, ErrRetryExhausted) {
			t.Error("should match ErrRetryExhausted via Unwrap")
		}
		if !errors.Is(err, cause) {
			t.Error("should unwrap to original cause")
		}
	})
}

func TestRunTools(t *testing.T) {
	ctx := context.Background()
	dispatch := map[string]ToolFunc{
		"echo": func(_ context.Context, input json.RawMessage) (string, error) {
			return string(input), nil
		},
		"fail": func(_ context.Context, _ json.RawMessage) (string, error) {
			return "", fmt.Errorf("tool failure")
		},
	}

	t.Run("successful tools", func(t *testing.T) {
		content := []ContentBlock{
			{
				Type:  TypeToolUse,
				ID:    "tu_1",
				Name:  "echo",
				Input: json.RawMessage(`"hello"`),
			},
		}
		results, err := runTools(ctx, content, dispatch, nil, "", 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 1 || results[0].ToolUseID != "tu_1" || results[0].Content != `"hello"` || results[0].IsError {
			t.Errorf("unexpected result: %+v", results)
		}
	})

	t.Run("tool errors become is_error results", func(t *testing.T) {
		content := []ContentBlock{
			{
				Type:  TypeToolUse,
				ID:    "tu_2",
				Name:  "fail",
				Input: json.RawMessage(`{}`),
			},
		}
		results, err := runTools(ctx, content, dispatch, nil, "", 0)
		if err != nil {
			t.Fatal("runTools should not return error for tool execution failures")
		}
		if len(results) != 1 || !results[0].IsError || !strings.Contains(results[0].Content, "tool failure") {
			t.Errorf("expected IsError result: %+v", results[0])
		}
	})

	t.Run("missing tool returns ToolError", func(t *testing.T) {
		content := []ContentBlock{
			{
				Type: TypeToolUse,
				ID:   "tu_3",
				Name: "missing_tool",
			},
		}
		_, err := runTools(ctx, content, dispatch, nil, "", 0)
		if err == nil {
			t.Fatal("expected error for missing tool")
		}
		var te *ToolError
		if !errors.As(err, &te) {
			t.Fatalf("expected *ToolError, got %T", err)
		}
		if te.ToolName != "missing_tool" || !errors.Is(err, ErrMissingTool) {
			t.Errorf("unexpected ToolError: %+v", te)
		}
	})

	t.Run("multiple tools preserve index order", func(t *testing.T) {
		// "slow" completes after "fast" to verify results are ordered by
		// original position, not completion order.
		ordered := map[string]ToolFunc{
			"slow": func(_ context.Context, _ json.RawMessage) (string, error) {
				time.Sleep(20 * time.Millisecond)
				return "slow-result", nil
			},
			"fast": func(_ context.Context, _ json.RawMessage) (string, error) {
				return "fast-result", nil
			},
		}
		content := []ContentBlock{
			{Type: TypeToolUse, ID: "tu_slow", Name: "slow"},
			{Type: TypeToolUse, ID: "tu_fast", Name: "fast"},
		}
		results, err := runTools(ctx, content, ordered, nil, "", 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 2 {
			t.Fatalf("expected 2 results, got %d", len(results))
		}
		if results[0].ToolUseID != "tu_slow" || results[0].Content != "slow-result" {
			t.Errorf("results[0] wrong: %+v", results[0])
		}
		if results[1].ToolUseID != "tu_fast" || results[1].Content != "fast-result" {
			t.Errorf("results[1] wrong: %+v", results[1])
		}
	})

	t.Run("multiple tools run concurrently", func(t *testing.T) {
		const sleep = 50 * time.Millisecond
		slow2 := map[string]ToolFunc{
			"a": func(_ context.Context, _ json.RawMessage) (string, error) {
				time.Sleep(sleep)
				return "a", nil
			},
			"b": func(_ context.Context, _ json.RawMessage) (string, error) {
				time.Sleep(sleep)
				return "b", nil
			},
		}
		content := []ContentBlock{
			{Type: TypeToolUse, ID: "tu_a", Name: "a"},
			{Type: TypeToolUse, ID: "tu_b", Name: "b"},
		}
		start := time.Now()
		results, err := runTools(ctx, content, slow2, nil, "", 0)
		elapsed := time.Since(start)
		if err != nil {
			t.Fatal(err)
		}
		// Concurrent execution finishes in ~sleep; sequential would take ~2*sleep.
		if elapsed >= 2*sleep {
			t.Errorf("tools appear to have run sequentially (elapsed %v >= 2*%v)", elapsed, sleep)
		}
		if results[0].Content != "a" || results[1].Content != "b" {
			t.Errorf("unexpected results: %+v", results)
		}
	})

	t.Run("missing tool among multiple aborts before goroutines start", func(t *testing.T) {
		var called bool
		mixedDispatch := map[string]ToolFunc{
			"echo": func(_ context.Context, input json.RawMessage) (string, error) {
				called = true
				return string(input), nil
			},
		}
		content := []ContentBlock{
			{Type: TypeToolUse, ID: "tu_1", Name: "echo"},
			{Type: TypeToolUse, ID: "tu_missing", Name: "not_registered"},
			{Type: TypeToolUse, ID: "tu_2", Name: "echo"},
		}
		_, err := runTools(ctx, content, mixedDispatch, nil, "", 0)
		if err == nil {
			t.Fatal("expected error for missing tool")
		}
		var te *ToolError
		if !errors.As(err, &te) || te.ToolName != "not_registered" {
			t.Errorf("expected ToolError for 'not_registered', got %v", err)
		}
		if called {
			t.Error("registered tools should not have been called when validation fails")
		}
	})
}

func TestParallel(t *testing.T) {
	ctx := context.Background()

	steps := []StepFunc[string]{
		func(ctx context.Context) (string, error) {
			return "success1", nil
		},
		func(ctx context.Context) (string, error) {
			return "", errors.New("step failed")
		},
		func(ctx context.Context) (string, error) {
			return "success2", nil
		},
	}

	results := Parallel(ctx, steps...)

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	if results[0].Value != "success1" || results[0].Err != nil {
		t.Errorf("unexpected result[0]: %+v", results[0])
	}
	if results[1].Value != "" || results[1].Err == nil || results[1].Err.Error() != "step failed" {
		t.Errorf("unexpected result[1]: %+v", results[1])
	}
	if results[2].Value != "success2" || results[2].Err != nil {
		t.Errorf("unexpected result[2]: %+v", results[2])
	}
}

func TestRetryConfig_applyDefaults(t *testing.T) {
	tests := []struct {
		name  string
		input RetryConfig
		want  RetryConfig
	}{
		{
			name:  "all defaults",
			input: RetryConfig{},
			want: RetryConfig{
				MaxRetries:  defaultMaxRetries,
				InitialWait: defaultInitialWait,
				MaxWait:     defaultMaxWait,
				Disabled:    false,
			},
		},
		{
			name: "custom values preserved",
			input: RetryConfig{
				MaxRetries:  10,
				InitialWait: 500 * time.Millisecond,
				MaxWait:     60 * time.Second,
				Disabled:    true,
			},
			want: RetryConfig{
				MaxRetries:  10,
				InitialWait: 500 * time.Millisecond,
				MaxWait:     60 * time.Second,
				Disabled:    true,
			},
		},
		{
			name: "partial override",
			input: RetryConfig{
				MaxRetries: 2,
				Disabled:   true,
			},
			want: RetryConfig{
				MaxRetries:  2,
				InitialWait: defaultInitialWait,
				MaxWait:     defaultMaxWait,
				Disabled:    true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.input.applyDefaults()
			if got.MaxRetries != tt.want.MaxRetries ||
				got.InitialWait != tt.want.InitialWait ||
				got.MaxWait != tt.want.MaxWait ||
				got.Disabled != tt.want.Disabled {
				t.Errorf("applyDefaults() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestIsTemporary(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil error", nil, false},
		{"temporary error", temporaryError{errors.New("temp")}, true},
		{"wrapped temporary", fmt.Errorf("wrapped: %w", temporaryError{errors.New("temp")}), true},
		{"ordinary error", errors.New("ordinary"), false},
		{"non-temporary interface", nonTemporary{errors.New("no")}, false},
		{"context canceled", context.Canceled, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isTemporary(tt.err); got != tt.want {
				t.Errorf("isTemporary(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestClientAsk(t *testing.T) {
	c, err := New(Config{
		Provider: newTestProvider(),
		Model:    "test-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	history := []Message{NewUserMessage("hello")}
	msg, usage, err := c.Ask(context.Background(), "be helpful", history)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg.Role != RoleAssistant || TextContent(msg) != "Hello from mock" {
		t.Errorf("unexpected message: role=%s content=%s", msg.Role, TextContent(msg))
	}
	if usage.InputTokens == 0 && usage.OutputTokens == 0 {
		// testProvider populates non-zero usage; if both are zero, the
		// passthrough is broken.
		t.Errorf("expected non-zero usage from provider, got %+v", usage)
	}
}

func TestClientAskStream(t *testing.T) {
	p := newTestProvider()
	p.Deltas = []ContentBlock{{Type: TypeText, Text: "Hello from mock"}}
	c, err := New(Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	var received []ContentBlock
	onToken := func(b ContentBlock) {
		received = append(received, b)
	}
	history := []Message{NewUserMessage("hello")}
	msg, _, err := c.AskStream(context.Background(), "", history, onToken)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(received) == 0 || received[0].Text != "Hello from mock" {
		t.Errorf("onToken not called with expected delta: %+v", received)
	}
	if TextContent(msg) != "Hello from mock" {
		t.Errorf("final message content mismatch: %s", TextContent(msg))
	}
	if p.StreamCount != 1 {
		t.Errorf("expected 1 Stream call, got %d", p.StreamCount)
	}
}

func TestClientLoop(t *testing.T) {
	p := newTestProvider()
	c, err := New(Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	history := []Message{NewUserMessage("start conversation")}
	result, err := c.Loop(context.Background(), "system", history, Toolset{}, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Messages) != 2 || result.Messages[1].Role != RoleAssistant {
		t.Errorf("expected 1 new assistant message, got %d messages", len(result.Messages))
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 {
		t.Errorf("unexpected usage: %+v", result.Usage)
	}
	if p.CallCount != 1 {
		t.Errorf("expected 1 Call, got %d", p.CallCount)
	}
}

func TestClientLoopStream(t *testing.T) {
	p := newTestProvider()
	p.Deltas = []ContentBlock{{Type: TypeText, Text: "Hello from mock"}}
	c, err := New(Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	var tokenCalls int
	onToken := func(ContentBlock) { tokenCalls++ }
	var toolCalls int
	onToolResult := func([]ToolResult) { toolCalls++ }
	history := []Message{NewUserMessage("start")}
	result, err := c.LoopStream(context.Background(), "system", history, Toolset{}, 0, onToken, onToolResult)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tokenCalls == 0 {
		t.Error("onToken callback was not invoked")
	}
	if toolCalls != 0 {
		t.Error("onToolResult should not be called on end_turn path")
	}
	if len(result.Messages) != 2 {
		t.Errorf("expected history + assistant msg, got %d", len(result.Messages))
	}
	if p.StreamCount != 1 {
		t.Errorf("expected 1 Stream call, got %d", p.StreamCount)
	}
}

func TestCallWithRetry(t *testing.T) {
	rand.Seed(1) // for deterministic jitter in backoff tests
	ctx := context.Background()

	tests := []struct {
		name         string
		retryCfg     RetryConfig
		fn           func() (ProviderResponse, error)
		wantErr      error // checked with errors.Is; nil means no error expected
		wantSomeErr  bool  // true when an error is expected but no sentinel to check
		wantAttempts int
		wantResp     bool
	}{
		{
			name:     "success on first try",
			retryCfg: RetryConfig{},
			fn: func() (ProviderResponse, error) {
				return testResponse, nil
			},
			wantErr:      nil,
			wantAttempts: 1,
			wantResp:     true,
		},
		{
			name:     "non-retryable 400 returns immediately",
			retryCfg: RetryConfig{},
			fn: func() (ProviderResponse, error) {
				return ProviderResponse{}, &APIError{StatusCode: 400, Message: "bad request"}
			},
			wantSomeErr:  true,
			wantAttempts: 1,
		},
		{
			name:     "retryable 429 with exhaustion",
			retryCfg: RetryConfig{MaxRetries: 1},
			fn: func() (ProviderResponse, error) {
				return ProviderResponse{}, &APIError{StatusCode: 429, Message: "rate limited"}
			},
			wantErr:      ErrRetryExhausted,
			wantAttempts: 2,
		},
		{
			name:     "context canceled aborts immediately",
			retryCfg: RetryConfig{},
			fn: func() (ProviderResponse, error) {
				return ProviderResponse{}, context.Canceled
			},
			wantErr: context.Canceled,
		},
		{
			name:     "ErrMissingTool never retries",
			retryCfg: RetryConfig{},
			fn: func() (ProviderResponse, error) {
				return ProviderResponse{}, &ToolError{ToolName: "foo", Cause: ErrMissingTool}
			},
			wantErr: ErrMissingTool,
		},
		{
			name:     "disabled retry",
			retryCfg: RetryConfig{Disabled: true},
			fn: func() (ProviderResponse, error) {
				return ProviderResponse{}, &APIError{StatusCode: 503}
			},
			wantSomeErr:  true,
			wantAttempts: 1,
		},
		{
			name:     "retryable 529 overloaded with exhaustion",
			retryCfg: RetryConfig{MaxRetries: 1, InitialWait: time.Millisecond},
			fn: func() (ProviderResponse, error) {
				return ProviderResponse{}, &APIError{StatusCode: 529, Message: "overloaded"}
			},
			wantErr:      ErrRetryExhausted,
			wantAttempts: 2,
		},
		{
			name:     "retryable 500 with exhaustion",
			retryCfg: RetryConfig{MaxRetries: 1, InitialWait: time.Millisecond},
			fn: func() (ProviderResponse, error) {
				return ProviderResponse{}, &APIError{StatusCode: 500, Message: "server error"}
			},
			wantErr:      ErrRetryExhausted,
			wantAttempts: 2,
		},
		{
			name:     "non-retryable 422 returns immediately",
			retryCfg: RetryConfig{MaxRetries: 3, InitialWait: time.Millisecond},
			fn: func() (ProviderResponse, error) {
				return ProviderResponse{}, &APIError{StatusCode: 422, Message: "unprocessable"}
			},
			wantSomeErr:  true,
			wantAttempts: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var attempts int
			wrappedFn := func() (ProviderResponse, error) {
				attempts++
				return tt.fn()
			}
			resp, err := callWithRetry(ctx, tt.retryCfg, wrappedFn)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("got err %v, want %v", err, tt.wantErr)
				}
				var exhausted *RetryExhaustedError
				if errors.As(err, &exhausted) {
					if exhausted.Attempts != tt.wantAttempts {
						t.Errorf("expected %d attempts, got %d", tt.wantAttempts, exhausted.Attempts)
					}
				}
			} else if tt.wantSomeErr {
				if err == nil {
					t.Error("expected an error, got nil")
				}
			} else if err != nil {
				t.Errorf("unexpected err: %v", err)
			}
			if tt.wantAttempts > 0 && attempts != tt.wantAttempts {
				t.Errorf("attempts = %d, want %d", attempts, tt.wantAttempts)
			}
			if tt.wantResp && len(resp.Content) == 0 {
				t.Error("expected valid response")
			}
		})
	}
}

func TestLoop(t *testing.T) {
	toolUseResp := ProviderResponse{
		Content:    []ContentBlock{{Type: TypeToolUse, ID: "tu1", Name: "echo"}},
		StopReason: "tool_use",
		Usage:      Usage{InputTokens: 10, OutputTokens: 5},
	}
	echoDispatch := map[string]ToolFunc{
		"echo": func(context.Context, json.RawMessage) (string, error) { return "result", nil },
	}

	tests := []struct {
		name            string
		responses       []ProviderResponse // scripted sequence; last entry repeated if exhausted
		tools           []Tool
		dispatch        map[string]ToolFunc
		maxIter         int
		wantErr         error
		wantMsgCount    int
		wantInputTokens int
	}{
		{
			name:            "end_turn success",
			responses:       []ProviderResponse{testResponse},
			wantMsgCount:    2,
			wantInputTokens: 10,
		},
		{
			name: "tool_use with successful dispatch",
			// First call: tool_use; second call: end_turn.
			responses:       []ProviderResponse{toolUseResp, testResponse},
			tools:           []Tool{{Name: "echo"}},
			dispatch:        echoDispatch,
			wantMsgCount:    4, // history + assistant(tool_use) + tool_result + assistant(end_turn)
			wantInputTokens: 20,
		},
		{
			name:         "missing tool aborts with ToolError",
			responses:    []ProviderResponse{toolUseResp},
			tools:        []Tool{{Name: "echo"}},
			dispatch:     map[string]ToolFunc{}, // echo missing → ErrMissingTool
			wantErr:      ErrMissingTool,
			wantMsgCount: 2,
		},
		{
			name: "maxIter exhaustion",
			// Provider always returns tool_use; maxIter=1 forces exit after 1 iteration.
			responses:    []ProviderResponse{toolUseResp},
			tools:        []Tool{{Name: "echo"}},
			dispatch:     echoDispatch,
			maxIter:      1,
			wantErr:      ErrMaxIter,
			wantMsgCount: 3, // history + assistant(tool_use) + tool_result
		},
		{
			name: "max_tokens error",
			responses: []ProviderResponse{{
				Content:    []ContentBlock{{Type: TypeText, Text: "truncated"}},
				StopReason: "max_tokens",
				Usage:      Usage{InputTokens: 10, OutputTokens: 5},
			}},
			wantErr:         nil, // wrapped in LoopError — checked via wantSomeErr below
			wantMsgCount:    2,
			wantInputTokens: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestProvider()
			// Load the scripted sequence; last entry is the fallback Resp.
			if len(tt.responses) > 0 {
				p.Resp = tt.responses[len(tt.responses)-1]
				if len(tt.responses) > 1 {
					p.Responses = tt.responses[:len(tt.responses)-1]
				}
			}
			c, _ := New(Config{Provider: p, Model: "test-model"})
			history := []Message{NewUserMessage("hi")}
			ts := Toolset{}
			for _, tool := range tt.tools {
				ts.Bindings = append(ts.Bindings, ToolBinding{Tool: tool, Func: tt.dispatch[tool.Name]})
			}
			result, err := c.Loop(context.Background(), "sys", history, ts, tt.maxIter)
			if tt.wantErr != nil {
				found := errors.Is(err, tt.wantErr)
				if !found {
					var le *LoopError
					if errors.As(err, &le) {
						found = errors.Is(le.Cause, tt.wantErr)
					}
				}
				if !found {
					t.Errorf("got err %v, want containing %v", err, tt.wantErr)
				}
			} else if tt.name == "max_tokens error" {
				if err == nil {
					t.Error("expected error for max_tokens, got nil")
				}
			} else if err != nil {
				t.Errorf("unexpected err: %v", err)
			}
			if len(result.Messages) != tt.wantMsgCount {
				t.Errorf("got %d messages, want %d", len(result.Messages), tt.wantMsgCount)
			}
			if tt.wantInputTokens > 0 && result.Usage.InputTokens != tt.wantInputTokens {
				t.Errorf("InputTokens = %d, want %d", result.Usage.InputTokens, tt.wantInputTokens)
			}
		})
	}
}

func TestLoopStream(t *testing.T) {
	// Similar table-driven structure as TestLoop but with callback spies
	p := newTestProvider()
	p.Resp.StopReason = "end_turn"
	p.Deltas = []ContentBlock{{Type: TypeText, Text: "delta1"}, {Type: TypeText, Text: "delta2"}}
	c, _ := New(Config{Provider: p, Model: "test-model"})
	var tokenCount, toolCount int
	onToken := func(ContentBlock) { tokenCount++ }
	onTool := func([]ToolResult) { toolCount++ }
	history := []Message{NewUserMessage("hi")}
	result, err := c.LoopStream(context.Background(), "sys", history, Toolset{}, 0, onToken, onTool)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if tokenCount != 2 {
		t.Errorf("expected 2 onToken calls, got %d", tokenCount)
	}
	if toolCount != 0 {
		t.Error("no tool calls expected")
	}
	if len(result.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(result.Messages))
	}
}

func TestAskStreamNilCallback(t *testing.T) {
	p := newTestProvider()
	p.Deltas = []ContentBlock{{Type: TypeText, Text: "hello"}}
	c, _ := New(Config{Provider: p, Model: "test-model"})
	// nil onToken must not panic
	msg, _, err := c.AskStream(context.Background(), "", []Message{NewUserMessage("hi")}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if TextContent(msg) != "Hello from mock" {
		t.Errorf("unexpected message content: %s", TextContent(msg))
	}
}

func TestLoopStreamNilCallbacks(t *testing.T) {
	p := newTestProvider()
	p.Deltas = []ContentBlock{{Type: TypeText, Text: "hello"}}
	c, _ := New(Config{Provider: p, Model: "test-model"})
	// Both nil — must not panic even when provider fires deltas.
	_, err := c.LoopStream(context.Background(), "", []Message{NewUserMessage("hi")}, Toolset{}, 0, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRetryConfigOnRetry(t *testing.T) {
	ctx := context.Background()
	var retryCalls []int
	cfg := RetryConfig{
		MaxRetries:  2,
		InitialWait: time.Millisecond, // fast for tests
		OnRetry:     func(attempt int, _ time.Duration) { retryCalls = append(retryCalls, attempt) },
	}

	calls := 0
	_, err := callWithRetry(ctx, cfg, func() (ProviderResponse, error) {
		calls++
		if calls <= 2 {
			return ProviderResponse{}, &APIError{StatusCode: 429, Message: "rate limited"}
		}
		return testResponse, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Two failures → two retries → OnRetry called twice, with attempt numbers 1 and 2.
	if len(retryCalls) != 2 {
		t.Fatalf("expected 2 OnRetry calls, got %d", len(retryCalls))
	}
	if retryCalls[0] != 1 || retryCalls[1] != 2 {
		t.Errorf("unexpected attempt numbers: %v", retryCalls)
	}
}

func TestRetryConfigOnRetryNotCalledOnSuccess(t *testing.T) {
	ctx := context.Background()
	var called bool
	cfg := RetryConfig{
		OnRetry: func(int, time.Duration) { called = true },
	}
	_, err := callWithRetry(ctx, cfg, func() (ProviderResponse, error) {
		return testResponse, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if called {
		t.Error("OnRetry should not be called when the first attempt succeeds")
	}
}

func TestRetryConfigOnRetryNotCalledWhenDisabled(t *testing.T) {
	ctx := context.Background()
	var called bool
	cfg := RetryConfig{
		Disabled: true,
		OnRetry:  func(int, time.Duration) { called = true },
	}
	_, _ = callWithRetry(ctx, cfg, func() (ProviderResponse, error) {
		return ProviderResponse{}, &APIError{StatusCode: 503}
	})
	if called {
		t.Error("OnRetry should not be called when retry is disabled")
	}
}
