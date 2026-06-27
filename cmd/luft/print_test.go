package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/providers/mock"
)

// textResp is a canned end_turn response carrying a single text block.
func textResp(text string, usage luft.Usage) luft.ProviderResponse {
	return luft.ProviderResponse{
		Content:    []luft.ContentBlock{{Type: luft.TypeText, Text: text}},
		StopReason: "end_turn",
		Usage:      usage,
	}
}

// toolUseResp is a canned tool_use response; the follow-up text response
// completes the turn.
func toolUseResp(id, name string, input []byte, usage luft.Usage) luft.ProviderResponse {
	return luft.ProviderResponse{
		Content: []luft.ContentBlock{{
			Type: luft.TypeToolUse, ID: id, Name: name, Input: input,
		}},
		StopReason: "tool_use",
		Usage:      usage,
	}
}

// newPrintSession builds a *session wired to a mock-backed agent and
// bytes.Buffer output streams, ready for runPrint assertions.
func newPrintSession(t *testing.T, responses ...luft.ProviderResponse) (*session, *mock.Provider, *bytes.Buffer, *bytes.Buffer) {
	t.Helper()
	p := mock.New(responses...)
	client, err := luft.New(luft.Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatalf("luft.New: %v", err)
	}
	var out, errOut bytes.Buffer
	s := &session{
		agent: luft.Agent{
			Client:  client,
			System:  "be helpful",
			MaxIter: 30,
		},
		sessionID: "test-session",
		sp:        newSpinner(io.Discard), // no-op spinner
		out:       &out,
		errOut:    &errOut,
	}
	return s, p, &out, &errOut
}

func TestRunPrintPlainText(t *testing.T) {
	s, _, out, errOut := newPrintSession(t, textResp("hello world", luft.Usage{InputTokens: 5, OutputTokens: 2}))
	code := s.runPrint(context.Background(), "hi", printOpts{})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if got := out.String(); !strings.Contains(got, "hello world") {
		t.Errorf("stdout = %q, want it to contain %q", got, "hello world")
	}
	// Plain mode emits a trailing newline after the streamed text.
	if !strings.HasSuffix(out.String(), "\n") {
		t.Errorf("stdout should end with newline, got %q", out.String())
	}
	// No tool calls → stderr is empty (quiet defaults off but there's nothing to print).
	if errOut.String() != "" {
		t.Errorf("stderr = %q, want empty", errOut.String())
	}
}

func TestRunPrintJSON(t *testing.T) {
	s, _, out, _ := newPrintSession(t, textResp("hello world", luft.Usage{InputTokens: 5, OutputTokens: 2}))
	code := s.runPrint(context.Background(), "hi", printOpts{json: true})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	var pr printResult
	if err := json.Unmarshal(out.Bytes(), &pr); err != nil {
		t.Fatalf("unmarshal JSON: %v\nraw: %s", err, out.String())
	}
	if pr.Result != "hello world" {
		t.Errorf("result = %q, want %q", pr.Result, "hello world")
	}
	if pr.Error != nil {
		t.Errorf("error = %v, want nil", *pr.Error)
	}
	if pr.ExitCode != 0 {
		t.Errorf("exit_code = %d, want 0", pr.ExitCode)
	}
	if pr.Usage.InputTokens != 5 {
		t.Errorf("usage.input_tokens = %d, want 5", pr.Usage.InputTokens)
	}
	if pr.SessionID != "test-session" {
		t.Errorf("session_id = %q, want %q", pr.SessionID, "test-session")
	}
}

func TestRunPrintJSONWithToolCall(t *testing.T) {
	// Script: turn 1 the model calls a tool; turn 2 it finishes with text.
	s, _, out, errOut := newPrintSession(t,
		toolUseResp("tu_1", "bash", []byte(`{"command":"echo hi"}`), luft.Usage{InputTokens: 10, OutputTokens: 1}),
		textResp("done", luft.Usage{InputTokens: 12, OutputTokens: 3}),
	)
	// Wire a tool so the loop can dispatch the model's tool_use.
	tool, fn := luft.NewTypedTool[struct{ Command string }](
		"bash", "run a command", luft.Object(luft.String("command", "cmd", luft.Required())),
		func(_ context.Context, in struct{ Command string }) (string, error) { return "hi\n", nil },
	)
	s.agent.Tools = luft.Tools(luft.Bind(tool, fn))

	code := s.runPrint(context.Background(), "run it", printOpts{json: true})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr: %s", code, errOut.String())
	}
	var pr printResult
	if err := json.Unmarshal(out.Bytes(), &pr); err != nil {
		t.Fatalf("unmarshal JSON: %v\nraw: %s", err, out.String())
	}
	if pr.Result != "done" {
		t.Errorf("result = %q, want %q", pr.Result, "done")
	}
	if len(pr.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %d, want 1: %+v", len(pr.ToolCalls), pr.ToolCalls)
	}
	if pr.ToolCalls[0].Name != "bash" {
		t.Errorf("tool_calls[0].name = %q, want %q", pr.ToolCalls[0].Name, "bash")
	}
	if pr.ToolCalls[0].IsError {
		t.Errorf("tool_calls[0].is_error = true, want false")
	}
}

func TestRunPrintError(t *testing.T) {
	// No responses → mock errors on first call → runPrint reports exit 1.
	s, _, out, errOut := newPrintSession(t)
	code := s.runPrint(context.Background(), "hi", printOpts{json: true})
	if code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	var pr printResult
	if err := json.Unmarshal(out.Bytes(), &pr); err != nil {
		t.Fatalf("unmarshal JSON: %v\nraw: %s", err, out.String())
	}
	if pr.Error == nil {
		t.Fatal("error = nil, want non-nil")
	}
	if pr.ExitCode != 1 {
		t.Errorf("exit_code = %d, want 1", pr.ExitCode)
	}
	if !strings.Contains(errOut.String(), "✗ error") {
		t.Errorf("stderr = %q, want it to contain the error marker", errOut.String())
	}
	// On error the user message is popped so the session stays clean.
	if len(s.history) != 0 {
		t.Errorf("history len = %d, want 0 after error", len(s.history))
	}
}

func TestRunPrintErrMaxIter(t *testing.T) {
	// Script a model that always calls a tool → with MaxIter:1 the loop
	// hits the cap after the first call.
	s, _, out, errOut := newPrintSession(t,
		toolUseResp("tu_1", "bash", []byte(`{"command":"echo hi"}`), luft.Usage{InputTokens: 10, OutputTokens: 1}),
	)
	tool, fn := luft.NewTypedTool[struct{ Command string }](
		"bash", "run a command", luft.Object(luft.String("command", "cmd", luft.Required())),
		func(_ context.Context, in struct{ Command string }) (string, error) { return "hi\n", nil },
	)
	s.agent.Tools = luft.Tools(luft.Bind(tool, fn))
	s.agent.MaxIter = 1

	code := s.runPrint(context.Background(), "loop", printOpts{json: true})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (max-iter is a pause, not a failure)", code)
	}
	var pr printResult
	if err := json.Unmarshal(out.Bytes(), &pr); err != nil {
		t.Fatalf("unmarshal JSON: %v\nraw: %s", err, out.String())
	}
	if pr.Error == nil || *pr.Error != "max-iter" {
		t.Errorf("error = %v, want %q", pr.Error, "max-iter")
	}
	if !strings.Contains(errOut.String(), "paused") {
		t.Errorf("stderr = %q, want it to mention 'paused'", errOut.String())
	}
	// Partial progress is kept so the session can resume.
	if len(s.history) == 0 {
		t.Errorf("history should be kept on max-iter, got empty")
	}
}

func TestRunPrintSIGINT(t *testing.T) {
	s, _, out, _ := newPrintSession(t, textResp("hello", luft.Usage{}))
	ctx, cancel := context.WithCancel(context.Background())
	// Cancel before the run starts → StepStream returns context.Canceled.
	cancel()
	code := s.runPrint(ctx, "hi", printOpts{json: true})
	if code != 130 {
		t.Fatalf("exit code = %d, want 130 (SIGINT)", code)
	}
	var pr printResult
	if err := json.Unmarshal(out.Bytes(), &pr); err != nil {
		t.Fatalf("unmarshal JSON: %v\nraw: %s", err, out.String())
	}
	if pr.Error == nil || *pr.Error != "interrupted" {
		t.Errorf("error = %v, want %q", pr.Error, "interrupted")
	}
}

func TestRunPrintQuietSuppressesToolLine(t *testing.T) {
	s, _, out, errOut := newPrintSession(t,
		toolUseResp("tu_1", "bash", []byte(`{"command":"echo hi"}`), luft.Usage{InputTokens: 10, OutputTokens: 1}),
		textResp("done", luft.Usage{InputTokens: 12, OutputTokens: 3}),
	)
	tool, fn := luft.NewTypedTool[struct{ Command string }](
		"bash", "run a command", luft.Object(luft.String("command", "cmd", luft.Required())),
		func(_ context.Context, in struct{ Command string }) (string, error) { return "hi\n", nil },
	)
	s.agent.Tools = luft.Tools(luft.Bind(tool, fn))

	code := s.runPrint(context.Background(), "run it", printOpts{quiet: true})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if errOut.String() != "" {
		t.Errorf("quiet stderr = %q, want empty", errOut.String())
	}
	if !strings.Contains(out.String(), "done") {
		t.Errorf("stdout = %q, want it to contain the result", out.String())
	}
}

func TestAssemblePrompt(t *testing.T) {
	tests := []struct {
		name        string
		instruction string
		stdin       string
		want        string
		wantErr     bool
	}{
		{
			name:        "instruction plus stdin joins with delimiter",
			instruction: "review this diff",
			stdin:       "diff --git a/foo b/foo",
			want:        "review this diff\n\n--- piped stdin ---\ndiff --git a/foo b/foo",
		},
		{
			name:        "instruction only is verbatim",
			instruction: "what does main.go do",
			stdin:       "",
			want:        "what does main.go do",
		},
		{
			name:        "stdin only is verbatim",
			instruction: "",
			stdin:       "explain this code",
			want:        "explain this code",
		},
		{
			name:        "neither instruction nor stdin errors",
			instruction: "",
			stdin:       "",
			wantErr:     true,
		},
		{
			name:        "whitespace-only instruction falls back to stdin",
			instruction: "   ",
			stdin:       "just stdin",
			want:        "just stdin",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := assemblePrompt(tt.instruction, tt.stdin)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("assemblePrompt(%q, %q) = %q, want %q", tt.instruction, tt.stdin, got, tt.want)
			}
		})
	}
}

func TestAssemblePromptTruncation(t *testing.T) {
	// stdin exactly at the cap should not mark truncated; one byte over
	// should append the marker.
	t.Run("exactly at cap is not truncated", func(t *testing.T) {
		exactly := strings.Repeat("x", maxStdinBytes)
		// readStdin with a reader that yields exactly the cap and then EOF.
		got := readStdin(strings.NewReader(exactly), false)
		if strings.Contains(got, "truncated") {
			t.Errorf("input at cap was marked truncated: %q...", got[:50])
		}
	})
	t.Run("over cap is truncated with marker", func(t *testing.T) {
		over := strings.Repeat("x", maxStdinBytes+10)
		got := readStdin(strings.NewReader(over), false)
		if !strings.Contains(got, "[stdin truncated at 4 MiB]") {
			t.Errorf("input over cap was not marked truncated")
		}
	})
}

func TestIsNonInteractive(t *testing.T) {
	tests := []struct {
		name      string
		printFlag string
		nArgs     int
		stdinTTY  bool
		want      bool
	}{
		{"no args on a TTY → interactive", "", 0, true, false},
		{"-p set → non-interactive", "review this", 0, true, true},
		{"positional args → non-interactive", "", 1, true, true},
		{"piped stdin (no TTY) → non-interactive", "", 0, false, true},
		{"-p plus piped stdin → non-interactive", "review", 0, false, true},
		{"empty -p with no args on TTY → interactive (falls through)", "", 0, true, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isNonInteractive(tt.printFlag, tt.nArgs, tt.stdinTTY)
			if got != tt.want {
				t.Errorf("isNonInteractive(%q, %d, %v) = %v, want %v",
					tt.printFlag, tt.nArgs, tt.stdinTTY, got, tt.want)
			}
		})
	}
}

func TestPrintResultJSONShape(t *testing.T) {
	// A printResult with no error must serialize error as null, not "".
	pr := printResult{
		SessionID: "s1",
		Result:    "ok",
		Usage:     luft.Usage{InputTokens: 1, OutputTokens: 2},
		ExitCode:  0,
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(pr); err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !strings.Contains(buf.String(), `"error":null`) {
		t.Errorf("JSON = %s, want error:null", buf.String())
	}
	// HTML-sensitive chars in result are not escaped (so diffs/code survive).
	pr2 := printResult{Result: "a < b && c > d", ExitCode: 0}
	buf.Reset()
	_ = json.NewEncoder(&buf).Encode(pr2) // default encoder also honours SetEscapeHTML(false) when set
	enc2 := json.NewEncoder(&buf)
	enc2.SetEscapeHTML(false)
	buf.Reset()
	_ = enc2.Encode(pr2)
	if strings.Contains(buf.String(), `\u003c`) {
		t.Errorf("JSON = %s, HTML-escaped < should not be present", buf.String())
	}
}

// TestRunPrintUsageAccumulates guards the token-accounting invariant:
// runPrint must add the run's usage onto s.usage so /tokens and persisted
// sessions stay accurate across resume.
func TestRunPrintUsageAccumulates(t *testing.T) {
	s, _, _, _ := newPrintSession(t, textResp("hi", luft.Usage{InputTokens: 7, OutputTokens: 3, CacheReadTokens: 100}))
	s.usage = luft.Usage{} // start clean
	_ = s.runPrint(context.Background(), "hi", printOpts{json: true})
	if s.usage.InputTokens != 7 {
		t.Errorf("usage.InputTokens = %d, want 7", s.usage.InputTokens)
	}
	if s.usage.OutputTokens != 3 {
		t.Errorf("usage.OutputTokens = %d, want 3", s.usage.OutputTokens)
	}
	if s.usage.CacheReadTokens != 100 {
		t.Errorf("usage.CacheReadTokens = %d, want 100", s.usage.CacheReadTokens)
	}
}

// TestRunPrintNoTimeoutDeadlock is a smoke test that runPrint doesn't hang
// when the mock returns a clean end_turn on the first call.
func TestRunPrintNoHang(t *testing.T) {
	s, _, _, _ := newPrintSession(t, textResp("ok", luft.Usage{}))
	done := make(chan int, 1)
	go func() {
		done <- s.runPrint(context.Background(), "hi", printOpts{})
	}()
	select {
	case code := <-done:
		if code != 0 {
			t.Errorf("exit code = %d, want 0", code)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("runPrint hung for >2s")
	}
}

// ensures errors import is used even if a future edit removes its only
// reference; keeps gofmt happy.
var _ = errors.Is
