package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/providers/mock"
)

// TestChat_SSEStreamsAndPersistsHistory drives a two-turn conversation
// through the handler with a scripted mock provider and asserts that (a) the
// SSE response body contains the streamed assistant text, and (b) a second
// request to the same session sees the first turn's history, verified via
// the mock's captured Requests().
func TestChat_SSEStreamsAndPersistsHistory(t *testing.T) {
	p := mock.New(
		luft.ProviderResponse{
			Content:    []luft.ContentBlock{{Type: luft.TypeText, Text: "hello there"}},
			StopReason: "end_turn",
		},
		luft.ProviderResponse{
			Content:    []luft.ContentBlock{{Type: luft.TypeText, Text: "general kenobi"}},
			StopReason: "end_turn",
		},
	)
	client, err := luft.New(luft.Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatalf("luft.New: %v", err)
	}
	srv := NewServer(luft.Agent{Client: client, System: "be helpful"})

	// Turn 1.
	rec1 := httptest.NewRecorder()
	req1 := httptest.NewRequest(http.MethodPost, "/chat",
		strings.NewReader(`{"session":"s1","message":"hi"}`))
	srv.HandleChat(rec1, req1)

	resp1 := rec1.Result()
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("turn 1 status = %d, want 200", resp1.StatusCode)
	}
	if got := resp1.Header.Get("content-type"); got != "text/event-stream" {
		t.Fatalf("turn 1 content-type = %q, want text/event-stream", got)
	}
	// (a) The SSE body contains the streamed assistant text.
	if !strings.Contains(rec1.Body.String(), "data: hello there\n\n") {
		t.Fatalf("turn 1 body = %q, want it to contain the SSE data line", rec1.Body.String())
	}

	// Turn 2 (same session) — context from turn 1 must be present.
	rec2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodPost, "/chat",
		strings.NewReader(`{"session":"s1","message":"bye"}`))
	srv.HandleChat(rec2, req2)
	if !strings.Contains(rec2.Body.String(), "data: general kenobi\n\n") {
		t.Fatalf("turn 2 body = %q, want the streamed second reply", rec2.Body.String())
	}

	// (b) The provider's second request must carry turn 1's history:
	//     [user "hi", assistant "hello there", user "bye"].
	reqs := p.Requests()
	if len(reqs) != 2 {
		t.Fatalf("provider captured %d requests, want 2", len(reqs))
	}
	msgs := reqs[1].Messages
	if len(msgs) < 3 {
		t.Fatalf("turn-2 provider request had %d messages, want >= 3", len(msgs))
	}
	if msgs[0].Role != luft.RoleUser || luft.TextContent(msgs[0]) != "hi" {
		t.Errorf("msgs[0] = %+v, want user \"hi\"", msgs[0])
	}
	if msgs[1].Role != luft.RoleAssistant || luft.TextContent(msgs[1]) != "hello there" {
		t.Errorf("msgs[1] = %+v, want assistant \"hello there\"", msgs[1])
	}
	last := msgs[len(msgs)-1]
	if last.Role != luft.RoleUser || luft.TextContent(last) != "bye" {
		t.Errorf("last msg = %+v, want user \"bye\"", last)
	}
}

// TestChat_MultilineTextIsSSEEncoded guards the SSE framing: a text delta
// containing a newline (routine in LLM output) must become two "data:" lines,
// not "data: a\nb", which an SSE client would truncate at the first line.
func TestChat_MultilineTextIsSSEEncoded(t *testing.T) {
	p := mock.New(luft.ProviderResponse{
		Content:    []luft.ContentBlock{{Type: luft.TypeText, Text: "line one\nline two"}},
		StopReason: "end_turn",
	})
	client, err := luft.New(luft.Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatalf("luft.New: %v", err)
	}
	srv := NewServer(luft.Agent{Client: client})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/chat",
		strings.NewReader(`{"session":"s1","message":"hi"}`))
	srv.HandleChat(rec, req)

	if !strings.Contains(rec.Body.String(), "data: line one\ndata: line two\n\n") {
		t.Fatalf("multi-line text not SSE-encoded per spec; body = %q", rec.Body.String())
	}
}

func TestChat_BadJSON(t *testing.T) {
	p := mock.New()
	client, err := luft.New(luft.Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatalf("luft.New: %v", err)
	}
	srv := NewServer(luft.Agent{Client: client})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/chat",
		strings.NewReader(`{"session":"s1","message":`)) // malformed
	srv.HandleChat(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "invalid JSON") {
		t.Fatalf("body = %q, want \"invalid JSON\"", rec.Body.String())
	}
	if len(p.Requests()) != 0 {
		t.Errorf("mock was called on a rejected request: %d calls", len(p.Requests()))
	}
}

// TestChat_AgentErrorBeforeStream verifies the lazy-header design: with the
// mock's script exhausted immediately, StepStream errors before any token is
// flushed, so the handler can still return a clean 500.
func TestChat_AgentErrorBeforeStream(t *testing.T) {
	p := mock.New() // no responses → first Call errors
	client, err := luft.New(luft.Config{Provider: p, Model: "test-model"})
	if err != nil {
		t.Fatalf("luft.New: %v", err)
	}
	srv := NewServer(luft.Agent{Client: client})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/chat",
		strings.NewReader(`{"session":"s1","message":"hi"}`))
	srv.HandleChat(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "error") {
		t.Fatalf("body = %q, want an error message", rec.Body.String())
	}
}
