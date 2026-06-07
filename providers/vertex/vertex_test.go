package vertex

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/lukemuz/luft"
)

func testProvider(t *testing.T, baseURL string) *Provider {
	t.Helper()
	p, err := NewProvider(Config{
		Project:     "test-proj",
		Location:    "us-central1",
		BaseURL:     baseURL,
		TokenSource: staticTokenSource{tok: "TEST-TOKEN"},
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	return p
}

func TestMessagesToGemini_AssistantRoleAndContent(t *testing.T) {
	msgs := []luft.Message{
		luft.NewUserMessage("hello"),
		{
			Role: luft.RoleAssistant,
			Content: []luft.ContentBlock{
				{Type: luft.TypeText, Text: "let me check"},
				{Type: luft.TypeToolUse, Name: "calc", Input: json.RawMessage(`{"a":1}`)},
			},
		},
	}
	out, err := messagesToGemini(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("contents: %d", len(out))
	}
	if out[0].Role != "user" {
		t.Errorf("user role: %q", out[0].Role)
	}
	if out[1].Role != "model" {
		t.Errorf("assistant must map to model: %q", out[1].Role)
	}
	if out[1].Parts[1].FunctionCall == nil || out[1].Parts[1].FunctionCall.Name != "calc" {
		t.Errorf("functionCall mapping: %+v", out[1].Parts[1])
	}
}

func TestMessagesToGemini_ToolResultIDMapping(t *testing.T) {
	msg := luft.NewToolResultMessage([]luft.ToolResult{
		{ToolUseID: "calc::0", Content: "3"},
		{ToolUseID: "lookup_user::1", Content: "alice"},
		{ToolUseID: "external_legacy_id", Content: "x"},
	})
	out, err := messagesToGemini([]luft.Message{msg})
	if err != nil {
		t.Fatal(err)
	}
	parts := out[0].Parts
	if parts[0].FunctionResponse.Name != "calc" {
		t.Errorf("calc name: %q", parts[0].FunctionResponse.Name)
	}
	if parts[1].FunctionResponse.Name != "lookup_user" {
		t.Errorf("multi-underscore name: %q", parts[1].FunctionResponse.Name)
	}
	if parts[2].FunctionResponse.Name != "external_legacy_id" {
		t.Errorf("id without separator should pass through: %q", parts[2].FunctionResponse.Name)
	}
}

func TestToolResultResponse(t *testing.T) {
	if got := string(toolResultResponse("hello", false)); got != `{"output":"hello"}` {
		t.Errorf("success: %s", got)
	}
	if got := string(toolResultResponse("boom", true)); got != `{"error":"boom"}` {
		t.Errorf("error: %s", got)
	}
}

func TestCandidateToLuft_SynthesizesIDs(t *testing.T) {
	c := geminiContent{
		Role: "model",
		Parts: []geminiPart{
			{Text: "thinking..."},
			{FunctionCall: &geminiFunctionCall{Name: "calc", Args: json.RawMessage(`{"a":1}`)}},
			{FunctionCall: &geminiFunctionCall{Name: "calc", Args: json.RawMessage(`{"a":2}`)}},
		},
	}
	out := candidateToLuft(c)
	if len(out) != 3 {
		t.Fatalf("blocks: %d", len(out))
	}
	if out[1].ID != "calc::1" || out[2].ID != "calc::2" {
		t.Errorf("ids: %q %q", out[1].ID, out[2].ID)
	}
	// Round-trip through nameFromToolUseID.
	if got := nameFromToolUseID(out[1].ID); got != "calc" {
		t.Errorf("round-trip: %q", got)
	}
}

func TestBuildGenerateRequest_SystemAndTools(t *testing.T) {
	tool := luft.Tool{
		Name:        "calc",
		Description: "Add",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"}}}`),
	}
	req := luft.ProviderRequest{
		System:    "be helpful",
		MaxTokens: 256,
		Messages:  []luft.Message{luft.NewUserMessage("hi")},
		Tools:     []luft.Tool{tool},
	}
	wire, err := buildGenerateRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if wire.SystemInstruction == nil || wire.SystemInstruction.Parts[0].Text != "be helpful" {
		t.Errorf("systemInstruction: %+v", wire.SystemInstruction)
	}
	if wire.GenerationConfig == nil || wire.GenerationConfig.MaxOutputTokens != 256 {
		t.Errorf("generationConfig: %+v", wire.GenerationConfig)
	}
	if len(wire.Tools) != 1 || len(wire.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("tools: %+v", wire.Tools)
	}
	if wire.Tools[0].FunctionDeclarations[0].Name != "calc" {
		t.Errorf("decl name: %+v", wire.Tools[0].FunctionDeclarations[0])
	}
}

func TestProvider_Call(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
            "candidates":[{
                "content":{"role":"model","parts":[{"text":"hello back"}]},
                "finishReason":"STOP"
            }],
            "usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":5,"totalTokenCount":17}
        }`))
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	resp, err := p.Call(context.Background(), luft.ProviderRequest{
		Model:    "gemini-2.5-pro",
		System:   "be helpful",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	wantPath := "/v1/projects/test-proj/locations/us-central1/publishers/google/models/gemini-2.5-pro:generateContent"
	if gotPath != wantPath {
		t.Errorf("path: got %s want %s", gotPath, wantPath)
	}
	if gotAuth != "Bearer TEST-TOKEN" {
		t.Errorf("auth: %q", gotAuth)
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

func TestProvider_CallToolUse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
            "candidates":[{
                "content":{"role":"model","parts":[
                    {"functionCall":{"name":"calc","args":{"a":1,"b":2}}}
                ]},
                "finishReason":"STOP"
            }],
            "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}
        }`))
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	resp, err := p.Call(context.Background(), luft.ProviderRequest{
		Model:    "gemini-2.5-pro",
		Messages: []luft.Message{luft.NewUserMessage("calc 1+2")},
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stopReason: %q", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != luft.TypeToolUse {
		t.Fatalf("content: %+v", resp.Content)
	}
	if !strings.HasPrefix(resp.Content[0].ID, "calc::") {
		t.Errorf("id format: %q", resp.Content[0].ID)
	}
}

func TestProvider_CallError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"code":400,"message":"bad model","status":"INVALID_ARGUMENT"}}`))
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	_, err := p.Call(context.Background(), luft.ProviderRequest{
		Model:    "x",
		Messages: []luft.Message{luft.NewUserMessage("hi")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *luft.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *luft.APIError, got %T", err)
	}
	if apiErr.StatusCode != 400 || apiErr.Type != "INVALID_ARGUMENT" {
		t.Errorf("apiErr: %+v", apiErr)
	}
}

func TestProvider_Stream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("alt") != "sse" {
			t.Errorf("missing alt=sse: %s", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}`)
		fmt.Fprintln(w, ``)
		fmt.Fprintln(w, `data: {"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]}}]}`)
		fmt.Fprintln(w, ``)
		fmt.Fprintln(w, `data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"totalTokenCount":12}}`)
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	var deltas []string
	resp, err := p.Stream(context.Background(),
		luft.ProviderRequest{Model: "gemini-2.5-pro", Messages: []luft.Message{luft.NewUserMessage("hi")}},
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

func TestProvider_StreamFunctionCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"calc","args":{"a":1}}}]},"finishReason":"STOP"}]}`)
	}))
	defer srv.Close()

	p := testProvider(t, srv.URL)
	resp, err := p.Stream(context.Background(),
		luft.ProviderRequest{Model: "gemini-2.5-pro", Messages: []luft.Message{luft.NewUserMessage("hi")}},
		func(b luft.ContentBlock) {},
	)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != luft.TypeToolUse {
		t.Fatalf("content: %+v", resp.Content)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stopReason: %q", resp.StopReason)
	}
}

func TestProvider_GlobalLocation(t *testing.T) {
	p, err := NewProvider(Config{
		Project:     "p",
		Location:    "global",
		TokenSource: staticTokenSource{tok: "x"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(p.cfg.BaseURL, "https://aiplatform.googleapis.com") {
		t.Errorf("global base URL: %s", p.cfg.BaseURL)
	}
}

// ---- auth tests ----

func makeTestSA(t *testing.T) (ServiceAccount, *rsa.PrivateKey) {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	pkcs8, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: pkcs8})
	return ServiceAccount{
		Type:        "service_account",
		ProjectID:   "test-proj",
		ClientEmail: "svc@test-proj.iam.gserviceaccount.com",
		PrivateKey:  string(pemBytes),
	}, key
}

func TestServiceAccountTokenSource_Exchange(t *testing.T) {
	sa, _ := makeTestSA(t)

	var gotAssertion, gotGrant string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatal(err)
		}
		gotAssertion = r.Form.Get("assertion")
		gotGrant = r.Form.Get("grant_type")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"access_token":"AT-123","expires_in":3600,"token_type":"Bearer"}`))
	}))
	defer srv.Close()

	sa.TokenURI = srv.URL
	ts, err := newServiceAccountTokenSource(sa, srv.Client(), func() time.Time {
		return time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
	})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := ts.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if tok != "AT-123" {
		t.Errorf("token: %q", tok)
	}
	if gotGrant != jwtGrantTyp {
		t.Errorf("grant_type: %q", gotGrant)
	}
	if !strings.Contains(gotAssertion, ".") {
		t.Errorf("assertion not JWT-shaped: %q", gotAssertion)
	}

	// Second call should be cached (server would 500 if hit again).
	srv.Close()
	tok2, err := ts.Token(context.Background())
	if err != nil {
		t.Fatalf("cached Token: %v", err)
	}
	if tok2 != "AT-123" {
		t.Errorf("cached token: %q", tok2)
	}
}

func TestParsePrivateKey_RoundTrip(t *testing.T) {
	sa, originalKey := makeTestSA(t)
	parsed, err := parsePrivateKey(sa.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.N.Cmp(originalKey.N) != 0 {
		t.Error("parsed modulus differs from original")
	}
}

func TestLoadServiceAccountFile(t *testing.T) {
	sa, _ := makeTestSA(t)
	data, _ := json.Marshal(sa)
	path := t.TempDir() + "/sa.json"
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadServiceAccountFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.ClientEmail != sa.ClientEmail {
		t.Errorf("client_email: %q vs %q", loaded.ClientEmail, sa.ClientEmail)
	}
}
