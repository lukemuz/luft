// Recipe 08-http-sse: a net/http server that streams a luft Agent's replies
// to the client as Server-Sent Events, keeping per-session conversation
// history in an in-memory store.
//
//	POST /chat     — {"session":"<id>","message":"<text>"} → text/event-stream
//	GET  /healthz  — liveness probe
//
// The luft pattern stays visible: load history → append user message →
// StepStream (writing each text delta as an SSE data line) → save history.
// The Agent and store live on a Server struct so tests inject a mock-backed
// Agent without touching main().
//
// Local run:
//
//	export OPENROUTER_API_KEY=...
//	go run ./examples/recipes/08-http-sse
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/providers/openrouter"
)

const systemPrompt = "You are a concise, helpful assistant."

// inMemoryStore is a session-keyed conversation log guarded by a single
// mutex. One lock serializes all sessions — simple and fine for a single
// process; cross-replica or high-throughput deploys should back this with a
// database (see recipe 07-web-service for a FileStore-backed variant).
type inMemoryStore struct {
	mu       sync.Mutex
	sessions map[string][]luft.Message
}

func newStore() *inMemoryStore {
	return &inMemoryStore{sessions: make(map[string][]luft.Message)}
}

// Get returns a copy of the session's history (nil if absent), so the caller
// may append freely without aliasing the stored slice before Put replaces it.
func (s *inMemoryStore) Get(id string) []luft.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]luft.Message(nil), s.sessions[id]...)
}

// Put replaces the session's history wholesale with the post-turn slice.
func (s *inMemoryStore) Put(id string, history []luft.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[id] = history
}

// Server wires a luft Agent and a session store. The handler is a method on
// *Server so the agent and store are injected — this is what makes the
// handler testable with a mock provider.
type Server struct {
	Agent luft.Agent
	Store *inMemoryStore
}

func NewServer(agent luft.Agent) *Server {
	return &Server{Agent: agent, Store: newStore()}
}

type chatRequest struct {
	Session string `json:"session"`
	Message string `json:"message"`
}

// HandleChat is the one place the luft pattern lives:
// load history → append user message → StepStream (streaming text deltas as
// SSE data lines) → save history.
//
// Headers are written lazily, on the first token: a pre-stream agent error
// can still return a clean 500, while a mid-stream error (headers already
// committed) is reported as an SSE "error" event.
func (s *Server) HandleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.Session == "" || req.Message == "" {
		writeJSONError(w, http.StatusBadRequest, "session and message are required")
		return
	}

	history := s.Store.Get(req.Session)
	history = append(history, luft.NewUserMessage(req.Message))

	flusher, _ := w.(http.Flusher)
	headersWritten := false
	startStream := func() {
		if headersWritten {
			return
		}
		w.Header().Set("content-type", "text/event-stream")
		w.Header().Set("cache-control", "no-cache")
		w.Header().Set("connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		headersWritten = true
	}

	onToken := func(b luft.ContentBlock) {
		if b.Type != luft.TypeText || b.Text == "" {
			return
		}
		startStream()
		writeSSE(w, "", b.Text)
		if flusher != nil {
			flusher.Flush()
		}
	}

	result, err := s.Agent.StepStream(r.Context(), history, onToken, nil)
	if err != nil {
		if !headersWritten {
			writeJSONError(w, http.StatusInternalServerError, "agent error: "+err.Error())
			return
		}
		writeSSE(w, "error", err.Error())
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	// If the turn produced no text deltas (e.g. tool calls only), still open
	// the stream so the client gets a well-formed response.
	if !headersWritten {
		startStream()
		if ft := result.FinalText(); ft != "" {
			writeSSE(w, "", ft)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}

	s.Store.Put(req.Session, result.Messages)
}

// writeSSE writes one Server-Sent Event. data is split on newlines into
// multiple "data:" lines as the SSE spec requires — a bare newline in the
// payload would otherwise end the event early and silently drop everything
// after the first line (easy to miss, since single-line replies look fine).
func writeSSE(w io.Writer, event, data string) {
	if event != "" {
		fmt.Fprintf(w, "event: %s\n", event)
	}
	for _, line := range strings.Split(data, "\n") {
		fmt.Fprintf(w, "data: %s\n", line)
	}
	fmt.Fprint(w, "\n")
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("content-type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	provider, err := openrouter.NewProviderFromEnv()
	if err != nil {
		log.Fatal(err)
	}

	model := envOr("MODEL", "anthropic/claude-sonnet-4-6")
	client, err := luft.New(luft.Config{
		Provider:  provider,
		Model:     model,
		MaxTokens: 4096,
	})
	if err != nil {
		log.Fatal(err)
	}

	srv := NewServer(luft.Agent{
		Client:  client,
		System:  systemPrompt,
		MaxIter: 4,
	})

	mux := http.NewServeMux()
	mux.HandleFunc("/chat", srv.HandleChat)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok"))
	})

	addr := ":" + envOr("PORT", "8080")
	log.Printf("listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

