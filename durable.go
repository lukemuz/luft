package luft

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
)

// ToolResultStore is a minimal key/value store for durable tool outputs.
// Get returns the stored value for key and whether one was found; a non-nil
// error indicates a store failure rather than a miss. Put records value under
// key, overwriting any prior entry.
//
// The in-memory MemoryToolResultStore satisfies it; callers may supply a
// persistent implementation (file, SQL, ...) to survive process restarts.
type ToolResultStore interface {
	Get(key string) (value string, found bool, err error)
	Put(key string, value string) error
}

// MemoryToolResultStore is a concurrency-safe in-memory ToolResultStore.
// The zero value is not usable; construct one with NewMemoryToolResultStore.
type MemoryToolResultStore struct {
	mu sync.RWMutex
	m  map[string]string
}

// NewMemoryToolResultStore returns an empty, ready-to-use MemoryToolResultStore.
func NewMemoryToolResultStore() *MemoryToolResultStore {
	return &MemoryToolResultStore{m: make(map[string]string)}
}

// Get returns the value for key and true if one was recorded.
func (s *MemoryToolResultStore) Get(key string) (string, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.m[key]
	return v, ok, nil
}

// Put records value under key, overwriting any prior entry.
func (s *MemoryToolResultStore) Put(key string, value string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[key] = value
	return nil
}

// durableKey returns the idempotency key for a tool call: the hex-encoded
// SHA-256 of the tool name, a NUL separator, and the raw input bytes. The
// separator prevents collisions between (name, input) pairs that would
// concatenate ambiguously.
func durableKey(toolName string, input json.RawMessage) string {
	h := sha256.New()
	h.Write([]byte(toolName))
	h.Write([]byte{0})
	h.Write(input)
	return hex.EncodeToString(h.Sum(nil))
}

// WithDurability returns a Middleware that makes a tool's successful results
// durable: on the first successful call the output is recorded in store, and
// an identical subsequent call returns the recorded value without invoking the
// underlying tool (or any inner middleware). A call that returns an error is
// never recorded, so failed calls remain retryable.
//
// Idempotency key. ToolFunc does not receive the tool-use ID — the agent loop
// (agent.go, dispatch) knows use.ID but never places it in the context handed
// to the tool, so the middleware seam cannot see it. The key is therefore
// SHA-256(toolName + "\x00" + input). Limitation: two genuinely distinct calls
// that carry the same tool name and byte-identical input collapse to a single
// execution and the second call returns the first's cached result. This is the
// intended semantics for side-effectful, idempotent tools surviving a
// crash-resume, but it is wrong for tools where repeated identical calls should
// re-run. WithDurability is opt-in; apply it only to tools whose identical
// calls are safe to deduplicate.
//
// Concurrency. The lookup-then-execute is not atomic: two identical calls in
// flight at the same time both miss the store and both run, because dedup keys
// off a completed, recorded result. That is harmless for the crash-resume case
// this targets (the duplicate spans a restart, not a concurrent window), but
// WithDurability is not a substitute for a per-key in-flight lock.
//
// Stack order. List WithDurability first in Wrap so it is outermost. On a
// cache hit it short-circuits before timeout, logging, result limiting, and
// confirmation middlewares run, so (for example) a replayed call is not logged
// as a fresh execution and does not start a timeout. On a miss the inner stack
// runs normally and the successful result is recorded once it returns.
func WithDurability(store ToolResultStore) Middleware {
	return func(b ToolBinding) ToolFunc {
		return func(ctx context.Context, input json.RawMessage) (string, error) {
			key := durableKey(b.Tool.Name, input)
			if v, found, err := store.Get(key); err == nil && found {
				return v, nil
			}
			out, err := b.Func(ctx, input)
			if err == nil {
				_ = store.Put(key, out)
			}
			return out, err
		}
	}
}
