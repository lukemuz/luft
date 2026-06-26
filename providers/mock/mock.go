// Package mock provides a scripted [github.com/lukemuz/luft.Provider]
// implementation for offline testing of agent loops and tools.
//
// A mock Provider returns a pre-programmed sequence of responses — one per
// Call, in order — so tests can drive a real luft.Agent or luft.Client
// through a multi-turn tool-use conversation without hitting a live LLM API.
// Every request the mock receives is captured and exposed via Requests, so
// tests can assert on exactly what the agent sent.
//
// Both Call and Stream are supported. Stream replays the chosen response's
// ContentBlocks through the onDelta callback (text blocks at minimum) and
// then returns the same aggregated ProviderResponse that Call would have
// returned, so streaming and non-streaming code paths behave identically.
//
// When the canned responses are exhausted the next Call (or Stream) returns
// a descriptive error rather than panicking, so a mis-sized script fails
// loudly instead of silently looping or hanging.
package mock

import (
	"context"
	"fmt"
	"sync"

	"github.com/lukemuz/luft"
)

// Provider is a scripted Provider implementation for offline testing. It
// returns the responses passed to New in order, one per Call, and records
// every ProviderRequest it receives for later inspection via Requests.
//
// A Provider is safe for concurrent use, though tests that race concurrent
// Calls against a single scripted sequence are inherently order-dependent
// and usually better served by one Provider per goroutine.
type Provider struct {
	mu        sync.Mutex
	responses []luft.ProviderResponse
	requests  []luft.ProviderRequest
	idx       int
}

// New returns a Provider that replies to each Call (and Stream) with the
// next response in responses, in order. Responses are consumed one at a
// time: the first Call gets responses[0], the second gets responses[1],
// and so on. When the sequence is exhausted, subsequent calls return a
// descriptive error.
func New(responses ...luft.ProviderResponse) *Provider {
	return &Provider{
		responses: append([]luft.ProviderResponse(nil), responses...),
	}
}

// Call returns the next canned response, recording req for later
// inspection. If the scripted responses are exhausted it returns an error
// describing how many calls were expected versus received, so a test whose
// script is too short fails with a clear message instead of panicking.
func (p *Provider) Call(_ context.Context, req luft.ProviderRequest) (luft.ProviderResponse, error) {
	p.mu.Lock()
	p.requests = append(p.requests, req)
	if p.idx >= len(p.responses) {
		n := p.idx + 1
		p.mu.Unlock()
		return luft.ProviderResponse{}, fmt.Errorf("mock: scripted responses exhausted (have %d, this is call #%d); add another response to mock.New", len(p.responses), n)
	}
	resp := p.responses[p.idx]
	p.idx++
	p.mu.Unlock()
	return resp, nil
}

// Stream replays the chosen response's ContentBlocks through onDelta and
// then returns the same aggregated ProviderResponse that Call would have
// returned for this position in the script. Text blocks (TypeText) are
// delivered as individual deltas; other block types (e.g. tool_use) are
// emitted whole, since they have no natural incremental form in the
// canonical content model.
//
// Like Call, Stream records the request and returns a descriptive error
// when the scripted responses are exhausted. The two methods share a single
// cursor, so interleaving Call and Stream consumes the sequence in order.
func (p *Provider) Stream(ctx context.Context, req luft.ProviderRequest, onDelta func(luft.ContentBlock)) (luft.ProviderResponse, error) {
	resp, err := p.Call(ctx, req)
	if err != nil {
		return luft.ProviderResponse{}, err
	}
	if onDelta != nil {
		for _, block := range resp.Content {
			if block.Type == luft.TypeText && block.Text != "" {
				onDelta(luft.ContentBlock{Type: luft.TypeText, Text: block.Text})
			} else {
				onDelta(block)
			}
		}
	}
	return resp, nil
}

// Requests returns the slice of every ProviderRequest the mock has received,
// in the order it received them. The returned slice is a copy, so tests can
// mutate it freely without affecting the mock's internal state.
func (p *Provider) Requests() []luft.ProviderRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]luft.ProviderRequest(nil), p.requests...)
}

// Reset restores the mock to its initial state: response cursor back to the
// start and the captured request log cleared. Useful when a single test
// helper wants to reuse one scripted Provider across independent subtests.
func (p *Provider) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.idx = 0
	p.requests = nil
}
