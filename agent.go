package luft

import (
	"context"
	"fmt"
	"time"
)

// Config holds everything needed to create a Client.
type Config struct {
	Provider  Provider    // required — the LLM backend to use
	Model     string      // required — provider-specific model identifier
	MaxTokens int         // max tokens per response; defaults to 1024
	Retry     RetryConfig // controls automatic retry behaviour for transient API errors

	// Recorder, if non-nil, receives Events as Loop / LoopStream runs:
	// turn start/end, model request/response, retry attempts, and tool
	// call start/end. See recorder.go for event semantics. Recording is
	// best-effort and must not block the loop; implementations should be
	// fast and non-blocking. Ask and AskStream do not emit events.
	Recorder Recorder

	// SystemCache, when set, marks the system prompt as a cache breakpoint
	// on every API call this Client makes via Ask, AskStream, Loop, and
	// LoopStream. The most useful single cache placement when the system
	// text is long and stable across turns.
	SystemCache *CacheControl

	// Generation sets optional sampling controls (temperature, top_p, stop
	// sequences, …) applied to every call this Client makes. The zero value
	// leaves provider defaults untouched. See GenerationParams.
	Generation GenerationParams
}

// Well-known Anthropic model identifiers.
const (
	ModelOpus   = "claude-opus-4-7"
	ModelSonnet = "claude-sonnet-4-6"
	ModelHaiku  = "claude-haiku-4-5-20251001"
)

// Client is a stateless API facade. It holds configuration but no conversation
// state — history is owned by the caller. The same Client is safe for
// concurrent use across goroutines.
type Client struct {
	cfg Config
}

// New creates a Client from cfg, filling in defaults for zero-value fields.
// Returns an error if Provider or Model is empty.
func New(cfg Config) (*Client, error) {
	if cfg.Provider == nil {
		return nil, fmt.Errorf("luft: Config.Provider is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("luft: Config.Model is required")
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = defaultMaxTokens
	}
	return &Client{cfg: cfg}, nil
}

// WithModel returns a new Client that shares the provider, MaxTokens, and
// Retry config with c, but uses a different model. Useful for cost-tiering
// — a cheap summarizer alongside a smart loop, for example. The returned
// Client is independent: mutations to one do not affect the other.
//
// For more elaborate derivation (different MaxTokens, different Retry),
// construct a fresh Client with luft.New.
func (c *Client) WithModel(model string) *Client {
	cfg := c.cfg
	cfg.Model = model
	return &Client{cfg: cfg}
}

// WithRecorder returns a new Client that shares the rest of c's config but
// replaces the Recorder. Pass nil to disable recording on the derived Client.
// The returned Client is independent of c.
func (c *Client) WithRecorder(rec Recorder) *Client {
	cfg := c.cfg
	cfg.Recorder = rec
	return &Client{cfg: cfg}
}

// Ask makes a single LLM call and returns the model's reply along with token
// usage for the call.
//
// system sets the system prompt; pass "" to omit it.
// history is the conversation so far and is not modified by Ask.
// Append the returned Message to your history slice to continue the conversation.
//
// Token usage is reported separately so callers can aggregate cost across
// multiple Ask calls without paying for a Loop. Pre-1.0 note: Ask previously
// returned (Message, error); the Usage return was added so cost-conscious
// callers don't have to fall back to Loop with an empty toolset.
func (c *Client) Ask(ctx context.Context, system string, history []Message) (Message, Usage, error) {
	req := ProviderRequest{
		Model:       c.cfg.Model,
		MaxTokens:   c.cfg.MaxTokens,
		System:      system,
		Messages:    history,
		SystemCache: c.cfg.SystemCache,
		Generation:  c.cfg.Generation,
	}
	resp, err := callWithRetry(ctx, c.cfg.Retry, func() (ProviderResponse, error) {
		return c.cfg.Provider.Call(ctx, req)
	})
	if err != nil {
		return Message{}, Usage{}, err
	}
	return Message{Role: RoleAssistant, Content: resp.Content}, resp.Usage, nil
}

// AskStream is the streaming variant of Ask. It invokes the onToken
// callback for every ContentBlock delta delivered by the provider
// (typically incremental TypeText blocks). The final assembled
// Message is returned once the stream completes. history is not
// modified by this call.
//
// Retry interaction: callWithRetry wraps the stream call, so onToken may fire
// for partial content on a failed attempt before a successful retry begins.
// Use StreamBuffer with RetryConfig.OnRetry to react to retries and clear
// partial output before the next attempt starts.
//
// onToken may be nil, in which case token deltas are discarded.
//
// Pre-1.0 note: AskStream previously returned (Message, error); Usage was
// added so streaming callers don't have to drop down to Loop for cost tracking.
func (c *Client) AskStream(ctx context.Context, system string, history []Message, onToken func(ContentBlock)) (Message, Usage, error) {
	if onToken == nil {
		onToken = func(ContentBlock) {}
	}
	req := ProviderRequest{
		Model:       c.cfg.Model,
		MaxTokens:   c.cfg.MaxTokens,
		System:      system,
		Messages:    history,
		SystemCache: c.cfg.SystemCache,
		Generation:  c.cfg.Generation,
	}
	resp, err := callWithRetry(ctx, c.cfg.Retry, func() (ProviderResponse, error) {
		return c.cfg.Provider.Stream(ctx, req, onToken)
	})
	if err != nil {
		return Message{}, Usage{}, err
	}
	return Message{Role: RoleAssistant, Content: resp.Content}, resp.Usage, nil
}

// LoopResult is returned by Loop and carries the complete updated history
// together with aggregate token usage across all API calls in the run.
type LoopResult struct {
	Messages []Message // full conversation: original history + all new turns
	Usage    Usage     // total tokens consumed across all iterations
}

// Final returns the last message in Messages, which is conventionally the
// final assistant reply. Returns the zero Message if Messages is empty.
func (r LoopResult) Final() Message {
	if len(r.Messages) == 0 {
		return Message{}
	}
	return r.Messages[len(r.Messages)-1]
}

// FinalText returns the concatenated text of the final assistant message.
// Equivalent to TextContent(r.Final()). Returns "" if no messages are present.
func (r LoopResult) FinalText() string {
	return TextContent(r.Final())
}

// Loop runs the agent in a tool-use loop until the model signals end_turn or
// an error occurs. It returns the full conversation including all new turns.
//
// tools is the Toolset advertised to the model on every call. A tool name that
// appears in a model response but is absent from the toolset's dispatch map
// causes an immediate LoopError wrapping ErrMissingTool.
// maxIter caps the total number of API calls; 0 means no limit.
//
// The Toolset is read once: its Tools() and Dispatch() are computed at the
// start of the loop. Mutating the underlying Bindings during the loop has
// no effect on iterations already in flight.
func (c *Client) Loop(
	ctx context.Context,
	system string,
	history []Message,
	tools Toolset,
	maxIter int,
) (LoopResult, error) {
	return c.runLoop(ctx, system, history, tools, maxIter, loopOpts{}, nil, nil)
}

// retryWithRecorder returns a copy of cfg whose OnRetry callback also emits a
// RetryAttempt event into rec. The user's existing OnRetry, if any, is still
// invoked. cfg is unchanged.
func retryWithRecorder(cfg RetryConfig, rec Recorder, ctx context.Context, turnID string, iter int) RetryConfig {
	if rec == nil {
		return cfg
	}
	user := cfg.OnRetry
	cfg.OnRetry = func(attempt int, wait time.Duration) {
		if user != nil {
			user(attempt, wait)
		}
		emit(ctx, rec, Event{
			TurnID: turnID, Iter: iter, Type: EventRetryAttempt,
			Attempt: attempt, Wait: wait,
		})
	}
	return cfg
}

// LoopStream is the streaming variant of Loop. It mirrors Loop's structure,
// control flow, error handling, stop-reason switching, maxIter limiting, and
// history/usage accumulation but invokes Provider.Stream (wrapped by
// callWithRetry) on every iteration so that onToken receives each
// ContentBlock delta as it arrives. After runTools completes, onToolResult
// is called with the results (allowing live UI updates or logging) before
// the tool results are appended and the loop continues.
//
// Retry interaction: onToken may fire multiple times for a given turn if retries
// occur. Use StreamBuffer with RetryConfig.OnRetry to react to retries and
// clear partial output before the next attempt starts.
//
// Both callbacks may be nil, in which case their respective events are discarded.
func (c *Client) LoopStream(
	ctx context.Context,
	system string,
	history []Message,
	tools Toolset,
	maxIter int,
	onToken func(ContentBlock),
	onToolResult func([]ToolResult),
) (LoopResult, error) {
	if onToken == nil {
		onToken = func(ContentBlock) {}
	}
	if onToolResult == nil {
		onToolResult = func([]ToolResult) {}
	}
	return c.runLoop(ctx, system, history, tools, maxIter, loopOpts{}, onToken, onToolResult)
}

// loopOpts carries optional per-iteration hooks for the internal loop body.
// All fields are nil by default and unused by Loop / LoopStream — they exist
// so Agent can layer mid-loop trimming and per-iteration observation on top
// of the shared loop without duplicating its body.
type loopOpts struct {
	// beforeIter, if non-nil, transforms the working history before each
	// model call. Used by Agent for mid-loop context trimming. If it returns
	// an error, the loop aborts and the error is wrapped in LoopError.
	beforeIter func(ctx context.Context, history []Message) ([]Message, error)
	// onIter, if non-nil, observes the effective history immediately before
	// each model call. Used by Agent.Hooks.OnIteration.
	onIter func(ctx context.Context, iter int, history []Message)
}

// runLoop is the shared implementation behind Loop, LoopStream, and the
// Agent block. If onToken is nil the provider is invoked via Call; otherwise
// via Stream(req, onToken). onToolResult, if non-nil, is invoked with each
// tool batch's results before they are appended to history.
func (c *Client) runLoop(
	ctx context.Context,
	system string,
	history []Message,
	tools Toolset,
	maxIter int,
	opts loopOpts,
	onToken func(ContentBlock),
	onToolResult func([]ToolResult),
) (LoopResult, error) {
	toolDefs := tools.Tools()
	dispatch := tools.Dispatch()
	terminals := terminalSet(tools.Bindings)
	msgs := make([]Message, len(history))
	copy(msgs, history)
	var total Usage

	rec := c.cfg.Recorder
	turnID := ""
	if rec != nil {
		turnID = newTurnID()
		emit(ctx, rec, Event{TurnID: turnID, Type: EventTurnStart, History: msgs})
	}

	for iter := 0; maxIter == 0 || iter < maxIter; iter++ {
		if opts.beforeIter != nil {
			trimmed, err := opts.beforeIter(ctx, msgs)
			if err != nil {
				emit(ctx, rec, Event{TurnID: turnID, Iter: iter, Type: EventTurnError, Err: err.Error()})
				return LoopResult{Messages: msgs, Usage: total}, &LoopError{Iter: iter, Cause: err}
			}
			msgs = trimmed
		}
		if opts.onIter != nil {
			opts.onIter(ctx, iter, msgs)
		}
		req := ProviderRequest{
			Model:         c.cfg.Model,
			MaxTokens:     c.cfg.MaxTokens,
			System:        system,
			Messages:      msgs,
			Tools:         toolDefs,
			ProviderTools: tools.ProviderTools,
			SystemCache:   c.cfg.SystemCache,
			Generation:    c.cfg.Generation,
		}
		emit(ctx, rec, Event{TurnID: turnID, Iter: iter, Type: EventModelRequest})
		var resp ProviderResponse
		var err error
		retry := retryWithRecorder(c.cfg.Retry, rec, ctx, turnID, iter)
		if onToken == nil {
			resp, err = callWithRetry(ctx, retry, func() (ProviderResponse, error) {
				return c.cfg.Provider.Call(ctx, req)
			})
		} else {
			resp, err = callWithRetry(ctx, retry, func() (ProviderResponse, error) {
				return c.cfg.Provider.Stream(ctx, req, onToken)
			})
		}
		if err != nil {
			emit(ctx, rec, Event{TurnID: turnID, Iter: iter, Type: EventTurnError, Err: err.Error()})
			return LoopResult{Messages: msgs, Usage: total}, &LoopError{Iter: iter, Cause: err}
		}
		total.InputTokens += resp.Usage.InputTokens
		total.OutputTokens += resp.Usage.OutputTokens
		total.CacheCreationTokens += resp.Usage.CacheCreationTokens
		total.CacheReadTokens += resp.Usage.CacheReadTokens
		assistantMsg := Message{Role: RoleAssistant, Content: resp.Content}
		msgs = append(msgs, assistantMsg)
		if rec != nil {
			usage := resp.Usage
			emit(ctx, rec, Event{
				TurnID: turnID, Iter: iter, Type: EventModelResponse,
				Message: &assistantMsg, Usage: &usage, StopReason: resp.StopReason,
			})
		}

		switch resp.StopReason {
		case "end_turn":
			emit(ctx, rec, Event{TurnID: turnID, Iter: iter, Type: EventTurnEnd, Usage: &total})
			return LoopResult{Messages: msgs, Usage: total}, nil

		case "tool_use":
			results, err := runTools(ctx, resp.Content, dispatch, rec, turnID, iter)
			if err != nil {
				emit(ctx, rec, Event{TurnID: turnID, Iter: iter, Type: EventTurnError, Err: err.Error()})
				return LoopResult{Messages: msgs, Usage: total}, &LoopError{Iter: iter, Cause: err}
			}
			if onToolResult != nil {
				onToolResult(results)
			}
			msgs = append(msgs, NewToolResultMessage(results))
			if terminatedByTool(resp.Content, results, terminals) {
				emit(ctx, rec, Event{TurnID: turnID, Iter: iter, Type: EventTurnEnd, Usage: &total})
				return LoopResult{Messages: msgs, Usage: total}, nil
			}

		case "max_tokens":
			cause := fmt.Errorf("model hit max_tokens limit; increase Config.MaxTokens")
			emit(ctx, rec, Event{TurnID: turnID, Iter: iter, Type: EventTurnError, Err: cause.Error()})
			return LoopResult{Messages: msgs, Usage: total}, &LoopError{Iter: iter, Cause: cause}

		default:
			cause := fmt.Errorf("unexpected stop_reason %q", resp.StopReason)
			emit(ctx, rec, Event{TurnID: turnID, Iter: iter, Type: EventTurnError, Err: cause.Error()})
			return LoopResult{Messages: msgs, Usage: total}, &LoopError{Iter: iter, Cause: cause}
		}
	}
	emit(ctx, rec, Event{TurnID: turnID, Iter: maxIter, Type: EventTurnError, Err: ErrMaxIter.Error()})
	return LoopResult{Messages: msgs, Usage: total}, &LoopError{Iter: maxIter, Cause: ErrMaxIter}
}

// terminalSet collects names of bindings whose Meta.Terminal is set, so Loop
// can recognise them after a tool batch runs. Returns nil if there are none
// (the common case) so the per-iteration check is essentially free.
func terminalSet(bindings []ToolBinding) map[string]bool {
	var s map[string]bool
	for _, b := range bindings {
		if b.Meta.Terminal {
			if s == nil {
				s = make(map[string]bool)
			}
			s[b.Tool.Name] = true
		}
	}
	return s
}

// terminatedByTool reports whether content contains a tool_use of a terminal
// tool whose result was not an error. The IsError check matters: we don't
// want a validator that rejected the model's submission to count as "done".
func terminatedByTool(content []ContentBlock, results []ToolResult, terminals map[string]bool) bool {
	if len(terminals) == 0 {
		return false
	}
	okByID := make(map[string]bool, len(results))
	for _, r := range results {
		if !r.IsError {
			okByID[r.ToolUseID] = true
		}
	}
	for _, b := range content {
		if b.Type == TypeToolUse && terminals[b.Name] && okByID[b.ID] {
			return true
		}
	}
	return false
}

// runTools executes all tool_use blocks in content concurrently and returns
// their results in the original order. Individual tool errors become
// is_error=true results so the model can see and recover from them. A missing
// tool is a programming error (wrong dispatch map) and aborts before any
// goroutines are spawned.
func runTools(ctx context.Context, content []ContentBlock, dispatch map[string]ToolFunc, rec Recorder, turnID string, iter int) ([]ToolResult, error) {
	uses := extractToolUses(content)

	// Validate all names up front so a missing tool aborts immediately rather
	// than after other calls have already started. A nil func is treated as
	// missing — advertising a tool without an implementation is a programmer
	// error, not a runtime condition the model should see.
	for _, use := range uses {
		fn, ok := dispatch[use.Name]
		if !ok || fn == nil {
			return nil, &ToolError{ToolName: use.Name, ToolUseID: use.ID, Cause: ErrMissingTool}
		}
	}

	type indexed struct {
		i      int
		result ToolResult
	}
	ch := make(chan indexed, len(uses))
	for i, use := range uses {
		i, use := i, use
		go func() {
			emit(ctx, rec, Event{
				TurnID: turnID, Iter: iter, Type: EventToolCallStart,
				ToolUseID: use.ID, ToolName: use.Name, ToolInput: use.Input,
			})
			toolCtx, sink := withImageSink(ctx)
			output, err := dispatch[use.Name](toolCtx, use.Input)
			r := ToolResult{ToolUseID: use.ID}
			endEv := Event{
				TurnID: turnID, Iter: iter, Type: EventToolCallEnd,
				ToolUseID: use.ID, ToolName: use.Name,
			}
			if err != nil {
				r.Content = err.Error()
				r.IsError = true
				endEv.IsError = true
				endEv.ToolError = err.Error()
			} else {
				r.Content = output
				r.Images = sink.drain()
				endEv.ToolOutput = output
			}
			emit(ctx, rec, endEv)
			ch <- indexed{i: i, result: r}
		}()
	}
	results := make([]ToolResult, len(uses))
	for range uses {
		it := <-ch
		results[it.i] = it.result
	}
	return results, nil
}
