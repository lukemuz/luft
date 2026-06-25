// Package subagent packages an Agent (client + system prompt + toolset)
// as a single tool the parent agent can call.
//
// Why: a subagent solves three problems at once.
//
//  1. Cost tiering. Bind the subagent to a cheap model (Haiku) and the
//     parent to a smart one (Sonnet/Opus); heavy inspection happens
//     cheaply while planning and edits stay on the smart model.
//
//  2. Context isolation. The subagent's iteration history — verbose
//     greps, file dumps, repeated reads — never enters the parent's
//     context. Only the final textual summary returns.
//
//  3. Parallel fan-out. The parent can call multiple subagent tools (or
//     the same subagent with different tasks) in one turn; if wrapped
//     by the batch tool they execute concurrently.
//
// The subagent runs through the Agent block with its own MaxIter cap and an
// optional ContextManager (Config.Context), so a long run is trimmed and
// summarized in place just like the top-level agent. On success the final
// assistant text is returned to the parent as the tool result. On failure,
// any partial text is included in the error so the parent has something to
// react to.
package subagent

import (
	"context"
	"fmt"

	"github.com/lukemuz/luft"
)

// Config defines a subagent's wiring.
type Config struct {
	// Name is the tool name advertised to the parent. Required.
	Name string

	// Description is the tool description shown to the parent model.
	// It should describe both *what* the subagent is good at and *what
	// shape of task* to pass in. Required.
	Description string

	// Client is the LLM client the subagent uses. Required. Typically
	// derived from the parent client via WithModel for cost tiering.
	Client *luft.Client

	// System is the subagent's system prompt. Required.
	System string

	// Tools are the toolset available to the subagent. May be empty
	// (e.g. for pure-text writer subagents).
	Tools luft.Toolset

	// Context, when its MaxTokens > 0, enables history trimming /
	// summarization for the subagent's internal loop — the same mechanism
	// the top-level Agent uses. Leave it zero for short research subagents
	// whose loops never approach the context window; set it for subagents
	// that do real, long-running work (e.g. an implementer) so a lengthy
	// run is summarized in place rather than blowing the window.
	Context luft.ContextManager

	// MaxIter caps the subagent's loop. 0 means no limit. Recommend
	// 4-12 for inspection subagents, 2-4 for writers.
	MaxIter int
}

// New returns a binding for the subagent tool. The schema is a fixed
// {"task": string}. RequiresConfirmation is left false because the
// subagent's own tools carry their own confirmation flags.
func New(cfg Config) (luft.ToolBinding, error) {
	if cfg.Name == "" || cfg.Description == "" || cfg.Client == nil || cfg.System == "" {
		return luft.ToolBinding{}, fmt.Errorf("subagent: Name, Description, Client and System are required")
	}
	type input struct {
		Task string `json:"task"`
	}
	tool, fn := luft.NewTypedTool[input](
		cfg.Name,
		cfg.Description,
		luft.InputSchema{
			Type: "object",
			Properties: map[string]luft.SchemaProperty{
				"task": {Type: "string", Description: "Self-contained task description for the subagent."},
			},
			Required: []string{"task"},
		},
		func(ctx context.Context, in input) (string, error) {
			if in.Task == "" {
				return "", fmt.Errorf("subagent %q: task is empty", cfg.Name)
			}
			// The subagent runs through the Agent block rather than Loop
			// directly so it gets the same context-trimming behaviour as the
			// top-level agent. With a zero-value Context (MaxTokens == 0) the
			// trim step is a no-op, so research subagents pay nothing for it.
			agent := luft.Agent{
				Client:  cfg.Client,
				System:  cfg.System,
				Tools:   cfg.Tools,
				Context: cfg.Context,
				MaxIter: cfg.MaxIter,
			}
			result, err := agent.Step(ctx, []luft.Message{luft.NewUserMessage(in.Task)})
			if err != nil {
				if t := lastText(result); t != "" {
					return "", fmt.Errorf("subagent %q failed mid-task: %w (partial: %s)", cfg.Name, err, truncate(t, 1000))
				}
				return "", fmt.Errorf("subagent %q: %w", cfg.Name, err)
			}
			text := result.FinalText()
			if text == "" {
				return "", fmt.Errorf("subagent %q returned no text", cfg.Name)
			}
			return text, nil
		},
	)
	return luft.ToolBinding{
		Tool: tool,
		Func: fn,
		Meta: luft.ToolMetadata{Source: "subagent/" + cfg.Name},
	}, nil
}

func lastText(r luft.LoopResult) string {
	for i := len(r.Messages) - 1; i >= 0; i-- {
		if t := luft.TextContent(r.Messages[i]); t != "" {
			return t
		}
	}
	return ""
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
