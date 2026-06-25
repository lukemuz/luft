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
	"strings"

	"github.com/lukemuz/luft"
)

// Tier binds a model-class name to the client (and thus model) that serves it.
// When a Config carries Tiers, the subagent advertises a "tier" argument so the
// parent model can pick the class per call — e.g. escalate a hard task to a
// stronger model, or keep a trivial one on the fast/cheap tier.
type Tier struct {
	// Name is the enum value advertised to the parent, e.g. "powerful".
	Name string
	// Description is the one-liner shown alongside Name in the tool schema.
	Description string
	// Client is the LLM client (carrying the model) that serves this tier.
	Client *luft.Client
}

// Config defines a subagent's wiring.
type Config struct {
	// Name is the tool name advertised to the parent. Required.
	Name string

	// Description is the tool description shown to the parent model.
	// It should describe both *what* the subagent is good at and *what
	// shape of task* to pass in. Required.
	Description string

	// Client is the LLM client the subagent uses. Required UNLESS Tiers is
	// set, in which case the DefaultTier's client is the base and Client need
	// not be provided. Typically derived from the parent client via WithModel.
	Client *luft.Client

	// Tiers, when non-empty, lets the parent model choose a model class per
	// call through a "tier" argument advertised in the schema. DefaultTier
	// (required when Tiers is set) names the class used when the caller omits
	// the argument or names an unknown one.
	Tiers       []Tier
	DefaultTier string

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

// tierClient returns the client for the named tier, or nil if not found.
func tierClient(tiers []Tier, name string) *luft.Client {
	for _, t := range tiers {
		if t.Name == name {
			return t.Client
		}
	}
	return nil
}

// New returns a binding for the subagent tool. The schema is {"task": string},
// plus an optional {"tier": enum} when Config.Tiers is set. RequiresConfirmation
// is left false because the subagent's own tools carry their own confirmation
// flags.
func New(cfg Config) (luft.ToolBinding, error) {
	if cfg.Name == "" || cfg.Description == "" || cfg.System == "" {
		return luft.ToolBinding{}, fmt.Errorf("subagent: Name, Description and System are required")
	}
	if cfg.Client == nil && len(cfg.Tiers) == 0 {
		return luft.ToolBinding{}, fmt.Errorf("subagent %q: a Client or Tiers must be provided", cfg.Name)
	}
	if len(cfg.Tiers) > 0 {
		if cfg.DefaultTier == "" {
			return luft.ToolBinding{}, fmt.Errorf("subagent %q: DefaultTier is required when Tiers is set", cfg.Name)
		}
		for _, t := range cfg.Tiers {
			if t.Name == "" || t.Client == nil {
				return luft.ToolBinding{}, fmt.Errorf("subagent %q: every tier needs a Name and Client", cfg.Name)
			}
		}
		if tierClient(cfg.Tiers, cfg.DefaultTier) == nil {
			return luft.ToolBinding{}, fmt.Errorf("subagent %q: DefaultTier %q is not among Tiers", cfg.Name, cfg.DefaultTier)
		}
	}
	type input struct {
		Task string `json:"task"`
		Tier string `json:"tier,omitempty"`
	}
	props := map[string]luft.SchemaProperty{
		"task": {Type: "string", Description: "Self-contained task description for the subagent."},
	}
	if len(cfg.Tiers) > 0 {
		names := make([]any, len(cfg.Tiers))
		for i, t := range cfg.Tiers {
			names[i] = t.Name
		}
		props["tier"] = luft.SchemaProperty{
			Type:        "string",
			Enum:        names,
			Description: tierArgDescription(cfg.Tiers, cfg.DefaultTier),
		}
	}
	tool, fn := luft.NewTypedTool[input](
		cfg.Name,
		cfg.Description,
		luft.InputSchema{
			Type:       "object",
			Properties: props,
			Required:   []string{"task"},
		},
		func(ctx context.Context, in input) (string, error) {
			if in.Task == "" {
				return "", fmt.Errorf("subagent %q: task is empty", cfg.Name)
			}
			// Pick the model: the requested tier, else the default tier, else
			// the plain Client. An unknown tier name falls back to the default
			// rather than erroring — the parent picked a reasonable intent.
			client := cfg.Client
			if len(cfg.Tiers) > 0 {
				name := in.Tier
				if c := tierClient(cfg.Tiers, name); c != nil {
					client = c
				} else if c := tierClient(cfg.Tiers, cfg.DefaultTier); c != nil {
					client = c
				}
			}
			// The subagent runs through the Agent block rather than Loop
			// directly so it gets the same context-trimming behaviour as the
			// top-level agent. With a zero-value Context (MaxTokens == 0) the
			// trim step is a no-op, so research subagents pay nothing for it.
			agent := luft.Agent{
				Client:  client,
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

// tierArgDescription builds the schema description for the optional "tier"
// argument: which classes exist, what each is for, and which is the default.
func tierArgDescription(tiers []Tier, def string) string {
	var b strings.Builder
	b.WriteString("Optional model class for this call. Omit to use the default (")
	b.WriteString(def)
	b.WriteString("). Options: ")
	for i, t := range tiers {
		if i > 0 {
			b.WriteString("; ")
		}
		b.WriteString(t.Name)
		if t.Description != "" {
			b.WriteString(" — ")
			b.WriteString(t.Description)
		}
	}
	b.WriteString(". Escalate to a stronger class for a genuinely hard task; drop to a faster, cheaper one for clearly trivial work; otherwise omit it and take the default.")
	return b.String()
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
