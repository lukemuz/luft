package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/lukemuz/luft"
)

const compactSystemPrompt = `You are summarizing a coding-agent conversation so it can be continued in a fresh context window. The summary REPLACES the earlier turns — the assistant must be able to pick up from your summary alone.

Preserve:
- The user's overall goal and any constraints they stated
- Files inspected and key findings (cite paths and line numbers)
- Files modified and what the change was
- Decisions made and the reasoning behind them
- The current plan: what's done, what's in progress, what's pending
- Outstanding errors, failures, or open questions

Drop:
- Verbose tool outputs (greps, file dumps) — keep only what informed a decision
- Intermediate explorations that didn't pan out
- Redundant restatements of the same fact

Format the output as Markdown with these sections, in this order:
## Goal
## Files touched
## Findings
## Decisions
## Plan status
## Open questions

Be specific and concrete. Cite paths. This summary is the only thing the next assistant turn will see of the past.`

// makeAutoSummarizer builds the ContextManager.Summarizer used for automatic
// in-loop trimming. It reuses the /compact system prompt so manual and
// automatic compaction produce consistent summaries, and renders the trim
// zone with luft.RenderForSummary (tool outputs capped) to keep the call
// cheap. Returning an error aborts the turn rather than silently inserting an
// empty summary.
func makeAutoSummarizer(client *luft.Client) func(context.Context, []luft.Message) (string, error) {
	return func(ctx context.Context, trimmed []luft.Message) (string, error) {
		rendered := luft.RenderForSummary(trimmed, 1500)
		reply, _, err := client.Ask(ctx, compactSystemPrompt,
			[]luft.Message{luft.NewUserMessage(
				"Summarize the following earlier portion of the conversation so it can be dropped from context:\n\n" + rendered)},
		)
		if err != nil {
			return "", fmt.Errorf("auto-compact: %w", err)
		}
		summary := luft.TextContent(reply)
		if strings.TrimSpace(summary) == "" {
			return "", fmt.Errorf("auto-compact: summarizer returned empty text")
		}
		return "[Compacted summary of earlier turns; continue from here.]\n\n" + summary, nil
	}
}

// compact takes the conversation history and produces a new, shorter
// history that preserves recent verbatim turns and replaces older turns
// with a single synthetic summary message.
//
// The cut point is chosen at the start of the (keepRecentTurns)th
// most recent user-typed message — this guarantees we never split in
// the middle of a tool_use / tool_result pair, which would produce an
// invalid request.
//
// extraInstructions, if non-empty, is appended to the summarizer's
// system prompt so the user can steer the summary (e.g. "focus on the
// auth module", "preserve the test failure traces verbatim").
//
// The summary call uses summarizer (typically Haiku) so it's near-free
// even on long histories.
func compact(
	ctx context.Context,
	summarizer *luft.Client,
	history []luft.Message,
	keepRecentTurns int,
	extraInstructions string,
) ([]luft.Message, luft.Usage, error) {
	if keepRecentTurns < 1 {
		keepRecentTurns = 1
	}
	cut := findCutPoint(history, keepRecentTurns)
	if cut <= 0 {
		// Nothing to compact — fewer turns than the keep target.
		return history, luft.Usage{}, nil
	}

	transcript := renderTranscript(history[:cut])
	if strings.TrimSpace(transcript) == "" {
		return history, luft.Usage{}, nil
	}

	system := compactSystemPrompt
	if s := strings.TrimSpace(extraInstructions); s != "" {
		system += "\n\nAdditional user-supplied instructions for this compaction:\n" + s
	}

	userTurn := luft.NewUserMessage("Summarize the following conversation transcript:\n\n" + transcript)
	resp, usage, err := summarizer.Ask(ctx, system, []luft.Message{userTurn})
	if err != nil {
		return history, usage, fmt.Errorf("compact: summarizer: %w", err)
	}

	summary := luft.TextContent(resp)
	if strings.TrimSpace(summary) == "" {
		return history, usage, fmt.Errorf("compact: summarizer returned empty text")
	}

	synthetic := luft.NewUserMessage(
		"[Compacted summary of the earlier conversation; the assistant should continue from here.]\n\n" + summary,
	)
	out := make([]luft.Message, 0, 1+(len(history)-cut))
	out = append(out, synthetic)
	out = append(out, history[cut:]...)
	return out, usage, nil
}

// findCutPoint returns the index of the start of the keepRecent-th most
// recent user-typed message. Returns 0 if there are not enough such
// messages (meaning: nothing to compact).
//
// A "user-typed" message is a Role==user message that contains a text
// block but no tool_result blocks — i.e., the user actually typed it,
// it isn't a synthetic tool-results turn from the loop.
func findCutPoint(history []luft.Message, keepRecent int) int {
	if len(history) == 0 || keepRecent <= 0 {
		return 0
	}
	var userTurnIdx []int
	for i, m := range history {
		if m.Role != luft.RoleUser {
			continue
		}
		isToolResults := false
		hasText := false
		for _, b := range m.Content {
			if b.Type == luft.TypeToolResult {
				isToolResults = true
			}
			if b.Type == luft.TypeText && strings.TrimSpace(b.Text) != "" {
				hasText = true
			}
		}
		if isToolResults || !hasText {
			continue
		}
		userTurnIdx = append(userTurnIdx, i)
	}
	if len(userTurnIdx) <= keepRecent {
		return 0
	}
	return userTurnIdx[len(userTurnIdx)-keepRecent]
}

// renderTranscript flattens a slice of messages into a plain-text
// transcript suitable for feeding to the summarizer. Tool calls and
// results are rendered as compact bracketed annotations rather than
// raw JSON to keep the summary call cheap.
func renderTranscript(history []luft.Message) string {
	var b strings.Builder
	for _, m := range history {
		role := m.Role
		switch role {
		case luft.RoleUser:
			fmt.Fprintf(&b, "\n## user\n")
		case luft.RoleAssistant:
			fmt.Fprintf(&b, "\n## assistant\n")
		default:
			fmt.Fprintf(&b, "\n## %s\n", role)
		}
		for _, blk := range m.Content {
			switch blk.Type {
			case luft.TypeText:
				if t := strings.TrimSpace(blk.Text); t != "" {
					b.WriteString(t)
					b.WriteString("\n")
				}
			case luft.TypeToolUse:
				fmt.Fprintf(&b, "[tool_use %s id=%s input=%s]\n", blk.Name, blk.ID, truncateForTranscript(string(blk.Input), 500))
			case luft.TypeToolResult:
				fmt.Fprintf(&b, "[tool_result id=%s]\n%s\n", blk.ToolUseID, truncateForTranscript(blk.Content, 1500))
			}
		}
	}
	return strings.TrimSpace(b.String())
}

func truncateForTranscript(s string, max int) string {
	if len(s) <= max {
		return s
	}
	keep := max - 80
	if keep < 0 {
		keep = 0
	}
	return s[:keep] + fmt.Sprintf("... [truncated %d bytes]", len(s)-keep)
}
