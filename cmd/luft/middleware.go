package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/lukemuz/luft"
)

// summarizingResultLimit returns middleware that, when a tool's (non-error)
// output exceeds maxBytes, replaces it with a model-generated summary that
// preserves errors, failures, file:line references, and the final status while
// dropping repetitive progress noise. It is intended for high-volume, noisy
// tools (bash build/test logs) where the model needs the gist plus the key
// errors, not every line.
//
// It is deliberately NOT applied to file-reading tools: those return verbatim
// content the model needs intact to write a correct edit, and summarizing them
// would silently corrupt str_replace inputs. On any summarizer error the
// middleware falls back to head+tail truncation rather than failing the call.
func summarizingResultLimit(client *luft.Client, maxBytes int) luft.Middleware {
	return func(b luft.ToolBinding) luft.ToolFunc {
		return func(ctx context.Context, input json.RawMessage) (string, error) {
			out, err := b.Func(ctx, input)
			if err != nil || len(out) <= maxBytes {
				return out, err
			}
			summary, serr := summarizeToolOutput(ctx, client, b.Tool.Name, out)
			if serr != nil {
				return truncateMiddle(out, maxBytes), nil
			}
			return fmt.Sprintf("[compressed from %d bytes; errors and final status preserved verbatim]\n%s", len(out), summary), nil
		}
	}
}

const toolOutputSummarySystem = `You compress verbose command output (build, test, lint, and shell logs) for an AI coding agent that will act on it. Preserve, verbatim where possible: every error and warning message, failed assertions, panics and stack-frame locations (file:line), the failing command, counts (e.g. "3 failed, 12 passed"), and the final pass/fail status. Drop repetitive progress lines, successful-step noise, and duplicated output. Be terse. Output only the compressed log — no preamble, no commentary.`

// summarizeToolOutput sends the full (already byte-capped) tool output to the
// summarizer. The output reaching here is at most the upstream 64 KiB cap, so
// it always fits the summarizer's context comfortably; sending it whole avoids
// missing errors buried in the middle.
func summarizeToolOutput(ctx context.Context, client *luft.Client, toolName, output string) (string, error) {
	prompt := fmt.Sprintf("Tool: %s\nOutput (%d bytes):\n\n%s", toolName, len(output), output)
	reply, _, err := client.Ask(ctx, toolOutputSummarySystem, []luft.Message{luft.NewUserMessage(prompt)})
	if err != nil {
		return "", err
	}
	summary := luft.TextContent(reply)
	if strings.TrimSpace(summary) == "" {
		return "", fmt.Errorf("summarizer returned empty text")
	}
	return summary, nil
}

// truncateMiddle keeps the head and tail of s and elides the middle, used as
// the fallback when summarization fails. It beats head-only truncation for
// command logs, where the final status lives at the tail.
func truncateMiddle(s string, max int) string {
	if len(s) <= max {
		return s
	}
	half := max / 2
	return s[:half] + fmt.Sprintf("\n... [%d bytes elided] ...\n", len(s)-2*half) + s[len(s)-half:]
}

// dedupReads returns middleware that suppresses re-reads. It keys on
// (tool name + exact input) and remembers a hash of the last output. When a
// repeat call produces byte-identical output (file unchanged on disk), it
// returns a short notice instead of the full content, saving the tokens of a
// redundant read. Outputs smaller than minBytes are never deduped — they're
// cheap and not worth the indirection.
//
// Correctness note: the tool is always re-run, so a changed file yields
// different output, a different hash, and the full fresh content. The only
// assumption is that the model still holds its earlier read in context (true
// well under the summarization budget); the notice tells it how to force a
// fresh read if not. The cache is per-construction, so each agent that needs
// isolation must wrap its own copy.
func dedupReads(minBytes int) luft.Middleware {
	var mu sync.Mutex
	seen := map[string]string{}
	return func(b luft.ToolBinding) luft.ToolFunc {
		return func(ctx context.Context, input json.RawMessage) (string, error) {
			out, err := b.Func(ctx, input)
			if err != nil || len(out) < minBytes {
				return out, err
			}
			key := b.Tool.Name + "\x00" + string(input)
			sum := sha256.Sum256([]byte(out))
			h := hex.EncodeToString(sum[:8])
			mu.Lock()
			prev, ok := seen[key]
			seen[key] = h
			mu.Unlock()
			if ok && prev == h {
				return fmt.Sprintf("[identical to your earlier %s call this session (%d bytes, unchanged on disk) — reuse that result. If it's no longer in your context, request a specific line range or a different view to force a fresh read.]", b.Tool.Name, len(out)), nil
			}
			return out, nil
		}
	}
}
