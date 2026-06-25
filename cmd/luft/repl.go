package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/lukemuz/luft"
)

// session is the mutable state owned by the REPL. It bundles the
// running history, accumulated usage, and the auxiliary clients we
// need for slash commands like :compact and :model.
type session struct {
	agent      luft.Agent
	summarizer *luft.Client // typically Haiku, used by :compact
	provider   luft.Provider
	memory     string   // loaded project memory, for :memory
	sp         *spinner // shared activity indicator; the confirmer suspends it while prompting

	history []luft.Message
	usage   luft.Usage // accumulated across the session
	logPath string     // empty if no JSONL log is active
}

func (s *session) repl(ctx context.Context) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for {
		fmt.Fprint(os.Stderr, "\n"+boldCyan("❯")+" ")
		if !scanner.Scan() {
			fmt.Fprintln(os.Stderr)
			return
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "/") || strings.HasPrefix(line, ":") {
			if quit := s.runCommand(ctx, line); quit {
				return
			}
			continue
		}

		s.runTurn(ctx, line)
	}
}

func (s *session) runTurn(ctx context.Context, input string) {
	s.history = append(s.history, luft.NewUserMessage(input))

	turnCtx, cancel := signal.NotifyContext(ctx, syscall.SIGINT)
	defer cancel()

	// Track tool_use blocks as they stream so we can show the tool name
	// (and a useful preview of its arguments) when the result lands.
	toolNames := map[string]string{}
	toolInputs := map[string]json.RawMessage{}

	// Spinner with two-mode label:
	//   - "thinking…"     between turns / iterations (model latency)
	//   - "running tools…" when deltas pause for >500ms within a turn
	//                      (proxy signal that tool execution is in flight)
	// Shared with the confirmer, which suspends it while prompting so the
	// repaint loop can't wipe the approval prompt off the screen.
	sp := s.sp
	defer sp.Stop()

	var idleMu sync.Mutex
	var idleTimer *time.Timer
	armIdle := func() {
		idleMu.Lock()
		defer idleMu.Unlock()
		if idleTimer != nil {
			idleTimer.Stop()
		}
		idleTimer = time.AfterFunc(500*time.Millisecond, func() {
			sp.Start("running tools…")
		})
	}
	disarmIdle := func() {
		idleMu.Lock()
		defer idleMu.Unlock()
		if idleTimer != nil {
			idleTimer.Stop()
			idleTimer = nil
		}
	}

	// Start the spinner immediately — first model latency is the most
	// common "is it doing anything?" moment.
	sp.Start("thinking…")
	// And restart it before each subsequent iteration's model call.
	s.agent.Hooks.OnIteration = func(ctx context.Context, iter int, history []luft.Message) {
		if iter == 0 {
			return // already started above
		}
		sp.Start("thinking…")
	}
	defer func() { s.agent.Hooks.OnIteration = nil }()

	result, err := s.agent.StepStream(turnCtx, s.history,
		func(b luft.ContentBlock) {
			sp.Stop()
			armIdle()
			switch b.Type {
			case luft.TypeText:
				fmt.Print(b.Text)
			case luft.TypeToolUse:
				if b.ID == "" {
					return
				}
				if b.Name != "" {
					toolNames[b.ID] = b.Name
				}
				if len(b.Input) > 0 {
					toolInputs[b.ID] = b.Input
				}
			}
		},
		func(results []luft.ToolResult) {
			disarmIdle()
			sp.Stop()
			fmt.Fprintln(os.Stderr)
			for _, r := range results {
				name := toolNames[r.ToolUseID]
				if name == "" {
					name = "tool"
				}
				preview := toolInputPreview(toolInputs[r.ToolUseID])
				printToolResult(name, preview, r.IsError, len(r.Content))
			}
		},
	)
	disarmIdle()
	sp.Stop()
	fmt.Println()

	if err != nil {
		fmt.Fprintln(os.Stderr, red("✗ error: ")+err.Error())
		if n := len(s.history); n > 0 {
			s.history = s.history[:n-1]
		}
		return
	}

	s.history = result.Messages
	s.usage.InputTokens += result.Usage.InputTokens
	s.usage.OutputTokens += result.Usage.OutputTokens
	s.usage.CacheCreationTokens += result.Usage.CacheCreationTokens
	s.usage.CacheReadTokens += result.Usage.CacheReadTokens
}

// runCommand dispatches a slash command. Returns true if the REPL
// should exit. Both `/cmd` and `:cmd` styles are accepted.
func (s *session) runCommand(ctx context.Context, line string) bool {
	line = strings.TrimLeft(line, "/:")
	parts := strings.SplitN(line, " ", 2)
	cmd := parts[0]
	args := ""
	if len(parts) == 2 {
		args = strings.TrimSpace(parts[1])
	}

	switch cmd {
	case "exit", "quit":
		return true
	case "help", "?":
		s.printHelp()
	case "reset", "clear":
		s.history = nil
		fmt.Fprintln(os.Stderr, "(history cleared)")
	case "compact":
		s.doCompact(ctx, args)
	case "tokens":
		s.printTokens()
	case "memory":
		s.printMemory()
	case "tools":
		s.printTools()
	case "model":
		s.changeModel(args)
	case "log":
		if s.logPath == "" {
			fmt.Fprintln(os.Stderr, "(session logging is off — start with -log auto or -log <path>)")
		} else {
			fmt.Fprintln(os.Stderr, s.logPath)
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown command: /%s (try /help)\n", cmd)
	}
	return false
}

func (s *session) printHelp() {
	rows := [][2]string{
		{"/help", "show this list"},
		{"/exit | /quit", "leave"},
		{"/reset | /clear", "clear conversation history"},
		{"/compact [instructions]", "summarize older turns to free context (cache-resetting)"},
		{"/tokens", "print accumulated token usage and cache stats"},
		{"/memory", "print the loaded project memory (AGENTS.md / CLAUDE.md)"},
		{"/tools", "list the tools currently available to the agent"},
		{"/model <id>", "switch the main-agent model (e.g. anthropic/claude-opus-4.7)"},
		{"/log", "print the active JSONL log path (if any)"},
	}
	for _, r := range rows {
		fmt.Fprintf(os.Stderr, "  %s  %s\n", cyan(padRight(r[0], 24)), grey(r[1]))
	}
}

func padRight(s string, n int) string {
	if len(s) >= n {
		return s
	}
	return s + strings.Repeat(" ", n-len(s))
}

func (s *session) doCompact(ctx context.Context, instructions string) {
	if len(s.history) < 2 {
		fmt.Fprintln(os.Stderr, "(nothing to compact)")
		return
	}
	before := len(s.history)
	sp := s.sp
	sp.Start("compacting…")
	out, usage, err := compact(ctx, s.summarizer, s.history, 4, instructions)
	sp.Stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, red("✗ compact failed: ")+err.Error())
		return
	}
	s.history = out
	s.usage.InputTokens += usage.InputTokens
	s.usage.OutputTokens += usage.OutputTokens
	s.usage.CacheCreationTokens += usage.CacheCreationTokens
	s.usage.CacheReadTokens += usage.CacheReadTokens

	fmt.Fprintf(os.Stderr, "%s %s → %s messages %s\n",
		green("✓ compacted"),
		bold(fmt.Sprintf("%d", before)),
		bold(fmt.Sprintf("%d", len(s.history))),
		grey(fmt.Sprintf("(summarizer used %d in / %d out tokens)", usage.InputTokens, usage.OutputTokens)),
	)
}

func (s *session) printTokens() {
	u := s.usage
	total := u.InputTokens + u.OutputTokens
	row := func(label string, value string, hint string) {
		fmt.Fprintf(os.Stderr, "  %s %s %s\n",
			grey(padRight(label, 16)),
			bold(value),
			grey(hint),
		)
	}
	fmt.Fprintln(os.Stderr, bold("tokens this session"))
	row("input", fmt.Sprintf("%d", u.InputTokens), "")
	row("output", fmt.Sprintf("%d", u.OutputTokens), "")
	row("cache reads", fmt.Sprintf("%d", u.CacheReadTokens), "")
	row("cache writes", fmt.Sprintf("%d", u.CacheCreationTokens), "")
	row("total billable", fmt.Sprintf("%d", total), "(cache reads billed at ~10%)")
	if u.InputTokens > 0 {
		ratio := float64(u.CacheReadTokens) / float64(u.InputTokens+u.CacheReadTokens) * 100
		row("cache hit rate", fmt.Sprintf("%.1f%%", ratio), "")
	}
}

func (s *session) printMemory() {
	if s.memory == "" {
		fmt.Fprintln(os.Stderr, "(no project memory loaded — drop an AGENTS.md or CLAUDE.md in the workspace root)")
		return
	}
	fmt.Fprintln(os.Stderr, s.memory)
}

func (s *session) printTools() {
	for _, b := range s.agent.Tools.Bindings {
		flag := ""
		if b.Meta.RequiresConfirmation {
			flag = " " + yellow("[confirm]")
		}
		fmt.Fprintf(os.Stderr, "  %s%s\n", cyan(b.Tool.Name), flag)
	}
}

func (s *session) changeModel(model string) {
	if model == "" {
		fmt.Fprintln(os.Stderr, dim("usage: /model <model-id>"))
		return
	}
	s.agent.Client = s.agent.Client.WithModel(model)
	fmt.Fprintf(os.Stderr, "%s main-agent model set to %s %s\n",
		green("✓"),
		bold(model),
		grey("(subagent models unchanged)"),
	)
}

// printToolResult emits one line for a completed tool call:
//
//	✓ tool_name  preview…                       1.2 KiB
//	✗ tool_name  preview…                       error
func printToolResult(name, preview string, isError bool, contentLen int) {
	mark := green("✓")
	if isError {
		mark = red("✗")
	}
	left := fmt.Sprintf(" %s %s", mark, cyan(name))
	if preview != "" {
		left += "  " + dim(preview)
	}
	right := humanBytes(contentLen)
	if isError {
		right = red("error")
	}
	fmt.Fprintf(os.Stderr, "%s  %s\n", left, grey(right))
}

// toolInputPreview returns a short, single-line preview of the most
// useful argument the model passed. It tries a small priority list of
// well-known field names; falls back to the raw JSON when nothing
// matches. Always single-line and length-capped.
func toolInputPreview(input json.RawMessage) string {
	if len(input) == 0 {
		return ""
	}
	var m map[string]any
	if err := json.Unmarshal(input, &m); err == nil {
		// A subagent call with an explicit tier shows it, so an escalation is
		// visible on the result line (e.g. "implement  tier=powerful · task=…").
		var prefix string
		if v, ok := m["tier"].(string); ok && v != "" {
			prefix = "tier=" + v + " · "
		}
		// Priority order: prefer the field that's most informative
		// per tool family. First non-empty string wins.
		for _, k := range []string{"command", "path", "file_path", "url", "pattern", "query", "task", "description", "name", "old_str"} {
			if v, ok := m[k]; ok {
				if vs, ok := v.(string); ok && vs != "" {
					return truncate(oneLine(prefix+fmt.Sprintf("%s=%s", k, vs)), 80)
				}
			}
		}
		if prefix != "" {
			return truncate(oneLine(strings.TrimSuffix(prefix, " · ")), 80)
		}
		// Generic fallback: list the first few keys.
		var keys []string
		for k := range m {
			keys = append(keys, k)
		}
		if len(keys) > 0 {
			return dim(fmt.Sprintf("(%s)", strings.Join(keys, ", ")))
		}
	}
	return truncate(oneLine(string(input)), 80)
}
