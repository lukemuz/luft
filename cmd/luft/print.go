package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/mcp"
)

// maxStdinBytes caps how much of stdin is read when it's piped into a
// non-interactive run. A git diff is tiny; the cap exists to stop a
// stray `luft < giant.log` from OOMing. The remainder is dropped with a
// truncation marker so the model knows the input was cut.
const maxStdinBytes = 4 << 20 // 4 MiB

// printOpts configures a single non-interactive run.
type printOpts struct {
	json  bool // emit one JSON object on stdout instead of streaming text
	quiet bool // suppress tool-progress lines on stderr
}

// toolCallRec is one entry in the JSON output's tool_calls array. It
// records the model's request (name + raw input), not the tool's output
// (which can be large and is recoverable by re-running).
type toolCallRec struct {
	Name    string          `json:"name"`
	Input   json.RawMessage `json:"input,omitempty"`
	IsError bool            `json:"is_error,omitempty"`
}

// printResult is the shape emitted on stdout in --json mode. error is
// nil on success so scripts can jq `.error == null`.
type printResult struct {
	SessionID string        `json:"session_id"`
	Result    string        `json:"result"`
	Usage     luft.Usage    `json:"usage"`
	ToolCalls []toolCallRec `json:"tool_calls,omitempty"`
	Error     *string       `json:"error"`
	ExitCode  int           `json:"exit_code"`
}

// isNonInteractive decides whether luft should run one-shot instead of
// entering the REPL. Pure (no fd access) so it's unit-testable.
//   - printFlag: the -p value ("" means unset)
//   - nArgs:     flag.NArg() (positional args)
//   - stdinTTY:  whether os.Stdin is a terminal (false when piped)
func isNonInteractive(printFlag string, nArgs int, stdinTTY bool) bool {
	return printFlag != "" || nArgs > 0 || !stdinTTY
}

// readStdin reads up to maxStdinBytes from r. If the cap is hit, a
// truncation marker is appended. Returns "" when r is a terminal (no
// piped content).
func readStdin(r io.Reader, stdinTTY bool) string {
	if stdinTTY {
		return ""
	}
	b, err := io.ReadAll(io.LimitReader(r, maxStdinBytes))
	if err != nil {
		return fmt.Sprintf("\n[stdin read error: %v]", err)
	}
	if len(b) == maxStdinBytes {
		// We got exactly the cap; there may be more. Peek one more byte
		// to detect truncation without reading the whole stream.
		var one [1]byte
		n, _ := r.Read(one[:])
		if n > 0 {
			return string(b) + "\n[stdin truncated at 4 MiB]"
		}
	}
	return string(b)
}

// assemblePrompt builds the user-turn text for a non-interactive run.
//   - instruction: -p flag value, or positional args joined by " ".
//   - stdin:       piped stdin content (already read; "" if a terminal).
//
// Precedence: instruction + stdin → join with a delimited block so the
// model can tell instruction from context. Instruction only → verbatim.
// stdin only → verbatim (the piped content is the whole message).
// neither → error: the caller exits 2.
func assemblePrompt(instruction, stdin string) (string, error) {
	instruction = strings.TrimSpace(instruction)
	stdin = strings.TrimRight(stdin, "\n")
	switch {
	case instruction != "" && stdin != "":
		return instruction + "\n\n--- piped stdin ---\n" + stdin, nil
	case instruction != "":
		return instruction, nil
	case stdin != "":
		return stdin, nil
	default:
		return "", errors.New("no prompt: provide an argument/-p, or pipe input on stdin")
	}
}

// runPrint is the non-interactive counterpart to runTurn: it runs one
// agent turn from a single prompt, writes output according to opts, and
// returns the process exit code. Unlike runTurn it never prompts for
// confirmation (non-interactive forces auto-approve at the caller) and
// persists the turn so -resume works for multi-step scripted workflows.
func (s *session) runPrint(ctx context.Context, input string, opts printOpts) int {
	if s.undo != nil {
		s.undo.beginTurn()
		defer s.undo.endTurn()
	}

	expanded, images, warns := expandAtReferences(input, s.cwd)
	for _, w := range warns {
		fmt.Fprintln(s.errOut, yellow("⚠ ")+w)
	}
	if len(images) > 0 {
		s.history = append(s.history, luft.NewUserMessageWithImages(expanded, images))
	} else {
		s.history = append(s.history, luft.NewUserMessage(expanded))
	}

	turnCtx, cancel := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	// Track tool_use blocks by ID so the results callback can mark
	// is_error and print a result line. Same shape as runTurn's maps,
	// kept local because print mode never shares state with the REPL.
	toolNames := map[string]string{}
	toolInputs := map[string]json.RawMessage{}
	pending := map[string]int{} // tool_use ID → index in calls
	var calls []toolCallRec

	var textBuf strings.Builder // collected text (JSON mode only)

	onDelta := func(b luft.ContentBlock) {
		switch b.Type {
		case luft.TypeText:
			if opts.json {
				textBuf.WriteString(b.Text)
			} else {
				fmt.Fprint(s.out, b.Text)
			}
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
			pending[b.ID] = len(calls)
			calls = append(calls, toolCallRec{Name: b.Name, Input: b.Input})
		}
	}
	onResults := func(results []luft.ToolResult) {
		for _, r := range results {
			if idx, ok := pending[r.ToolUseID]; ok {
				if r.IsError {
					calls[idx].IsError = true
				}
			}
		}
		if opts.quiet {
			return
		}
		for _, r := range results {
			name := toolNames[r.ToolUseID]
			if name == "" {
				name = "tool"
			}
			preview := toolInputPreview(toolInputs[r.ToolUseID])
			s.printToolResult(name, preview, r.IsError, len(r.Content))
		}
	}

	result, err := s.agent.StepStream(turnCtx, s.history, onDelta, onResults)

	// Tokens were spent regardless of how the loop ended; always account.
	s.usage.InputTokens += result.Usage.InputTokens
	s.usage.OutputTokens += result.Usage.OutputTokens
	s.usage.CacheCreationTokens += result.Usage.CacheCreationTokens
	s.usage.CacheReadTokens += result.Usage.CacheReadTokens

	pr := printResult{SessionID: s.sessionID, Usage: s.usage, ToolCalls: calls}

	switch {
	case errors.Is(err, context.Canceled):
		// Interrupted: drop the unfinished turn (consistent with a cancelled
		// runTurn) but keep partial text for JSON output.
		if n := len(s.history); n > 0 {
			s.history = s.history[:n-1]
		}
		pr.Result = textBuf.String()
		msg := "interrupted"
		pr.Error = &msg
		pr.ExitCode = 130
		if !opts.quiet {
			fmt.Fprintln(s.errOut, yellow("⚠ interrupted"))
		}
	case errors.Is(err, luft.ErrMaxIter):
		// Max-iter is a safety pause, not a failure: partial work is kept
		// and persisted so the user/script can resume with -resume <id>.
		s.history = result.Messages
		s.persist(ctx)
		pr.Result = result.FinalText()
		if pr.Result == "" {
			pr.Result = textBuf.String()
		}
		msg := "max-iter"
		pr.Error = &msg
		pr.ExitCode = 0
		if !opts.quiet {
			fmt.Fprintln(s.errOut, yellow("⚠ paused")+grey(fmt.Sprintf(" at the %d model-call limit — progress kept. Resume with -resume %s", s.agent.MaxIter, s.sessionID)))
		}
	case err != nil:
		if n := len(s.history); n > 0 {
			s.history = s.history[:n-1]
		}
		msg := err.Error()
		pr.Error = &msg
		pr.ExitCode = 1
		if !opts.quiet {
			fmt.Fprintln(s.errOut, red("✗ error: ")+err.Error())
		}
	default:
		s.history = result.Messages
		s.persist(ctx)
		pr.Result = result.FinalText()
		if pr.Result == "" {
			pr.Result = textBuf.String()
		}
		pr.ExitCode = 0
	}

	if !opts.json {
		// Trailing newline after streamed (or empty) text so the shell
		// prompt doesn't run on into the output.
		fmt.Fprintln(s.out)
		return pr.ExitCode
	}

	enc := json.NewEncoder(s.out)
	enc.SetEscapeHTML(false) // don't mangle diffs/code with \u003c
	if err := enc.Encode(pr); err != nil {
		// Encoding failure is the one thing that can go wrong here; report
		// it to stderr without corrupting the (already-written) stdout.
		fmt.Fprintf(s.errOut, "%s encode output: %v\n", red("✗"), err)
		return 1
	}
	return pr.ExitCode
}

// cleanup releases resources that main()'s defers would otherwise handle.
// It must be called before os.Exit in the non-interactive path, because
// os.Exit skips deferred functions.
func cleanup(logFile *os.File, closers []io.Closer) {
	if logFile != nil {
		_ = logFile.Close()
	}
	for _, c := range closers {
		_ = c.Close()
	}
}

// toClosers adapts a slice of *mcp.Server (each has Close() error) to
// []io.Closer for cleanup. Exists only because Go slices are not
// covariant over element types.
func toClosers(servers []*mcp.Server) []io.Closer {
	out := make([]io.Closer, len(servers))
	for i, s := range servers {
		out[i] = s
	}
	return out
}
