// Package bash provides a sandboxed shell tool for the coding agent.
//
// The tool exposes a single `bash` tool that runs a command via `sh -c`,
// captures combined stdout+stderr, enforces a per-call timeout, and caps
// the returned output. Three safety modes trade off capability for risk:
//
//   - ModeRestricted (default): the command must match an allowlist of
//     read-only shell built-ins and common dev utilities (ls, cat, grep,
//     find, git status/diff/log, go list/vet/build, etc.). Anything else
//     is rejected with a descriptive error before execution.
//
//   - ModeStandard: an open shell with a deny-list of obviously dangerous
//     patterns (rm -rf /, dd if=, curl|sh, sudo, fork bombs, writing to
//     /etc or /dev/sd*). Bindings are marked RequiresConfirmation=true so
//     callers should pair the toolset with luft.WithConfirmation.
//
//   - ModeUnrestricted: no command-level filtering. Only the timeout and
//     output cap apply. RequiresConfirmation=true.
//
// Modes are deliberately coarse. The deny-list is not a security boundary
// — a determined model can evade any pattern matcher. Treat it as a guard
// rail against accidents and pair higher modes with confirmation, a
// dedicated working directory, or a containerised host.
package bash

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/lukemuz/luft"
)

// Mode controls how strictly commands are filtered before execution.
type Mode int

const (
	// ModeRestricted only allows commands whose first token is in a
	// curated allowlist of read-only utilities. This is the default and
	// is safe to run without confirmation.
	ModeRestricted Mode = iota

	// ModeStandard runs any command not matching the deny-list of
	// dangerous patterns. Bindings carry RequiresConfirmation=true.
	ModeStandard

	// ModeUnrestricted runs any command. Only the timeout and output
	// cap apply. Bindings carry RequiresConfirmation=true.
	ModeUnrestricted
)

const (
	defaultTimeout    = 30 * time.Second
	defaultMaxBytes   = 64 * 1024
	defaultShell      = "/bin/sh"
	allowlistFirstTok = "first token must be one of the allowed read-only commands; switch to ModeStandard for a broader shell"
)

// Config controls Tool behaviour.
type Config struct {
	// Root is the working directory commands run in. If empty, the
	// current process working directory is used. The optional `cwd`
	// argument on a tool call is resolved relative to Root and may not
	// escape it.
	Root string

	// Mode selects the safety policy. Zero value is ModeRestricted.
	Mode Mode

	// Timeout caps each command's wall-clock duration. 0 uses 30s.
	Timeout time.Duration

	// MaxOutputBytes caps how much combined stdout+stderr is returned.
	// Output beyond the cap is replaced with a truncation notice. 0 uses
	// 64 KiB.
	MaxOutputBytes int

	// ExtraAllow adds command names to the ModeRestricted allowlist.
	// Ignored in other modes.
	ExtraAllow []string

	// ExtraDeny adds regular expressions (matched against the full
	// command string) to the ModeStandard deny-list. Ignored in other
	// modes.
	ExtraDeny []string

	// ReportExitError, when true, returns a non-zero exit status or a timeout
	// as a tool error (is_error) rather than ordinary output. The full
	// stdout+stderr and the "[exit=N]" header still ride along in the error
	// message, so the model sees everything — but a failed command is now
	// unmistakable instead of rendering as a success with the exit code buried
	// in the text. Default false keeps every result, pass or fail, as plain
	// output (best for read-only inspection where a non-zero exit, e.g. grep
	// finding nothing, is informational rather than a failure).
	ReportExitError bool
}

// Tool is the bash tool binding.
type Tool struct {
	cfg     Config
	allow   map[string]struct{}
	deny    []*regexp.Regexp
	binding luft.ToolBinding
}

// Default allowlist for ModeRestricted: read-only inspection commands.
var defaultAllow = []string{
	"ls", "pwd", "cat", "head", "tail", "wc", "file", "stat",
	"find", "grep", "rg", "tree",
	"echo", "printf", "true", "false", "test", "which", "type",
	"go", "git", "make", "node", "npm", "python", "python3", "pip",
	"date", "uname", "env",
}

// Default deny patterns for ModeStandard.
var defaultDeny = []string{
	`(^|[^a-zA-Z0-9_])sudo($|\s)`,
	`(^|[^a-zA-Z0-9_])su\s+-`,
	`\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+/(\s|$)`,
	`\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+/\*`,
	`\bmkfs\b`,
	`\bdd\s+.*\bof=/dev/`,
	`>\s*/dev/sd[a-z]`,
	`>\s*/dev/nvme`,
	`>\s*/etc/`,
	`\bchmod\s+-R?\s*[0-7]*7[0-7]*\s+/(\s|$)`,
	`\bchown\s+-R\s+\S+\s+/(\s|$)`,
	`:\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:`, // fork bomb
	`\bcurl\b[^|;&]*\|\s*(sh|bash|zsh)\b`,
	`\bwget\b[^|;&]*\|\s*(sh|bash|zsh)\b`,
	`\bshutdown\b`, `\breboot\b`, `\bhalt\b`, `\bpoweroff\b`,
}

// New constructs a bash Tool from cfg, applying defaults.
func New(cfg Config) (*Tool, error) {
	if cfg.Timeout == 0 {
		cfg.Timeout = defaultTimeout
	}
	if cfg.MaxOutputBytes == 0 {
		cfg.MaxOutputBytes = defaultMaxBytes
	}
	if cfg.Root != "" {
		abs, err := filepath.Abs(cfg.Root)
		if err != nil {
			return nil, fmt.Errorf("bash: resolve root %q: %w", cfg.Root, err)
		}
		cfg.Root = abs
	}

	t := &Tool{cfg: cfg, allow: map[string]struct{}{}}
	for _, name := range defaultAllow {
		t.allow[name] = struct{}{}
	}
	for _, name := range cfg.ExtraAllow {
		t.allow[name] = struct{}{}
	}
	for _, pat := range append(append([]string{}, defaultDeny...), cfg.ExtraDeny...) {
		re, err := regexp.Compile(pat)
		if err != nil {
			return nil, fmt.Errorf("bash: compile deny pattern %q: %w", pat, err)
		}
		t.deny = append(t.deny, re)
	}

	t.binding = t.buildBinding()
	return t, nil
}

// Toolset returns a single-binding toolset suitable for luft.Join.
func (t *Tool) Toolset() luft.Toolset {
	return luft.Tools(t.binding)
}

// TrainedHandler returns a ToolFunc with the input shape Anthropic's
// bash_20250124 trained tool emits ({"command": "...", "restart": bool}).
// Pair with anthropic.BashTool to register a binding the model has been
// post-trained on. The "restart" flag is acknowledged but a no-op — we
// don't maintain a persistent shell session, so each invocation is fresh.
func (t *Tool) TrainedHandler() luft.ToolFunc {
	return func(ctx context.Context, raw json.RawMessage) (string, error) {
		var in struct {
			Command string `json:"command"`
			Restart bool   `json:"restart,omitempty"`
		}
		if len(raw) > 0 {
			if err := json.Unmarshal(raw, &in); err != nil {
				return "", fmt.Errorf("bash: parse trained input: %w", err)
			}
		}
		if in.Restart {
			return "(restart acknowledged; this handler runs each command in a fresh shell)", nil
		}
		return t.run(ctx, bashInput{Command: in.Command})
	}
}

type bashInput struct {
	Command string `json:"command"`
	Cwd     string `json:"cwd,omitempty"`
}

func (t *Tool) buildBinding() luft.ToolBinding {
	desc := bashDescription(t.cfg.Mode, t.cfg.Timeout)
	tool, fn := luft.NewTypedTool(
		"bash",
		desc,
		luft.InputSchema{
			Type: "object",
			Properties: map[string]luft.SchemaProperty{
				"command": {Type: "string", Description: "Shell command to execute via /bin/sh -c."},
				"cwd":     {Type: "string", Description: "Optional working directory, relative to the workspace root."},
			},
			Required: []string{"command"},
		},
		t.run,
	)
	requires := t.cfg.Mode != ModeRestricted
	return luft.ToolBinding{Tool: tool, Func: fn, Meta: luft.ToolMetadata{RequiresConfirmation: requires}}
}

func bashDescription(mode Mode, timeout time.Duration) string {
	switch mode {
	case ModeRestricted:
		return fmt.Sprintf("Run a read-only shell command (%s timeout). Allowed: ls, cat, grep, rg, find, git status/diff/log, go list/vet, wc, head, tail, etc. Returns combined stdout+stderr.", timeout)
	case ModeStandard:
		return fmt.Sprintf("Run a shell command via /bin/sh -c (%s timeout). Combined stdout+stderr is returned. A small deny-list rejects obviously dangerous patterns (rm -rf /, sudo, curl|sh, etc.).", timeout)
	case ModeUnrestricted:
		return fmt.Sprintf("Run an arbitrary shell command via /bin/sh -c (%s timeout). Combined stdout+stderr is returned.", timeout)
	}
	return "Run a shell command."
}

func (t *Tool) run(ctx context.Context, in bashInput) (string, error) {
	cmd := strings.TrimSpace(in.Command)
	if cmd == "" {
		return "", fmt.Errorf("command is required")
	}
	if err := t.check(cmd); err != nil {
		return "", err
	}

	cwd, err := t.resolveCwd(in.Cwd)
	if err != nil {
		return "", err
	}

	cctx, cancel := context.WithTimeout(ctx, t.cfg.Timeout)
	defer cancel()

	c := exec.CommandContext(cctx, defaultShell, "-c", cmd)
	c.Dir = cwd
	var buf bytes.Buffer
	c.Stdout = &buf
	c.Stderr = &buf

	start := time.Now()
	runErr := c.Run()
	elapsed := time.Since(start)

	out := truncate(buf.Bytes(), t.cfg.MaxOutputBytes)
	exit := 0
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			exit = ee.ExitCode()
		} else {
			exit = -1
		}
	}

	timedOut := cctx.Err() == context.DeadlineExceeded
	header := fmt.Sprintf("$ %s\n[exit=%d duration=%s", cmd, exit, elapsed.Round(time.Millisecond))
	if timedOut {
		header += " timeout=true"
	}
	header += "]\n"
	result := header + out

	// In strict mode a failure is surfaced as a tool error so the UI marks it
	// failed and the model can't overlook a buried exit code. The full output
	// rides along in the message, so nothing is hidden.
	if t.cfg.ReportExitError && (exit != 0 || timedOut) {
		return "", fmt.Errorf("%s", strings.TrimRight(result, "\n"))
	}
	return result, nil
}

func (t *Tool) check(cmd string) error {
	switch t.cfg.Mode {
	case ModeRestricted:
		first := firstToken(cmd)
		if first == "" {
			return fmt.Errorf("could not parse command")
		}
		if _, ok := t.allow[first]; !ok {
			return fmt.Errorf("command %q is not on the read-only allowlist (%s)", first, allowlistFirstTok)
		}
		// Reject shell metacharacters that smuggle in second commands.
		if strings.ContainsAny(cmd, ";&|`$") || strings.Contains(cmd, ">") || strings.Contains(cmd, "<") {
			return fmt.Errorf("shell metacharacters (;&|`$<>) are not permitted in restricted mode")
		}
		return nil
	case ModeStandard:
		for _, re := range t.deny {
			if re.MatchString(cmd) {
				return fmt.Errorf("command rejected by deny-list pattern %q", re.String())
			}
		}
		return nil
	case ModeUnrestricted:
		return nil
	}
	return fmt.Errorf("unknown bash mode %d", t.cfg.Mode)
}

func (t *Tool) resolveCwd(cwd string) (string, error) {
	root := t.cfg.Root
	if cwd == "" {
		return root, nil
	}
	if root == "" {
		return cwd, nil
	}
	abs := cwd
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(root, cwd)
	}
	abs = filepath.Clean(abs)
	rel, err := filepath.Rel(root, abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("cwd %q escapes workspace root", cwd)
	}
	return abs, nil
}

func firstToken(cmd string) string {
	// Skip leading env assignments like FOO=bar BAR=baz cmd ...
	fields := strings.Fields(cmd)
	for _, f := range fields {
		if strings.Contains(f, "=") && !strings.HasPrefix(f, "=") {
			continue
		}
		return filepath.Base(f)
	}
	return ""
}

func truncate(b []byte, max int) string {
	if len(b) <= max {
		return string(b)
	}
	keep := max - 64
	if keep < 0 {
		keep = 0
	}
	return string(b[:keep]) + fmt.Sprintf("\n... [truncated %d bytes of output]", len(b)-keep)
}

// Ensure io import stays used even if we drop streaming later.
var _ = io.Discard
