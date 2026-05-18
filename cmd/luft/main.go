// luft is a CLI coding agent built on the luft toolkit.
//
// Topology (Phase 2):
//
//	main agent           (x-ai/grok-4.3 by default — configurable via -model)
//	  ├── direct tools   workspace read-only + trained bash + trained editor
//	  │                  + todo + clock + batch
//	  ├── explore        subagent on openai/gpt-oss-120b (-explore-model) —
//	  │                  workspace read-only + restricted bash + batch + clock.
//	  │                  Used for cheap, parallelisable inspection; its
//	  │                  iteration history never enters the main context.
//	  └── plan           subagent on x-ai/grok-4.3 (-plan-model) — read-only
//	                     tools, no shell or edits. Used when the main agent
//	                     wants a focused reasoner for design or hard debugging.
//
// The batch tool is also offered to the main agent so it can run several
// reads/searches concurrently in a single turn without paying for a
// subagent loop.
//
// Usage:
//
//	export OPENROUTER_API_KEY=sk-or-...
//	cd ~/your-project && luft
//
// The agent is sandboxed to the current working directory by default.
// Pass -dir to operate on a different directory.
//
// Models default to x-ai/grok-4.3 for the main agent and plan subagent,
// and openai/gpt-oss-120b for the explore subagent. Override with
// -model / -explore-model / -plan-model to use any OpenRouter-supported
// model id.
//
// Flags:
//
//	-dir            working directory the agent is sandboxed to (default cwd)
//	-model          main-agent model id (default x-ai/grok-4.3)
//	-explore-model  model used for the explore subagent (default openai/gpt-oss-120b)
//	-plan-model     model used for the plan subagent (default x-ai/grok-4.3)
//	-no-subagents   disable the explore and plan subagent tools
//	-bash           bash safety mode: restricted | standard | unrestricted
//	-yes            auto-approve every confirmation prompt
//	-max-iter       max model calls per turn (default 30)
//	-version        print version and exit
//
// REPL commands:
//
//	:exit / :quit          leave
//	:reset                 clear conversation history
//	:tokens                print accumulated token usage
//	:help                  show this list
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/providers/openrouter"
	"github.com/lukemuz/luft/tools/bash"
	"github.com/lukemuz/luft/tools/batch"
	"github.com/lukemuz/luft/tools/clock"
	"github.com/lukemuz/luft/tools/editor"
	"github.com/lukemuz/luft/tools/subagent"
	"github.com/lukemuz/luft/tools/todo"
	"github.com/lukemuz/luft/tools/web"
	"github.com/lukemuz/luft/tools/workspace"
)

// version is the build-time version of the luft CLI. The default tracks
// the most recent tagged release; release builds can override it with
// `go build -ldflags "-X main.version=vX.Y.Z"`.
var version = "v0.1.2"

const mainSystemPrompt = `You are luft, a fast and economical CLI coding assistant built on the luft toolkit. You operate inside a workspace directory and help the user with software engineering tasks — reading code, making edits, running commands, and reasoning about changes.

Available tools:
- list_directory, Glob, Grep, read_file, file_info: read-only filesystem inspection
- str_replace_based_edit_tool: view/create/str_replace/insert against files
- bash: run shell commands (safety policy varies by configuration)
- todo_write, todo_read: maintain a short planning checklist for multi-step work
- batch: run 2+ independent read-only calls in one turn. Default for independent reads/greps/inspections; skip only when a later call depends on an earlier result.
- web_fetch (when available): download an http(s) URL and return its content as text. HTML is converted to a plain-text approximation; long pages paginate via max_length + start_index. Use this for documentation lookups and inspecting URLs from error messages.
- explore (when available): bounded research specialist on a fast, cheap model — repo inspection or well-scoped Q&A (provided context, fetched docs, or its own knowledge); returns a concise summary with file dumps and fetches kept out of your context
- plan (when available): design and architecture specialist on a stronger model — get a structured plan before non-trivial multi-file edits, interface changes, or when stuck on a hypothesis
- now: current time

# Working effectively

1. Default to the explore subagent for bounded research — repo inspection (understand a module, find all usages, audit a pattern) and well-scoped Q&A (look up a stdlib function, summarise an RFC, answer a factual question with provided context). It's cheap, its file dumps and fetches stay out of your context, and you receive only its summary.
2. For tight, surgical lookups (one file, one symbol), call read_file or Grep directly.
3. Default to batch for independent read-only work. Each tool call is a full LLM round trip, so if you'd otherwise issue 2+ reads/greps/inspections that don't depend on each other, batch them. Issue solo calls only when a later call's input depends on an earlier call's output (e.g. grep first, then read only the files it returned).
4. Call plan as a routine first step for substantive design work — non-trivial multi-file changes, interface or data-shape changes, decisions with multiple plausible tradeoffs, or stuck debugging (2+ turns without a clear hypothesis). Pass the question with the context you've gathered. Skip plan for routine single-file changes or when your approach is already clear.
5. Brief subagents like a colleague who just walked in: state the goal, the relevant context you've already gathered (file paths, line numbers, error messages, what you've ruled out), and what shape of answer you need. Terse one-line prompts produce shallow, generic work.
6. For multi-step tasks, call todo_write at the start and update it as you go. Keep at most one item in_progress.

# Writing code

7. Match the conventions of the surrounding code — naming, error handling, imports, file layout. Read a neighbouring file before introducing a new pattern.
8. Don't add features, refactor, or introduce abstractions beyond what the task requires. A bug fix doesn't need surrounding cleanup. Three similar lines is better than a premature abstraction. No half-finished scaffolding for hypothetical future requirements.
9. Don't add error handling, validation, or fallbacks for cases that can't happen. Trust internal code and framework guarantees. Validate only at system boundaries — user input, external APIs, untrusted data.
10. Don't add backwards-compatibility shims, feature flags, or "removed" marker comments for code you can just change. If something is unused, delete it.
11. Default to writing no comments. Only add one when the *why* is non-obvious — a hidden constraint, a subtle invariant, a workaround for a specific bug. Don't explain *what* the code does; well-named identifiers do that. Don't reference the current task ("added for X", "used by Y") — that belongs in commit messages and rots as the code evolves.
12. Watch for security issues: command injection, path traversal, SQL injection, XSS, and the rest of the OWASP top 10. If you notice you've written insecure code, fix it immediately.

# Verifying your work

13. After making edits, verify with appropriate checks: build, type-check, run affected tests via bash. Don't trust an edit you haven't checked.
14. Type-checking and tests verify code correctness, not feature correctness. If you can't actually exercise the feature (a UI change you can't see, a side effect you can't observe in this environment), say so explicitly rather than claiming success.
15. Be honest about uncertainty and about what you didn't verify. Don't claim a fix works when you've only inspected it. If the same approach has failed twice, stop and reconsider — call plan, ask the user, or name what you don't understand — rather than trying a third variation.

# Communication

16. State intent in one sentence before the first tool call of a turn. While working, give one-sentence updates only at key moments — when you find something, when you change direction, or when you hit a blocker. Don't narrate deliberation.
17. If a request is genuinely ambiguous in a way that changes the answer, ask one clarifying question before proceeding. Don't ask when context makes the intent clear.
18. Reference code with the path:line format (e.g. main.go:42) so the user can navigate to it directly. When an answer is based on a web_fetch result, cite the URL so the user can verify.
19. End-of-turn summary: one or two sentences on what changed and what's next. Nothing else. For simple questions, just answer — no headers, no bullets.

# Acting with care

20. Local reversible actions (edits, reads, tests) are free — just do them. Pause and confirm for risky actions:
   - Destructive ops: deleting files or branches, dropping tables, killing processes, rm -rf, overwriting uncommitted changes.
   - Hard-to-reverse ops: force-pushing, git reset --hard, amending published commits, removing or downgrading dependencies.
   - Shared-state ops: pushing, opening or closing PRs, posting messages, modifying CI/CD pipelines.
21. Never use a destructive shortcut to bypass a failing check — no --no-verify, no skipping hooks, no force-pushing past a conflict. Fix the underlying issue.
22. When a tool call is denied — by the user or by safety policy — don't retry it with a workaround. Read the denial as signal: either the action isn't wanted, or it isn't allowed. Ask the user how to proceed, or do the next-best non-destructive thing.
23. When asked to commit, prefer creating new commits over amending. Never force-push to main. If a pre-commit hook fails, the commit didn't happen — fix the issue and make a new commit; --amend would target the previous commit and could destroy work.
24. If you encounter unexpected state (unfamiliar files, branches, locks, uncommitted changes), investigate before deleting or overwriting. It may be the user's in-progress work. A user approving an action once doesn't authorize it for all contexts.`

const exploreSystemPrompt = `You are luft's explore specialist — a fast, focused researcher.

You answer self-contained, well-bounded questions and return concise, factual summaries. Two common shapes:
- Repo research: inspect the codebase to find callers, audit a pattern, summarise a module, locate references, etc.
- Bounded Q&A: answer a specific question using context the orchestrator provides, web docs you fetch, or your own knowledge — whichever fits.

You have read-only filesystem tools, restricted bash for read-only commands, web_fetch (when available) for documentation and external references, and a batch tool to fan out independent calls.

Operating principles:
1. Plan briefly, then execute. Default to batch for independent reads/greps/fetches — each tool call is a full LLM round trip, so issue solo calls only when later input depends on earlier output.
2. Cite specific files and line numbers (or URLs) in your findings.
3. Do NOT speculate about anything you have not directly verified. If your own knowledge is the source, say so explicitly so the orchestrator can judge confidence.
4. Keep your final summary tight — it's the only thing the orchestrator sees. Aim for the smallest answer that fully resolves the task.
5. Do not edit files. You have no write access. Refuse if asked.`

const planSystemPrompt = `You are luft's plan specialist — a careful reasoner backed by a strong model.

You receive a design, implementation-planning, or debugging question along with relevant context the orchestrator has gathered. You have read-only filesystem tools to verify specifics, but no shell and no edits.

Operating principles:
1. Think carefully. Cover trade-offs, edge cases, and likely failure modes.
2. Verify with read_file or Grep rather than guessing when a fact is in doubt.
3. Return a structured plan: numbered steps, files to touch, risks. Keep it implementable, not aspirational.
4. Be honest about what you don't know.`

func main() {
	dir := flag.String("dir", ".", "working directory the agent is sandboxed to (defaults to the current directory)")
	model := flag.String("model", envOr("LUFT_MODEL", "x-ai/grok-4.3"), "main-agent model id (any OpenRouter slug; env: LUFT_MODEL)")
	exploreModel := flag.String("explore-model", envOr("LUFT_EXPLORE_MODEL", "openai/gpt-oss-120b"), "model id for the explore subagent (env: LUFT_EXPLORE_MODEL)")
	planModel := flag.String("plan-model", envOr("LUFT_PLAN_MODEL", "x-ai/grok-4.3"), "model id for the plan subagent (env: LUFT_PLAN_MODEL)")
	noSubagents := flag.Bool("no-subagents", false, "disable explore and plan subagent tools")
	noFetch := flag.Bool("no-fetch", false, "disable the native web_fetch tool")
	bashMode := flag.String("bash", "restricted", "bash safety mode: restricted | standard | unrestricted")
	autoYes := flag.Bool("yes", false, "auto-approve every confirmation prompt")
	maxIter := flag.Int("max-iter", 30, "max model calls per turn")
	logPath := flag.String("log", "", "JSONL session log path. Pass `auto` to write under ~/.config/luft/sessions/")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	mode, err := parseBashMode(*bashMode)
	if err != nil {
		log.Fatal(err)
	}

	provider, err := openrouter.NewProviderFromEnv()
	if err != nil {
		log.Fatalf("openrouter provider: %v", err)
	}

	mainClient := mustClient(provider, *model)

	// Optional JSONL session log. The recorder is attached to mainClient
	// before any WithModel-derived clients (summarizer, explore, plan) are
	// created — WithModel preserves the recorder, so all loops in the
	// session log to the same file.
	var logFile *os.File
	resolvedLog := ""
	if *logPath != "" {
		path, err := resolveLogPath(*logPath)
		if err != nil {
			log.Fatalf("log path: %v", err)
		}
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			log.Fatalf("open log: %v", err)
		}
		logFile = f
		resolvedLog = path
		mainClient = mainClient.WithRecorder(luft.NewJSONLRecorder(f))
	}
	defer func() {
		if logFile != nil {
			_ = logFile.Close()
		}
	}()

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
	confirm := makeConfirmer(*autoYes)

	// --- shared building blocks --------------------------------------------

	ws, err := workspace.NewReadOnly(workspace.Config{Root: *dir})
	if err != nil {
		log.Fatal(err)
	}
	clk := clock.New()

	// Read-only middleware stack: timeout, output cap, logging. No
	// confirmation needed; these tools cannot mutate state.
	roMiddleware := []luft.Middleware{
		luft.WithTimeout(60 * time.Second),
		luft.WithResultLimit(64 * 1024),
		luft.WithLogging(logger),
	}
	roTools := luft.MustJoin(ws.Toolset(), clk.Toolset()).Wrap(roMiddleware...)

	// Restricted bash for subagents: read-only commands only, no
	// confirmation needed.
	subBashTool, err := bash.New(bash.Config{Root: *dir, Mode: bash.ModeRestricted})
	if err != nil {
		log.Fatal(err)
	}
	subBashToolset := subBashTool.Toolset().Wrap(roMiddleware...)

	// Web tools (web_fetch) are constructed up here so subagents can
	// include them and so batch can fan out concurrent fetches. Empty
	// when --no-fetch is set; an empty toolset contributes no bindings.
	var webTools luft.Toolset
	if !*noFetch {
		webTools = web.New(web.Config{}).Toolset().Wrap(
			luft.WithTimeout(30*time.Second),
			luft.WithResultLimit(64*1024),
			luft.WithLogging(logger),
		)
	}

	// Batch tool for read-only fan-out. Built from already-wrapped read-only
	// bindings so each sub-call inherits the timeout/limit/logging stack.
	roBatchBinding := batch.New(batch.Config{
		Bindings:    append(append(append([]luft.ToolBinding{}, roTools.Bindings...), subBashToolset.Bindings...), webTools.Bindings...),
		MaxParallel: 8,
	})

	// --- subagent tools ----------------------------------------------------

	var subagentBindings []luft.ToolBinding
	if !*noSubagents {
		exploreClient := mainClient.WithModel(*exploreModel)
		exploreTools := luft.MustJoin(roTools, subBashToolset, webTools, luft.Tools(roBatchBinding)).
			CacheLast(luft.Ephemeral())
		exploreBinding, err := subagent.New(subagent.Config{
			Name:        "explore",
			Description: "Delegate a bounded research task to a fast, cheap specialist. Two main shapes: (a) repo research — find callers, audit a pattern, summarise a module, locate references; (b) bounded Q&A — answer a well-scoped question from provided context, fetched docs, or general knowledge (e.g. 'what does this stdlib function do', 'summarise this RFC's caching rules'). The specialist has read-only filesystem tools, restricted bash, web_fetch, and batch fan-out; it returns a concise summary and its iteration history stays out of your context. Pass a self-contained task description with any context the specialist needs. Default for any task that would otherwise have you read 3+ files or research a well-scoped question yourself.",
			Client:      exploreClient,
			System:      exploreSystemPrompt,
			Tools:       exploreTools,
			MaxIter:     50,
		})
		if err != nil {
			log.Fatal(err)
		}

		planClient := mainClient.WithModel(*planModel)
		planBinding, err := subagent.New(subagent.Config{
			Name:        "plan",
			Description: "Delegate design and architecture work to a stronger reasoning model (better at multi-step architectural reasoning and invariant tracking across components). Get a structured plan before editing when work touches 3+ files, changes a public interface or shared data shape, has multiple plausible tradeoffs, or you've spent 2+ debugging turns without a clear hypothesis. Pass the question PLUS context you've gathered (file excerpts, error messages, prior attempts). Returns a numbered plan with files to touch and risks. Skip for routine single-file changes or when your approach is already clear.",
			Client:      planClient,
			System:      planSystemPrompt,
			Tools:       roTools.CacheLast(luft.Ephemeral()),
			MaxIter:     25,
		})
		if err != nil {
			log.Fatal(err)
		}
		subagentBindings = []luft.ToolBinding{exploreBinding, planBinding}
	}

	// --- main-agent edit tools ---------------------------------------------

	mainBashTool, err := bash.New(bash.Config{Root: *dir, Mode: mode})
	if err != nil {
		log.Fatal(err)
	}
	mainBashBindings := mainBashTool.Toolset().Bindings

	ed, err := editor.New(editor.Config{Root: *dir})
	if err != nil {
		log.Fatal(err)
	}
	editorBindings := ed.Toolset().Bindings

	editTools := luft.Tools(append(mainBashBindings, editorBindings...)...).Wrap(
		luft.WithConfirmation(confirm),
		luft.WithTimeout(60*time.Second),
		luft.WithResultLimit(64*1024),
		luft.WithLogging(logger),
	)

	// --- main agent assembly ----------------------------------------------

	mainTools := luft.MustJoin(
		roTools,
		luft.Tools(roBatchBinding),
		editTools,
		todo.New().Toolset(),
		webTools,
		luft.Tools(subagentBindings...),
	).CacheLast(luft.Ephemeral()) // cache the entire tool block — stable per session

	memory := loadProjectMemory(*dir)
	system := mainSystemPrompt
	if memory != "" {
		system += "\n\n## Project memory\n\n" + memory
	}

	agent := luft.Agent{
		Client: mainClient,
		System: system,
		Tools:  mainTools,
		Context: luft.ContextManager{
			MaxTokens:  120_000,
			KeepFirst:  1,
			KeepRecent: 30,
		},
		MaxIter: *maxIter,
	}

	// Summarizer for /compact defaults to grok-4.3 (the main-tier model);
	// override via LUFT_SUMMARIZE_MODEL to point it at a cheaper slug.
	summarizer := mainClient.WithModel(envOr("LUFT_SUMMARIZE_MODEL", "x-ai/grok-4.3"))

	// --- run ---------------------------------------------------------------

	abs, _ := absDir(*dir)
	subStatus := green("on")
	if *noSubagents {
		subStatus = dim("off")
	}
	row := func(label, value string) {
		fmt.Fprintf(os.Stderr, "  %s %s\n", grey(padRight(label, 10)), value)
	}
	fmt.Fprintln(os.Stderr)
	fmt.Fprintf(os.Stderr, "  %s %s\n", boldCyan("▍luft"), grey("— a fast, economical CLI coding agent"))
	fmt.Fprintln(os.Stderr)
	row("model", bold(*model))
	row("bash", *bashMode)
	row("subagents", subStatus)
	if !*noSubagents {
		row("explore", *exploreModel)
		row("plan", *planModel)
	}
	row("dir", abs)
	if resolvedLog != "" {
		row("log", resolvedLog)
	}
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, dim("  type a request, or /help for commands. ctrl-c to interrupt, ctrl-d to exit."))

	s := &session{
		agent:      agent,
		summarizer: summarizer,
		provider:   provider,
		memory:     memory,
		logPath:    resolvedLog,
	}
	s.repl(ctx)
}

// resolveLogPath turns a user-supplied -log value into an absolute path.
// "auto" expands to ~/.config/luft/sessions/<timestamp>.jsonl, creating
// the directory if needed. Any other value is treated as a literal path.
func resolveLogPath(spec string) (string, error) {
	if spec != "auto" {
		return filepath.Abs(spec)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".config", "luft", "sessions")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	stamp := time.Now().Format("2006-01-02T15-04-05")
	return filepath.Join(dir, stamp+".jsonl"), nil
}

func mustClient(provider luft.Provider, model string) *luft.Client {
	c, err := luft.New(luft.Config{
		Provider:  provider,
		Model:     model,
		MaxTokens: 8192,
		Retry: luft.RetryConfig{
			MaxRetries:  3,
			InitialWait: time.Second,
			MaxWait:     10 * time.Second,
		},
		SystemCache: &luft.CacheControl{Type: "ephemeral"},
	})
	if err != nil {
		log.Fatal(err)
	}
	return c
}

func makeConfirmer(autoYes bool) func(ctx context.Context, b luft.ToolBinding, input json.RawMessage) (bool, error) {
	reader := bufio.NewReader(os.Stdin)
	return func(ctx context.Context, b luft.ToolBinding, input json.RawMessage) (bool, error) {
		if !b.Meta.RequiresConfirmation || autoYes {
			return true, nil
		}
		fmt.Fprintf(os.Stderr, "\n[approve %s]\n", b.Tool.Name)
		if preview, ok := renderEditPreview(b.Tool.Name, input); ok {
			fmt.Fprint(os.Stderr, preview)
		} else if compact := compactJSON(input); compact != "" {
			fmt.Fprintf(os.Stderr, "  input: %s\n", compact)
		}
		fmt.Fprint(os.Stderr, "  approve? [y/N] ")
		line, err := reader.ReadString('\n')
		if err != nil {
			return false, nil
		}
		ans := strings.TrimSpace(strings.ToLower(line))
		return ans == "y" || ans == "yes", nil
	}
}

func compactJSON(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return string(raw)
	}
	out, err := json.Marshal(v)
	if err != nil {
		return string(raw)
	}
	s := string(out)
	if len(s) > 400 {
		s = s[:400] + "..."
	}
	return s
}

// envOr returns the value of env var name, or fallback if the variable
// is unset or empty. Used to give CLI flags env-var defaults.
func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

func parseBashMode(s string) (bash.Mode, error) {
	switch strings.ToLower(s) {
	case "restricted", "":
		return bash.ModeRestricted, nil
	case "standard":
		return bash.ModeStandard, nil
	case "unrestricted":
		return bash.ModeUnrestricted, nil
	}
	return 0, fmt.Errorf("unknown bash mode %q (want: restricted | standard | unrestricted)", s)
}

func absDir(dir string) (string, error) {
	if dir == "" {
		dir = "."
	}
	return filepath.Abs(dir)
}
