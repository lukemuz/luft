// luft is a CLI coding agent built on the luft toolkit.
//
// Topology (Phase 2):
//
//	main agent           (z-ai/glm-5.2 by default — configurable via -model)
//	  ├── direct tools   workspace read-only + trained bash + trained editor
//	  │                  + todo + clock + batch
//	  ├── explore        research subagent (default tier: ultrafast) —
//	  │                  workspace read-only + restricted bash + batch + clock.
//	  │                  Used for cheap, parallelisable inspection; its
//	  │                  iteration history never enters the main context.
//	  ├── plan           design subagent (default tier: powerful) — read-only
//	  │                  tools, no shell or edits. Used when the main agent
//	  │                  wants a focused reasoner for design or hard debugging.
//	  └── implement      editing clone (default tier: medium) — the full editing
//	                     toolset (read + edit + bash + batch + todo + web) but NO
//	                     subagents of its own. Carries one well-scoped change to
//	                     completion in an isolated context and returns a summary;
//	                     the main agent reviews the diff. Disable with -no-implement.
//
// Subagents share three model tiers — powerful / medium / ultrafast (set via
// -model-powerful / -model-medium / -model-ultrafast). Each subagent has a
// default tier (above), and the calling model may request a different tier per
// call via a "tier" argument. All subagents run through the Agent block, so each
// gets the same in-loop context trimming/summarization the main agent has.
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
// The main agent defaults to z-ai/glm-5.2 (-model). Subagent tiers default to
// z-ai/glm-5.2 (powerful), x-ai/grok-4.3 (medium), and openai/gpt-oss-120b:nitro
// (ultrafast) — any OpenRouter slug works.
//
// Flags:
//
//	-dir             working directory the agent is sandboxed to (default cwd)
//	-model           main-agent model id (default z-ai/glm-5.2)
//	-model-powerful  powerful subagent tier (default x-ai/grok-4.3)
//	-model-medium    medium subagent tier (default z-ai/glm-5.2)
//	-model-ultrafast ultrafast subagent tier (default openai/gpt-oss-120b:nitro)
//	-no-subagents    disable all subagent tools (explore, plan, implement)
//	-no-implement    disable only the implement subagent
//	-no-fetch       disable the native web_fetch tool
//	-no-search      disable the OpenRouter web_search provider tool
//	-bash           bash safety mode: restricted | standard | unrestricted
//	-yes            auto-approve every confirmation prompt
//	-max-iter       max model calls per turn (default 30)
//	-context-budget token budget before automatic summarization (default 750000)
//	-result-budget  byte threshold above which bash output is summarized (default 12000; 0 disables)
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
	"io"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
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
var version = "v0.2.0"

// Subagent model-tier names. Subagents share these three classes; the calling
// model may request one per call, otherwise each subagent uses its role tier.
const (
	tierPowerful  = "powerful"
	tierMedium    = "medium"
	tierUltrafast = "ultrafast"
)

const mainSystemPrompt = `You are luft, a fast and economical CLI coding assistant built on the luft toolkit. You operate inside a workspace directory and help the user with software engineering tasks — reading code, making edits, running commands, and reasoning about changes.

Available tools:
- list_directory, Glob, Grep, read_file, file_info: read-only filesystem inspection
- str_replace_based_edit_tool: view/create/str_replace/insert against files
- multi_edit: apply several str_replace edits to ONE file atomically (all-or-nothing). Prefer over repeated single edits when changing 2+ sites in the same file — one round trip, one confirmation.
- bash: run shell commands (safety policy varies by configuration)
- todo_write, todo_read: maintain a short planning checklist for multi-step work
- batch: run 2+ independent read-only calls in one turn. Default for independent reads/greps/inspections; skip only when a later call depends on an earlier result.
- web_fetch (when available): download an http(s) URL and return its content as text. HTML is converted to a plain-text approximation; long pages paginate via max_length + start_index. Use this for documentation lookups and inspecting URLs from error messages.
- web_search (when available): run a web search and get results with citations. Use when you do NOT already have a URL — current library docs, an unfamiliar API, or an error message you want to look up. Pair with web_fetch to read a promising result in full.
- explore (when available): bounded research specialist that defaults to the ultrafast tier (a small model at very high throughput — extremely fast and cheap) — repo inspection or well-scoped Q&A (provided context, fetched docs, web search, or its own knowledge); returns a concise summary with file dumps and fetches kept out of your context. Lean on it freely and fan several out in parallel.
- plan (when available): design specialist on the powerful tier, running in a fresh, focused context — get a structured plan before non-trivial multi-file edits, interface changes, or when stuck on a hypothesis
- implement (when available): a clone of yourself with the full editing toolset on the medium tier, running in its own clean context — delegate a well-scoped, independently-verifiable change you've already designed and get back a summary of what it did (its edit/test churn stays out of your context). You still review its diff.
- now: current time

Each subagent (explore, plan, implement) runs on one of three model tiers — powerful (strongest reasoning and benchmarks, but slowest/priciest — reserve for hard tasks), medium (nearly as capable, faster and cheaper), ultrafast (fastest and cheapest, small model) — and defaults to the tier noted above. Pass an optional "tier" argument to override per call: escalate to powerful for a genuinely hard task, or drop to ultrafast for something trivial. When unsure, omit it and take the default.

Two automatic context savers may alter tool results: re-issuing an identical read returns a brief "unchanged" notice instead of the full content (trust your earlier read, or pass a line range to force a fresh one), and very large command output is condensed to its errors and final status. Both preserve correctness-relevant detail.

# Working effectively

1. Default to the explore subagent for breadth — code you need to understand but won't necessarily edit: repo inspection (understand a module, find all usages, audit a pattern) and well-scoped Q&A (look up a stdlib function, summarise an RFC, answer a factual question with provided context). It's cheap, its file dumps and fetches stay out of your context, and you receive only its summary. Choose explore over batch when the search is broad or open-ended and you only need the conclusion; choose batch when you need the raw file contents in your own context to act on.
2. Read it yourself for (a) tight, surgical lookups (one file, one symbol) and (b) ANY code you are about to edit. explore's summary comes from a cheaper model: it loses detail you need to write a correct change, and you can't cheaply verify what you never saw. Delegating breadth is a good trade; delegating the reading of code you're about to modify is not.
3. Default to batch for independent read-only work. Each tool call is a full LLM round trip, so if you'd otherwise issue 2+ reads/greps/inspections that don't depend on each other, batch them. Issue solo calls only when a later call's input depends on an earlier call's output (e.g. grep first, then read only the files it returned).
4. Call plan as a routine first step for substantive design work — non-trivial multi-file changes, interface or data-shape changes, decisions with multiple plausible tradeoffs, or stuck debugging (2+ turns without a clear hypothesis). Pass the question with the context you've gathered. Skip plan for routine single-file changes or when your approach is already clear.
5. Delegate to implement when a change is well-scoped and independently verifiable and keeping its churn out of your context is worth it — a mechanical refactor repeated across files, or a self-contained unit you can hand off with a precise spec and a clear "done when tests pass". Do it yourself when the change is central to what you're reasoning about, depends on decisions you haven't made, or is small enough that writing the spec costs more than the edit. Either way you own the review — read implement's diff before trusting it.
6. Brief subagents like a colleague who just walked in: state the goal, the relevant context you've already gathered (file paths, line numbers, error messages, what you've ruled out), and what shape of answer you need. Terse one-line prompts produce shallow, generic work. For multi-step tasks, call todo_write at the start and update it as you go, keeping at most one item in_progress.

# Writing code

7. Match the conventions of the surrounding code — naming, error handling, imports, file layout. Read a neighbouring file before introducing a new pattern. When str_replace reports old_str isn't unique, add surrounding context to disambiguate rather than retrying the same string; to change several sites in one file, use multi_edit.
8. Don't add features, refactor, or introduce abstractions beyond what the task requires. A bug fix doesn't need surrounding cleanup. Three similar lines is better than a premature abstraction. No half-finished scaffolding for hypothetical future requirements.
9. Don't add error handling, validation, or fallbacks for cases that can't happen. Trust internal code and framework guarantees. Validate only at system boundaries — user input, external APIs, untrusted data.
10. Don't add backwards-compatibility shims, feature flags, or "removed" marker comments for code you can just change. If something is unused, delete it.
11. Default to writing no comments. Only add one when the *why* is non-obvious — a hidden constraint, a subtle invariant, a workaround for a specific bug. Don't explain *what* the code does; well-named identifiers do that. Don't reference the current task ("added for X", "used by Y") — that belongs in commit messages and rots as the code evolves.
12. Watch for security issues: command injection, path traversal, SQL injection, XSS, and the rest of the OWASP top 10. If you notice you've written insecure code, fix it immediately.

# Verifying your work

13. After making edits, verify with appropriate checks: build, type-check, run affected tests via bash. Review your own changes with git diff before declaring done. Don't trust an edit you haven't checked.
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

const implementSystemPrompt = `You are luft's implement specialist — a focused engineer who carries one well-scoped change to completion in your own clean context.

You receive a self-contained task from an orchestrator that has already done the surrounding design and reading. You have the full editing toolset: read-only filesystem inspection, str_replace_based_edit_tool and multi_edit for changes, bash for builds/tests/git, batch for read-only fan-out, todo for tracking multi-step work, and web_fetch/web_search when available. You have NO subagents — do the work yourself.

Operating principles:
1. Read before you write. Open every file you are about to edit; match the conventions of the surrounding code (naming, error handling, imports, layout). Read a neighbour before introducing a new pattern.
2. Stay inside the task. Make the change you were asked for and nothing more — no opportunistic refactors, no scaffolding for hypothetical futures, no backwards-compat shims for code you can just change. Three similar lines beat a premature abstraction.
3. Write no comments unless the *why* is non-obvious (a hidden constraint, a subtle invariant, a workaround). Don't narrate *what* the code does.
4. Watch for security issues (injection, path traversal, the OWASP top 10) and fix any you introduce immediately.
5. Verify before you report: build, type-check, and run the affected tests via bash. Review your own diff with git diff. If you can't exercise the change in this environment, say so plainly rather than claiming success.
6. If the same approach fails twice, stop and reconsider rather than trying a third variation — report what you tried, what failed, and your current hypothesis.
7. Act with care. Local reversible actions (edits, reads, tests, builds) are free — just do them. But do NOT take destructive or shared-state actions unless the task explicitly calls for them: deleting files/branches, dropping data, force-pushing, rewriting history, committing, pushing, or downgrading dependencies. Never use a destructive shortcut to get past a failing check (no --no-verify, no skipping hooks). If the task seems to require one of these, do the reversible work and flag the remaining step in your summary rather than performing it.

Your final message is the ONLY thing the orchestrator sees — your iteration history is discarded. Return a tight summary: the files you changed and how (path:line where useful), the key decisions you made, the exact verification you ran and its result, and anything you could NOT verify or that remains open. The orchestrator will review your actual diff, so be precise about what it will find.`

func main() {
	dir := flag.String("dir", ".", "working directory the agent is sandboxed to (defaults to the current directory)")
	model := flag.String("model", envOr("LUFT_MODEL", "z-ai/glm-5.2"), "main-agent model id (any OpenRouter slug; env: LUFT_MODEL)")
	// Subagents pick from three shared model tiers; the calling model may
	// request a tier per call (see subagent tool schemas), otherwise each
	// subagent uses its role default (explore→ultrafast, plan→powerful,
	// implement→medium).
	powerfulModel := flag.String("model-powerful", envOr("LUFT_MODEL_POWERFUL", "z-ai/glm-5.2"), "powerful tier: strong reasoning model for subagents (env: LUFT_MODEL_POWERFUL)")
	mediumModel := flag.String("model-medium", envOr("LUFT_MODEL_MEDIUM", "x-ai/grok-4.3"), "medium tier: fast, high-quality model for subagents (env: LUFT_MODEL_MEDIUM)")
	ultrafastModel := flag.String("model-ultrafast", envOr("LUFT_MODEL_ULTRAFAST", "openai/gpt-oss-120b:nitro"), "ultrafast tier: small model at very high throughput for subagents (env: LUFT_MODEL_ULTRAFAST)")
	noSubagents := flag.Bool("no-subagents", false, "disable all subagent tools (explore, plan, implement)")
	noImplement := flag.Bool("no-implement", false, "disable only the implement subagent (the editing clone)")
	noFetch := flag.Bool("no-fetch", false, "disable the native web_fetch tool")
	noSearch := flag.Bool("no-search", false, "disable the OpenRouter web_search provider tool")
	bashMode := flag.String("bash", "restricted", "bash safety mode: restricted | standard | unrestricted")
	autoYes := flag.Bool("yes", false, "auto-approve every confirmation prompt")
	maxIter := flag.Int("max-iter", 30, "max model calls per turn")
	contextBudget := flag.Int("context-budget", envOrInt("LUFT_CONTEXT_BUDGET", 750_000), "token budget before automatic in-loop summarization (env: LUFT_CONTEXT_BUDGET)")
	resultBudget := flag.Int("result-budget", envOrInt("LUFT_RESULT_BUDGET", 12_000), "byte threshold above which noisy bash output is summarized; 0 disables (env: LUFT_RESULT_BUDGET)")
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

	// One spinner for the whole session, shared between the REPL turn loop
	// and the confirmer so the confirmer can suspend it while it owns the
	// terminal for an approval prompt.
	sp := newSpinner(os.Stderr)
	confirm := makeConfirmer(*autoYes, sp, os.Stdin, os.Stderr)

	// Summarizer for /compact, automatic in-loop trimming, and oversized-
	// result compression. Defaults to glm-5.2 (the main-tier model); override
	// via LUFT_SUMMARIZE_MODEL to point it at a cheaper slug. Derived from
	// mainClient after the recorder is attached so its calls log too.
	summarizer := mainClient.WithModel(envOr("LUFT_SUMMARIZE_MODEL", "z-ai/glm-5.2"))

	// The three shared subagent model tiers. Derived from mainClient (after the
	// recorder is attached) so every tier's calls log to the same session file.
	// A subagent picks a tier per call via its "tier" argument, defaulting to
	// its role tier. Descriptions are advertised to the calling model.
	tiers := []subagent.Tier{
		{Name: tierPowerful, Description: "strongest reasoning and best benchmarks, but slowest and priciest — reserve for genuinely hard analysis, design, or debugging", Client: mainClient.WithModel(*powerfulModel)},
		{Name: tierMedium, Description: "nearly as capable but faster and cheaper — a strong general-purpose class for real work", Client: mainClient.WithModel(*mediumModel)},
		{Name: tierUltrafast, Description: "a small model at very high throughput — fastest and cheapest, for quick reads, lookups, and summaries", Client: mainClient.WithModel(*ultrafastModel)},
	}

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

	// Project memory is loaded once and shared: appended to the main system
	// prompt and to each subagent's system prompt so the cheap explore/plan
	// models also know the project's conventions.
	memory := loadProjectMemory(*dir)
	withMemory := func(base string) string {
		if memory == "" {
			return base
		}
		return base + "\n\n## Project memory\n\n" + memory
	}

	// OpenRouter's hosted web_search, advertised to the main agent and the
	// explore subagent unless -no-search is set. OpenRouter executes the
	// search server-side and returns results inline with citations, so it
	// works regardless of the underlying model slug. An empty slice is a
	// no-op when passed to WithProviderTools.
	var searchTools []luft.ProviderTool
	if !*noSearch {
		searchTools = []luft.ProviderTool{openrouter.WebSearch(openrouter.WebSearchOpts{})}
	}

	// Batch tool for read-only fan-out. Built from already-wrapped read-only
	// bindings so each sub-call inherits the timeout/limit/logging stack.
	roBatchBinding := batch.New(batch.Config{
		Bindings:    append(append(append([]luft.ToolBinding{}, roTools.Bindings...), subBashToolset.Bindings...), webTools.Bindings...),
		MaxParallel: 8,
	})

	// --- subagent tools ----------------------------------------------------

	// Every subagent runs through the Agent block (see subagent.New), so they
	// share the main agent's in-loop context management: a long subagent run is
	// trimmed and summarized in place rather than blowing its context window.
	// The summarizer client is the same one the main loop uses.
	subagentContext := luft.ContextManager{
		MaxTokens:  *contextBudget,
		KeepFirst:  1,
		KeepRecent: 30,
		Summarizer: makeAutoSummarizer(summarizer),
	}

	var subagentBindings []luft.ToolBinding
	if !*noSubagents {
		exploreTools := luft.MustJoin(roTools, subBashToolset, webTools, luft.Tools(roBatchBinding)).
			CacheLast(luft.Ephemeral()).
			WithProviderTools(searchTools...)
		exploreBinding, err := subagent.New(subagent.Config{
			Name:        "explore",
			Description: "Delegate a bounded research task to an EXTREMELY fast, cheap specialist (defaults to the ultrafast tier — a small model served at very high throughput, so it returns in a fraction of the time and cost of a main-model turn). Lean on it freely and prefer firing several in parallel over reading widely yourself. Two main shapes: (a) repo research — find callers, audit a pattern, summarise a module, locate references; (b) bounded Q&A — answer a well-scoped question from provided context, fetched docs, web search, or general knowledge (e.g. 'what does this stdlib function do', 'summarise this RFC's caching rules'). The specialist has read-only filesystem tools, restricted bash, web_fetch, web_search, and batch fan-out; it returns a concise summary and its iteration history stays out of your context. Pass a self-contained task description with any context it needs; raise `tier` for research that needs real reasoning. For the specific code you are about to edit, read it yourself: a small-model summary loses detail you need to write a correct change, and you can't cheaply verify what you didn't see.",
			Tiers:       tiers,
			DefaultTier: tierUltrafast,
			System:      withMemory(exploreSystemPrompt),
			Tools:       exploreTools,
			Context:     subagentContext,
			MaxIter:     50,
		})
		if err != nil {
			log.Fatal(err)
		}

		planBinding, err := subagent.New(subagent.Config{
			Name:        "plan",
			Description: "Delegate design and architecture work to a focused planning specialist running in its own clean context (defaults to the powerful tier — a strong reasoning model). Get a structured plan before editing when work touches 3+ files, changes a public interface or shared data shape, has multiple plausible tradeoffs, or you've spent 2+ debugging turns without a clear hypothesis. Pass the question PLUS context you've gathered (file excerpts, error messages, prior attempts). Returns a numbered plan with files to touch and risks. Skip for routine single-file changes or when your approach is already clear; drop `tier` to a faster class for lightweight planning.",
			Tiers:       tiers,
			DefaultTier: tierPowerful,
			System:      withMemory(planSystemPrompt),
			Tools:       roTools.CacheLast(luft.Ephemeral()),
			Context:     subagentContext,
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

	// bash gets summarizing compression (build/test logs are the main source
	// of oversized, noisy output); the editor does not — its output is verbatim
	// file content the model needs intact to write correct edits.
	bashMiddleware := []luft.Middleware{
		luft.WithConfirmation(confirm),
		luft.WithTimeout(60 * time.Second),
	}
	if *resultBudget > 0 {
		bashMiddleware = append(bashMiddleware, summarizingResultLimit(summarizer, *resultBudget))
	}
	bashMiddleware = append(bashMiddleware,
		luft.WithResultLimit(64*1024),
		luft.WithLogging(logger),
	)
	bashTools := luft.Tools(mainBashBindings...).Wrap(bashMiddleware...)
	editorTools := luft.Tools(editorBindings...).Wrap(
		luft.WithConfirmation(confirm),
		luft.WithTimeout(60*time.Second),
		luft.WithResultLimit(64*1024),
		luft.WithLogging(logger),
	)
	editTools := luft.MustJoin(bashTools, editorTools)

	// --- implement subagent ------------------------------------------------

	// A clone of the main agent that carries one well-scoped change to
	// completion in an isolated context. It shares the editing toolset (so its
	// edits and shell commands flow through the same confirmer) but is NOT
	// given the subagent bindings — subagents must not recurse. Built here,
	// after editTools, and appended to the subagent set the main agent sees.
	if !*noSubagents && !*noImplement {
		implementTools := luft.MustJoin(
			roTools,
			luft.Tools(roBatchBinding),
			editTools,
			todo.New().Toolset(),
			webTools,
		).CacheLast(luft.Ephemeral()).
			WithProviderTools(searchTools...)
		implementBinding, err := subagent.New(subagent.Config{
			Name:        "implement",
			Description: "Delegate a well-scoped, self-contained code change to a clone of yourself running in its own clean context (defaults to the medium tier — fast and high-quality). It has the full editing toolset (read, str_replace/multi_edit, bash, batch, todo, web) but no subagents, and returns a concise summary of what it changed while keeping its edit/grep/test churn out of your context. Use it for an independently-verifiable unit of work you've already scoped — e.g. 'implement function X per this signature and make its tests pass', 'apply this exact refactor across package Y'. Pass a precise spec PLUS the context it needs (target files, signatures, conventions, how to verify), and raise `tier` to powerful for a genuinely hard change. Run implement calls one at a time, NOT in parallel — the clones share one working tree, so concurrent edits can collide. You still own review: the clone reports what it did, but check its diff. Do it yourself instead when the change is central to your reasoning, spans decisions you haven't made yet, or is small enough that delegating costs more than it saves.",
			Tiers:       tiers,
			DefaultTier: tierMedium,
			System:      withMemory(implementSystemPrompt),
			Tools:       implementTools,
			Context:     subagentContext,
			MaxIter:     40,
		})
		if err != nil {
			log.Fatal(err)
		}
		subagentBindings = append(subagentBindings, implementBinding)
	}

	// --- main agent assembly ----------------------------------------------

	// The main agent's read tools carry SHA-based dedup so re-reading an
	// unchanged file returns a short notice instead of paying full tokens
	// again. roTools itself stays un-deduped for batch/explore, which have
	// their own contexts and must not share this cache.
	mainReadTools := roTools.Wrap(dedupReads(512))

	mainTools := luft.MustJoin(
		mainReadTools,
		luft.Tools(roBatchBinding),
		editTools,
		todo.New().Toolset(),
		webTools,
		luft.Tools(subagentBindings...),
	).CacheLast(luft.Ephemeral()). // cache the entire tool block — stable per session
					WithProviderTools(searchTools...)

	agent := luft.Agent{
		Client: mainClient,
		System: withMemory(mainSystemPrompt),
		Tools:  mainTools,
		Context: luft.ContextManager{
			MaxTokens:  *contextBudget,
			KeepFirst:  1,
			KeepRecent: 30,
			// Without a Summarizer, mid-turn trimming silently DROPS the middle
			// of a long autonomous run (including tool results). With it, the
			// trimmed span is replaced by a compact summary so the model keeps
			// the gist instead of losing it outright.
			Summarizer: makeAutoSummarizer(summarizer),
		},
		MaxIter: *maxIter,
	}

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
		// The three shared tiers, then each subagent's default tier (the
		// calling model may request a different tier per call).
		row("powerful", *powerfulModel)
		row("medium", *mediumModel)
		row("ultrafast", *ultrafastModel)
		row("explore", grey("tier ")+tierUltrafast)
		row("plan", grey("tier ")+tierPowerful)
		if *noImplement {
			row("implement", dim("off"))
		} else {
			row("implement", grey("tier ")+tierMedium)
		}
	}
	searchStatus := green("on")
	if *noSearch {
		searchStatus = dim("off")
	}
	row("web search", searchStatus)
	compressStatus := dim("off")
	if *resultBudget > 0 {
		compressStatus = fmt.Sprintf("%s %s", green("on"), grey(fmt.Sprintf("(>%d B)", *resultBudget)))
	}
	row("compress", compressStatus)
	approveStatus := grey("prompt") + grey(" (y/a/A per call)")
	if *autoYes {
		approveStatus = yellow("auto") + grey(" (-yes: every call approved)")
	}
	row("approve", approveStatus)
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
		sp:         sp,
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

// makeConfirmer builds the interactive tool-approval callback. in/out are the
// streams it prompts on (os.Stdin/os.Stderr in production; injectable in
// tests). sp is suspended for the duration of each prompt so its repaint loop
// can't erase the prompt off the terminal.
func makeConfirmer(autoYes bool, sp *spinner, in io.Reader, out io.Writer) func(ctx context.Context, b luft.ToolBinding, input json.RawMessage) (bool, error) {
	reader := bufio.NewReader(in)
	var mu sync.Mutex
	approvedAll := map[string]bool{} // tool names whitelisted for the session via "a"
	yesAll := autoYes                // session-wide auto-approve, settable at the prompt via "A"
	return func(ctx context.Context, b luft.ToolBinding, input json.RawMessage) (bool, error) {
		if !b.Meta.RequiresConfirmation {
			return true, nil
		}
		// Tool calls can run concurrently; hold the lock across the whole
		// prompt so two confirmations never interleave on the terminal and the
		// approval set is accessed safely.
		mu.Lock()
		defer mu.Unlock()
		if yesAll || approvedAll[b.Tool.Name] {
			return true, nil
		}
		// Suspend the spinner for the duration of the prompt: its repaint
		// loop writes to the same stream and would otherwise erase the
		// prompt line every tick, leaving the CLI looking hung while it
		// blocks on stdin.
		sp.Suspend()
		defer sp.Resume()

		fmt.Fprintf(out, "\n%s %s\n", yellow("⏵ approve"), bold(b.Tool.Name))
		if preview, ok := renderEditPreview(b.Tool.Name, input); ok {
			fmt.Fprint(out, preview)
		} else if compact := compactJSON(input); compact != "" {
			fmt.Fprintf(out, "  %s %s\n", grey("input:"), compact)
		}
		fmt.Fprintf(out, "  approve? %s  %s ",
			bold("[y/a/A/N]"),
			dim("y=yes · a=all "+b.Tool.Name+" · A=all tools · N=no"))
		line, err := reader.ReadString('\n')
		if err != nil && line == "" {
			// EOF (e.g. piped/closed stdin) or read error with nothing typed:
			// treat as a decline rather than silently approving. Print a newline
			// so the next output doesn't run onto the prompt line.
			fmt.Fprintln(out)
			return false, nil
		}
		switch strings.TrimSpace(line) {
		case "A", "All", "ALL":
			yesAll = true
			return true, nil
		case "a", "all":
			approvedAll[b.Tool.Name] = true
			return true, nil
		case "y", "Y", "yes", "Yes":
			return true, nil
		default:
			return false, nil
		}
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

// envOrInt is the integer counterpart of envOr. A non-numeric value falls
// back rather than erroring, so a typo in the environment can't abort startup.
func envOrInt(name string, fallback int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
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
