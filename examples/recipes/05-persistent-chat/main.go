// Recipe 05: persistent chat with a per-turn activity log.
//
// The headline claim: in luft, a session is plain data. The read-modify-write
// cycle is ordinary code, the storage backend is a five-method interface,
// and a Recorder captures intermediate turn activity (model calls, tool
// calls, retries) into Session.Events alongside the model-facing History.
//
// One process, multiple turns:
//
//	$ go run ./examples/recipes/05-persistent-chat -id alice "what's 2+2?"
//	$ go run ./examples/recipes/05-persistent-chat -id alice "and times 10?"
//	$ go run ./examples/recipes/05-persistent-chat -id alice -dump
//
// The first two calls each load the session, run a turn, and save. The
// -dump flag prints the full event log so you can see what happened inside
// each turn — not just the final messages.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/providers/anthropic"
	"github.com/lukemuz/luft/stores"
	"github.com/lukemuz/luft/tools/math"
)

func main() {
	id := flag.String("id", "default", "session id")
	dir := flag.String("dir", filepath.Join(os.TempDir(), "luft-chat"), "session directory")
	dump := flag.Bool("dump", false, "print the recorded event log instead of taking a turn")
	flag.Parse()

	ctx := context.Background()

	store, err := stores.NewFileStore(*dir)
	if err != nil {
		log.Fatal(err)
	}

	// Open-or-create. Errors other than "not found" are fatal.
	sess, err := luft.Load(ctx, store, *id)
	if err != nil {
		log.Fatal(err)
	}

	if *dump {
		dumpEvents(sess)
		return
	}

	user := strings.TrimSpace(strings.Join(flag.Args(), " "))
	if user == "" {
		log.Fatal("usage: persistent-chat [-id ID] [-dir PATH] \"your message\"   |   -dump")
	}

	provider, err := anthropic.NewProviderFromEnv()
	if err != nil {
		log.Fatal(err)
	}

	// A Recorder that appends to sess.Events. After Save, the full activity
	// log of this turn is on disk alongside History.
	rec := luft.RecorderToSession(sess)

	client, err := luft.New(luft.Config{
		Provider: provider,
		Model:    luft.ModelHaiku,
		Recorder: rec,
	})
	if err != nil {
		log.Fatal(err)
	}

	a := luft.Agent{
		Client:  client,
		System:  "You are a helpful assistant. Use the calculator tool when arithmetic is needed.",
		Tools:   math.New().Toolset(),
		MaxIter: 6,
	}

	// Read-modify-write. The session is unchanged until Step returns
	// successfully — a failed turn means the next attempt starts from the
	// same state.
	sess.History = append(sess.History, luft.NewUserMessage(user))
	result, err := a.Step(ctx, sess.History)
	if err != nil {
		log.Fatalf("turn failed: %v", err)
	}
	sess.History = result.Messages

	if err := luft.Save(ctx, store, sess); err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.FinalText())
	fmt.Fprintf(os.Stderr,
		"\nturn ok: %d events recorded; session at %s\n",
		recentEventCount(sess), filepath.Join(*dir, *id+".json"),
	)
}

// dumpEvents prints sess.Events in a compact, human-friendly form.
func dumpEvents(sess *luft.Session) {
	if len(sess.Events) == 0 {
		fmt.Println("(no events recorded)")
		return
	}
	for _, ev := range sess.Events {
		switch ev.Type {
		case luft.EventTurnStart:
			fmt.Printf("[%4d] turn=%s start (history=%d msgs)\n", ev.Seq, ev.TurnID, len(ev.History))
		case luft.EventModelRequest:
			fmt.Printf("[%4d] turn=%s iter=%d → model request\n", ev.Seq, ev.TurnID, ev.Iter)
		case luft.EventModelResponse:
			fmt.Printf("[%4d] turn=%s iter=%d ← model response stop=%s in=%d out=%d\n",
				ev.Seq, ev.TurnID, ev.Iter, ev.StopReason, ev.Usage.InputTokens, ev.Usage.OutputTokens)
		case luft.EventRetryAttempt:
			fmt.Printf("[%4d] turn=%s iter=%d retry attempt=%d wait=%s\n",
				ev.Seq, ev.TurnID, ev.Iter, ev.Attempt, ev.Wait)
		case luft.EventToolCallStart:
			fmt.Printf("[%4d] turn=%s iter=%d → tool %s(%s)\n",
				ev.Seq, ev.TurnID, ev.Iter, ev.ToolName, compactJSON(ev.ToolInput))
		case luft.EventToolCallEnd:
			if ev.IsError {
				fmt.Printf("[%4d] turn=%s iter=%d ← tool %s ERROR: %s\n",
					ev.Seq, ev.TurnID, ev.Iter, ev.ToolName, ev.ToolError)
			} else {
				fmt.Printf("[%4d] turn=%s iter=%d ← tool %s = %s\n",
					ev.Seq, ev.TurnID, ev.Iter, ev.ToolName, truncate(ev.ToolOutput, 80))
			}
		case luft.EventTurnEnd:
			fmt.Printf("[%4d] turn=%s end (in=%d out=%d)\n",
				ev.Seq, ev.TurnID, ev.Usage.InputTokens, ev.Usage.OutputTokens)
		case luft.EventTurnError:
			fmt.Printf("[%4d] turn=%s ERROR: %s\n", ev.Seq, ev.TurnID, ev.Err)
		}
	}
}

// recentEventCount returns the number of events in the most recent turn,
// found by counting back from the last TurnStart.
func recentEventCount(sess *luft.Session) int {
	for i := len(sess.Events) - 1; i >= 0; i-- {
		if sess.Events[i].Type == luft.EventTurnStart {
			return len(sess.Events) - i
		}
	}
	return len(sess.Events)
}

func compactJSON(b json.RawMessage) string {
	if len(b) == 0 {
		return ""
	}
	return string(b)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
