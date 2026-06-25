package main

import (
	"bytes"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestSpinnerSuspendKeepsPromptVisible is the regression test for the
// "agent hangs on a tool call" bug: a confirmation prompt written while the
// spinner runs was wiped every 80ms by the spinner's \r\x1b[K repaint, so
// the user never saw it and the CLI looked hung while blocked on stdin.
// Suspending the spinner for the duration of the prompt must stop the
// repaint loop entirely.
func TestSpinnerSuspendKeepsPromptVisible(t *testing.T) {
	prev := useColor
	useColor = true
	defer func() { useColor = prev }()

	var buf bytes.Buffer
	var mu sync.Mutex
	w := &lockedWriter{w: &buf, mu: &mu}

	sp := newSpinner(w)
	sp.Start("running tools…")
	time.Sleep(100 * time.Millisecond)

	// Confirmer takes the terminal, prints its prompt, and "waits on stdin".
	sp.Suspend()
	mu.Lock()
	buf.Reset() // ignore everything painted before the prompt
	buf.WriteString("approve? [y/a/A/N] ")
	mu.Unlock()
	time.Sleep(200 * time.Millisecond) // user reads the prompt

	// Capture the screen state during the suspended window — this is what the
	// user is looking at while typing. Before the fix it would be full of
	// \r\x1b[K erase sequences from the spinner; now it must be the bare prompt.
	mu.Lock()
	out := buf.String()
	mu.Unlock()

	sp.Resume() // user answered; tools still running
	sp.Stop()

	if out != "approve? [y/a/A/N] " {
		t.Errorf("spinner painted over the prompt while suspended: %q", out)
	}
}

// TestSpinnerResumeRestartsWhenSpinning verifies that a spinner suspended
// while running resumes spinning, so "running tools…" reappears after the
// user answers an approval prompt.
func TestSpinnerResumeRestartsWhenSpinning(t *testing.T) {
	prev := useColor
	useColor = true
	defer func() { useColor = prev }()

	var buf bytes.Buffer
	var mu sync.Mutex
	w := &lockedWriter{w: &buf, mu: &mu}

	sp := newSpinner(w)
	sp.Start("running tools…")
	time.Sleep(100 * time.Millisecond)
	sp.Suspend()

	mu.Lock()
	buf.Reset()
	mu.Unlock()

	sp.Resume()
	time.Sleep(180 * time.Millisecond)
	sp.Stop()

	mu.Lock()
	out := buf.String()
	mu.Unlock()
	if !strings.Contains(out, "running tools") {
		t.Errorf("expected spinner to resume after Resume, got %q", out)
	}
}

// TestSpinnerResumeStaysOffWhenIdle verifies that suspending a spinner that
// wasn't running doesn't spuriously start it on Resume.
func TestSpinnerResumeStaysOffWhenIdle(t *testing.T) {
	prev := useColor
	useColor = true
	defer func() { useColor = prev }()

	var buf bytes.Buffer
	var mu sync.Mutex
	w := &lockedWriter{w: &buf, mu: &mu}

	sp := newSpinner(w)
	sp.Suspend() // never started
	sp.Resume()
	time.Sleep(120 * time.Millisecond)
	sp.Stop()

	mu.Lock()
	out := buf.String()
	mu.Unlock()
	if out != "" {
		t.Errorf("expected no spinner output, got %q", out)
	}
}

// TestSpinnerStartDuringSuspendResumes verifies that a Start arriving while
// the spinner is suspended (e.g. the idle timer firing during a prompt)
// takes effect on Resume rather than being lost.
func TestSpinnerStartDuringSuspendResumes(t *testing.T) {
	prev := useColor
	useColor = true
	defer func() { useColor = prev }()

	var buf bytes.Buffer
	var mu sync.Mutex
	w := &lockedWriter{w: &buf, mu: &mu}

	sp := newSpinner(w)
	sp.Suspend()               // not running yet
	sp.Start("running tools…") // arrives while suspended — must not paint now
	mu.Lock()
	if buf.Len() != 0 {
		t.Errorf("Start painted while suspended: %q", buf.String())
	}
	mu.Unlock()

	sp.Resume()
	time.Sleep(180 * time.Millisecond)
	sp.Stop()

	mu.Lock()
	out := buf.String()
	mu.Unlock()
	if !strings.Contains(out, "running tools") {
		t.Errorf("expected deferred Start to paint after Resume, got %q", out)
	}
}
