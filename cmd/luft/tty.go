package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"
)

// ANSI escape sequences. We deliberately keep this tiny and self-contained
// rather than pull in a colour library — luft's go.mod is dep-free and
// staying that way is a feature.
const (
	ansiReset    = "\x1b[0m"
	ansiBold     = "\x1b[1m"
	ansiDim      = "\x1b[2m"
	ansiItalic   = "\x1b[3m"
	ansiRed      = "\x1b[31m"
	ansiGreen    = "\x1b[32m"
	ansiYellow   = "\x1b[33m"
	ansiBlue     = "\x1b[34m"
	ansiMagenta  = "\x1b[35m"
	ansiCyan     = "\x1b[36m"
	ansiGrey     = "\x1b[90m"
	ansiBoldCyan = "\x1b[1;36m"
)

// useColor is set during init based on TTY detection and the NO_COLOR /
// TERM=dumb conventions. When false, all paint helpers are no-ops.
var useColor = true

func init() {
	if os.Getenv("NO_COLOR") != "" || os.Getenv("TERM") == "dumb" {
		useColor = false
		return
	}
	if !isTTY(os.Stderr) {
		useColor = false
	}
}

// isTTY returns true if f is attached to a character device (a terminal).
// Pure stdlib: relies on os.ModeCharDevice, which the runtime fills in via
// fstat on Unix and GetFileType on Windows.
func isTTY(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func paint(s, code string) string {
	if !useColor || s == "" {
		return s
	}
	return code + s + ansiReset
}

func bold(s string) string     { return paint(s, ansiBold) }
func dim(s string) string      { return paint(s, ansiDim) }
func italic(s string) string   { return paint(s, ansiItalic) }
func red(s string) string      { return paint(s, ansiRed) }
func green(s string) string    { return paint(s, ansiGreen) }
func yellow(s string) string   { return paint(s, ansiYellow) }
func cyan(s string) string     { return paint(s, ansiCyan) }
func grey(s string) string     { return paint(s, ansiGrey) }
func boldCyan(s string) string { return paint(s, ansiBoldCyan) }

// humanBytes formats n as a short human-readable size string.
func humanBytes(n int) string {
	const (
		kib = 1024
		mib = 1024 * kib
	)
	switch {
	case n >= mib:
		return fmt.Sprintf("%.1f MiB", float64(n)/mib)
	case n >= kib:
		return fmt.Sprintf("%.1f KiB", float64(n)/kib)
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// truncate shortens s to at most n runes, appending an ellipsis when cut.
func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

// oneLine collapses any internal newlines/tabs to spaces. Useful when
// echoing user-supplied tool arguments (e.g. multi-line bash commands)
// onto a single status line.
func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	return strings.TrimSpace(s)
}

// spinner is a tiny activity indicator. It's safe to Start, Stop, and Set
// from any goroutine. Start is idempotent: a second Start while running
// just updates the label. Stop blocks until the render goroutine has
// emitted the line-erase escape, so it's safe to print other content
// immediately afterward without overlap.
//
// When useColor is false (non-TTY, NO_COLOR, or TERM=dumb) the spinner
// is a no-op — no goroutine, no output.
type spinner struct {
	w io.Writer

	mu        sync.Mutex
	running   bool
	suspended bool // held off while an interactive prompt owns the terminal
	resume    bool // restart on Resume (was running, or a Start arrived while suspended)
	label     string
	stopCh    chan struct{}
	doneCh    chan struct{}
}

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

func newSpinner(w io.Writer) *spinner {
	return &spinner{w: w}
}

// Start begins (or updates) the spinner with the given label. Concurrent
// calls are safe.
func (s *spinner) Start(label string) {
	if !useColor {
		return
	}
	s.mu.Lock()
	s.label = label
	if s.suspended {
		// An interactive prompt owns the terminal; remember that a spin
		// was wanted and let Resume repaint once the prompt clears.
		s.resume = true
		s.mu.Unlock()
		return
	}
	if s.running {
		s.mu.Unlock()
		return
	}
	s.running = true
	s.stopCh = make(chan struct{})
	s.doneCh = make(chan struct{})
	s.mu.Unlock()
	go s.run()
}

// Stop halts the spinner and erases its line. Returns once the render
// goroutine has exited.
func (s *spinner) Stop() {
	if !useColor {
		return
	}
	s.teardown()
}

// teardown stops the render goroutine and erases the line. Caller must not
// hold s.mu. A no-op if the spinner isn't currently running.
func (s *spinner) teardown() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	s.running = false
	close(s.stopCh)
	doneCh := s.doneCh
	s.mu.Unlock()
	<-doneCh
	// Carriage return + clear-to-end-of-line. Repaint owners write
	// after this, and the line will be empty.
	fmt.Fprint(s.w, "\r\x1b[K")
}

// Suspend halts the spinner and blocks it from repainting until Resume is
// called. Use it around interactive terminal reads (confirmation prompts):
// the render goroutine rewrites its line every tick with a carriage return
// + line-erase, which would otherwise wipe out a prompt as fast as it is
// printed — making the CLI look hung while it silently waits on stdin.
// Resume restores whatever spin state was wanted while suspended.
func (s *spinner) Suspend() {
	if !useColor {
		return
	}
	s.mu.Lock()
	if s.suspended {
		s.mu.Unlock()
		return
	}
	s.suspended = true
	s.resume = s.running // resume afterward only if it was actually spinning
	s.mu.Unlock()
	s.teardown()
}

// Resume lifts a Suspend. If the spinner was running when suspended (or a
// Start arrived in the meantime) it restarts with the most recent label.
func (s *spinner) Resume() {
	if !useColor {
		return
	}
	s.mu.Lock()
	if !s.suspended {
		s.mu.Unlock()
		return
	}
	s.suspended = false
	resume := s.resume
	s.resume = false
	label := s.label
	s.mu.Unlock()
	if resume {
		s.Start(label)
	}
}

func (s *spinner) run() {
	t := time.NewTicker(80 * time.Millisecond)
	defer t.Stop()
	i := 0
	for {
		select {
		case <-s.stopCh:
			close(s.doneCh)
			return
		case <-t.C:
			s.mu.Lock()
			label := s.label
			s.mu.Unlock()
			fmt.Fprintf(s.w, "\r\x1b[K%s %s", cyan(spinnerFrames[i%len(spinnerFrames)]), dim(label))
			i++
		}
	}
}
