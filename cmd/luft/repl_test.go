package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestReadInput(t *testing.T) {
	tests := []struct {
		name  string
		in    string
		want  string
		wantOK bool
	}{
		{"plain single line", "hello\n", "hello", true},
		{"command line", "/tokens\n", "/tokens", true},
		{
			"bracketed paste preserves internal newlines as one unit",
			pasteStart + "line one\nline two\nline three" + pasteEnd + "\n",
			"line one\nline two\nline three",
			true,
		},
		{
			"text typed before and after the paste is kept",
			"pre " + pasteStart + "a\nb" + pasteEnd + " post\n",
			"pre a\nb post",
			true,
		},
		{
			"bare CR line separators inside a paste are normalised",
			pasteStart + "a\rb\rc" + pasteEnd + "\n",
			"a\nb\nc",
			true,
		},
		{"eof returns not-ok", "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sc := bufio.NewScanner(strings.NewReader(tt.in))
			got, ok := readInput(sc)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if got != tt.want {
				t.Errorf("readInput() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestToolInputPreview(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string // substring expected in result
	}{
		{
			"bash command wins over other fields",
			`{"command": "go test ./...", "cwd": "."}`,
			"command=go test ./...",
		},
		{
			"path field for read_file",
			`{"path": "tools/web/web.go"}`,
			"path=tools/web/web.go",
		},
		{
			"url field for web_fetch",
			`{"url": "https://example.com/docs", "max_length": 4000}`,
			"url=https://example.com/docs",
		},
		{
			"newlines collapsed onto one line",
			`{"command": "echo hi\n\necho bye"}`,
			"echo hi echo bye",
		},
		{
			"long values truncated",
			`{"command": "` + strings.Repeat("x", 200) + `"}`,
			"…",
		},
		{
			"empty input returns empty",
			``,
			"",
		},
		{
			"unknown shape falls back to key list",
			`{"weird_key": 42}`,
			"weird_key",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toolInputPreview(json.RawMessage(tt.input))
			if tt.want == "" {
				if got != "" {
					t.Errorf("expected empty preview, got %q", got)
				}
				return
			}
			if !strings.Contains(got, tt.want) {
				t.Errorf("preview = %q, want substring %q", got, tt.want)
			}
		})
	}
}

func TestSpinnerLifecycle(t *testing.T) {
	// Force color on so the spinner actually runs (it's a no-op otherwise).
	prev := useColor
	useColor = true
	defer func() { useColor = prev }()

	var buf bytes.Buffer
	var bufMu sync.Mutex
	w := &lockedWriter{w: &buf, mu: &bufMu}

	sp := newSpinner(w)
	sp.Start("thinking…")
	time.Sleep(180 * time.Millisecond) // ~2 ticks at 80ms
	sp.Stop()

	bufMu.Lock()
	out := buf.String()
	bufMu.Unlock()

	if !strings.Contains(out, "thinking") {
		t.Errorf("expected label in output, got %q", out)
	}
	if !strings.Contains(out, "\r") {
		t.Errorf("expected carriage return in spinner output")
	}
	// Stop emits the clear-line escape.
	if !strings.Contains(out, "\x1b[K") {
		t.Errorf("expected clear-line escape after Stop")
	}
}

func TestSpinnerStartIsIdempotent(t *testing.T) {
	prev := useColor
	useColor = true
	defer func() { useColor = prev }()

	sp := newSpinner(&bytes.Buffer{})
	sp.Start("a")
	sp.Start("b") // should just update label, not panic or leak goroutine
	sp.Start("c")
	sp.Stop()
	sp.Stop() // double-stop is a no-op
}

func TestSpinnerNoOpWhenColorOff(t *testing.T) {
	prev := useColor
	useColor = false
	defer func() { useColor = prev }()

	var buf bytes.Buffer
	sp := newSpinner(&buf)
	sp.Start("nope")
	time.Sleep(120 * time.Millisecond)
	sp.Stop()
	if buf.Len() != 0 {
		t.Errorf("expected no output when color is off, got %q", buf.String())
	}
}

// lockedWriter serialises writes so the test can read the buffer mid-flight
// without racing the spinner goroutine.
type lockedWriter struct {
	w  *bytes.Buffer
	mu *sync.Mutex
}

func (l *lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

func TestHumanBytes(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{0, "0 B"},
		{500, "500 B"},
		{1024, "1.0 KiB"},
		{1536, "1.5 KiB"},
		{1024 * 1024, "1.0 MiB"},
		{5 * 1024 * 1024, "5.0 MiB"},
	}
	for _, tt := range tests {
		if got := humanBytes(tt.n); got != tt.want {
			t.Errorf("humanBytes(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}
