package workspace_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lukemuz/luft/tools/workspace"
)

// TestGlobDoubleStar exercises the new ** support and verifies that
// segment-aware patterns work against full relative paths.
func TestGlobDoubleStar(t *testing.T) {
	root := t.TempDir()
	mustWrite := func(rel, body string) {
		t.Helper()
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite("a.go", "")
	mustWrite("pkg/b.go", "")
	mustWrite("pkg/sub/c.go", "")
	mustWrite("pkg/sub/d.txt", "")
	mustWrite("internal/e.go", "")

	ws := newWS(t, root)
	got := func(pattern string) []string {
		raw := callTool(t, ws, "Glob", map[string]string{"pattern": pattern})
		var out []string
		_ = json.Unmarshal([]byte(raw), &out)
		return out
	}

	// Basename-only pattern (legacy behaviour).
	if want, names := 4, got("*.go"); len(names) != want {
		t.Errorf("`*.go` matched %d files, want %d (got %v)", len(names), want, names)
	}
	// ** at any depth.
	if want, names := 4, got("**/*.go"); len(names) != want {
		t.Errorf("`**/*.go` matched %d files, want %d (got %v)", len(names), want, names)
	}
	// Anchored prefix + ** + extension.
	pkgGo := got("pkg/**/*.go")
	if len(pkgGo) != 2 {
		t.Errorf("`pkg/**/*.go` matched %d files, want 2 (got %v)", len(pkgGo), pkgGo)
	}
	for _, p := range pkgGo {
		if !strings.HasPrefix(p, "pkg/") {
			t.Errorf("`pkg/**/*.go` returned non-pkg path %q", p)
		}
	}
}

// TestSkipListPrunesNoise verifies node_modules / .git / vendor are
// pruned by default during Glob and Grep.
func TestSkipListPrunesNoise(t *testing.T) {
	root := t.TempDir()
	mustWrite := func(rel, body string) {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite("src/main.go", "package main\n// TODO real")
	mustWrite("node_modules/lodash/index.js", "// TODO noise")
	mustWrite(".git/HEAD", "ref: refs/heads/main // TODO noise")
	mustWrite("vendor/dep/dep.go", "package dep\n// TODO noise")

	ws := newWS(t, root)

	rawGlob := callTool(t, ws, "Glob", map[string]string{"pattern": "**/*"})
	var globHits []string
	_ = json.Unmarshal([]byte(rawGlob), &globHits)
	for _, h := range globHits {
		if strings.HasPrefix(h, "node_modules/") || strings.HasPrefix(h, ".git/") || strings.HasPrefix(h, "vendor/") {
			t.Errorf("Glob walked into skipped dir: %q", h)
		}
	}

	rawGrep := callTool(t, ws, "Grep", map[string]string{"pattern": "TODO"})
	var grepHits []struct{ File, Text string }
	_ = json.Unmarshal([]byte(rawGrep), &grepHits)
	if len(grepHits) != 1 {
		t.Fatalf("Grep hit count = %d, want 1 (only src/main.go); got %+v", len(grepHits), grepHits)
	}
	if grepHits[0].File != "src/main.go" {
		t.Errorf("Grep hit was %q, want src/main.go", grepHits[0].File)
	}
}

// TestSkipListAllowsExplicitDescent — if a user explicitly searches
// inside a normally-skipped dir, the rule should not apply to the base.
func TestSkipListAllowsExplicitDescent(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "node_modules", "x"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "node_modules", "x", "y.js"), []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}

	ws := newWS(t, root)
	raw := callTool(t, ws, "Grep", map[string]any{"pattern": "hello", "path": "node_modules"})
	var hits []struct{ File string }
	_ = json.Unmarshal([]byte(raw), &hits)
	if len(hits) != 1 {
		t.Errorf("expected hit when explicitly searching node_modules, got %d (%+v)", len(hits), hits)
	}
}

// TestGrepSkipsBinaryFiles verifies the NUL-sniff and extension allow-list.
func TestGrepSkipsBinaryFiles(t *testing.T) {
	root := t.TempDir()
	// Binary by extension.
	if err := os.WriteFile(filepath.Join(root, "logo.png"), []byte("MATCH-needle"), 0644); err != nil {
		t.Fatal(err)
	}
	// Binary by NUL byte (no known extension).
	nul := append([]byte("MATCH-needle"), 0x00, 'x')
	if err := os.WriteFile(filepath.Join(root, "blob"), nul, 0644); err != nil {
		t.Fatal(err)
	}
	// Real text file.
	if err := os.WriteFile(filepath.Join(root, "notes.txt"), []byte("MATCH-needle here"), 0644); err != nil {
		t.Fatal(err)
	}

	ws := newWS(t, root)
	raw := callTool(t, ws, "Grep", map[string]string{"pattern": "MATCH-needle"})
	var hits []struct{ File string }
	_ = json.Unmarshal([]byte(raw), &hits)
	if len(hits) != 1 || hits[0].File != "notes.txt" {
		t.Errorf("expected single hit in notes.txt, got %+v", hits)
	}
}

// TestGrepLiteralFastPath — make sure pure-literal patterns still
// produce correct matches (the fast path is correctness-equivalent
// to the regex path; this just guards against regression).
func TestGrepLiteralFastPath(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "f.txt"), []byte("hello world\nfoo bar\nbaz\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ws := newWS(t, root)
	raw := callTool(t, ws, "Grep", map[string]string{"pattern": "foo bar"})
	var hits []struct {
		File string
		Line int
		Text string
	}
	_ = json.Unmarshal([]byte(raw), &hits)
	if len(hits) != 1 || hits[0].Text != "foo bar" || hits[0].Line != 2 {
		t.Errorf("expected one hit for literal 'foo bar' on line 2, got %+v", hits)
	}
}

// TestGrepInvalidRegexFallsBackToLiteral — a pattern that contains
// regex metacharacters but won't compile as a regex (e.g. "Trim("
// with an unbalanced paren) should be searched for literally rather
// than returning a hard error.
func TestGrepInvalidRegexFallsBackToLiteral(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "f.go"), []byte("func foo() { strings.Trim(s, \"x\") }\nbar\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ws := newWS(t, root)
	raw := callTool(t, ws, "Grep", map[string]string{"pattern": "Trim("})
	if strings.HasPrefix(raw, "Error") || strings.Contains(raw, "invalid pattern") {
		t.Fatalf("Grep returned an error for invalid-regex pattern: %s", raw)
	}
	var hits []struct {
		File string
		Line int
		Text string
	}
	if err := json.Unmarshal([]byte(raw), &hits); err != nil {
		t.Fatalf("failed to parse Grep result %q: %v", raw, err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected 1 hit, got %d (%+v)", len(hits), hits)
	}
	if !strings.Contains(hits[0].Text, "Trim(") {
		t.Errorf("hit text %q does not contain 'Trim('", hits[0].Text)
	}
}

// TestGrepStableOrder verifies that with parallel workers, output is
// still ordered (file ascending, line ascending). The model and the
// agent loop both benefit from deterministic results.
func TestGrepStableOrder(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"a.txt", "b.txt", "c.txt", "d.txt"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("x\nx\nx\n"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	ws := newWS(t, root)
	raw := callTool(t, ws, "Grep", map[string]string{"pattern": "x"})
	var hits []struct {
		File string
		Line int
	}
	_ = json.Unmarshal([]byte(raw), &hits)
	// 4 files * 3 lines = 12 hits.
	if len(hits) != 12 {
		t.Fatalf("expected 12 hits, got %d", len(hits))
	}
	for i := 1; i < len(hits); i++ {
		prev, cur := hits[i-1], hits[i]
		if prev.File > cur.File || (prev.File == cur.File && prev.Line >= cur.Line) {
			t.Errorf("results not stably ordered at index %d: %+v then %+v", i, prev, cur)
		}
	}
}

// TestGrepTimedSanity is a sanity-check timing test, not a strict
// benchmark. We build a synthetic tree of N small text files plus a
// matching number of fake-bloat node_modules files and confirm that
// Grep completes "fast enough" — i.e. that the parallel walk + skip
// list + literal fast path keep us in interactive territory.
//
// Threshold is generous (1 second for ~2k real files) so the test
// stays stable on slower CI machines. Times log with -v for visibility.
func TestGrepTimedSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping timed sanity test in -short mode")
	}
	root := t.TempDir()

	// Real source files: 50 dirs * 40 files = 2000 files containing
	// a needle on a known line.
	const realDirs = 50
	const realPerDir = 40
	body := strings.Repeat("// filler line\n", 80) +
		"NEEDLE marker line\n" +
		strings.Repeat("// trailing\n", 80)
	for i := 0; i < realDirs; i++ {
		dir := filepath.Join(root, "src", fmt.Sprintf("d%02d", i))
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < realPerDir; j++ {
			if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%02d.go", j)), []byte(body), 0644); err != nil {
				t.Fatal(err)
			}
		}
	}

	// Bloat: 100 dirs * 100 files inside node_modules. Each contains
	// the needle, but should be skipped — if the skip list ever
	// regresses, the result count will jump and timing will explode.
	const bloatDirs = 100
	const bloatPerDir = 100
	for i := 0; i < bloatDirs; i++ {
		dir := filepath.Join(root, "node_modules", fmt.Sprintf("pkg%02d", i))
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < bloatPerDir; j++ {
			if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("x%02d.js", j)), []byte("NEEDLE noise\n"), 0644); err != nil {
				t.Fatal(err)
			}
		}
	}

	totalFiles := realDirs*realPerDir + bloatDirs*bloatPerDir
	t.Logf("synthetic tree: %d real .go files, %d bloat node_modules files (%d total)", realDirs*realPerDir, bloatDirs*bloatPerDir, totalFiles)

	// Use a 5000-result cap so we count everything real (real files
	// = 2000 < 5000) and verify the bloat is skipped.
	ws, err := workspace.NewReadOnly(workspace.Config{Root: root, MaxResults: 5000})
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	raw := callTool(t, ws, "Grep", map[string]string{"pattern": "NEEDLE"})
	elapsed := time.Since(start)

	var hits []struct{ File string }
	if err := json.Unmarshal([]byte(raw), &hits); err != nil {
		t.Fatal(err)
	}

	t.Logf("Grep returned %d hits in %s", len(hits), elapsed)

	// Should be exactly realDirs*realPerDir hits — bloat skipped.
	if len(hits) != realDirs*realPerDir {
		t.Errorf("hit count = %d, want %d (skip list may have regressed)", len(hits), realDirs*realPerDir)
	}

	// Sanity threshold. On a developer laptop this completes in ~30-80ms.
	// We give 1s for CI so the test isn't flaky on slow runners.
	if elapsed > 1*time.Second {
		t.Errorf("Grep took %s, expected < 1s for %d files (parallel/skip path may have regressed)", elapsed, totalFiles)
	}
}

// TestGlobMatchUnit tests the matcher function in isolation.
func TestGlobMatchUnit(t *testing.T) {
	cases := []struct {
		pattern string
		path    string
		want    bool
	}{
		{"*.go", "foo.go", true},
		{"*.go", "pkg/foo.go", true}, // basename pattern matches in nested path
		{"*.go", "foo.txt", false},
		{"**/*.go", "foo.go", true},
		{"**/*.go", "a/b/c/foo.go", true},
		{"**/*.go", "foo.txt", false},
		{"pkg/**/*.go", "pkg/foo.go", true},
		{"pkg/**/*.go", "pkg/a/b/foo.go", true},
		{"pkg/**/*.go", "other/foo.go", false},
		{"pkg/*.go", "pkg/foo.go", true},
		{"pkg/*.go", "pkg/sub/foo.go", false}, // single * doesn't cross /
		{"a/**", "a", true},                   // ** matches zero segments
		{"a/**", "a/b/c", true},
	}
	for _, c := range cases {
		// We can't access globMatch directly from the _test package, so
		// exercise it through the Glob tool. Build a tiny tree.
		t.Run(fmt.Sprintf("%s_vs_%s", c.pattern, c.path), func(t *testing.T) {
			root := t.TempDir()
			full := filepath.Join(root, filepath.FromSlash(c.path))
			if err := os.MkdirAll(filepath.Dir(full), 0755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(full, []byte(""), 0644); err != nil {
				t.Fatal(err)
			}
			ws := newWS(t, root)
			raw := callTool(t, ws, "Glob", map[string]string{"pattern": c.pattern})
			var hits []string
			_ = json.Unmarshal([]byte(raw), &hits)
			matched := false
			for _, h := range hits {
				if h == c.path {
					matched = true
					break
				}
			}
			if matched != c.want {
				t.Errorf("globMatch(%q, %q) hits=%v want=%v", c.pattern, c.path, hits, c.want)
			}
		})
	}
}
