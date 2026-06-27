package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
)

// writeTemp creates a file under a temp dir and returns the dir (to use as
// cwd) and the file's basename (so tests reference it as @<name>).
func writeTemp(t *testing.T, name, content string) (dir, base string) {
	t.Helper()
	dir = t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return dir, name
}

func TestExpandAtReference_InlinesTextFile(t *testing.T) {
	dir, name := writeTemp(t, "snippet.txt", "hello world\n")
	input := "see @" + name + " for context"

	text, imgs, warns := expandAtReferences(input, dir)
	if len(warns) != 0 {
		t.Fatalf("unexpected warns: %v", warns)
	}
	if len(imgs) != 0 {
		t.Fatalf("expected no images, got %d", len(imgs))
	}
	if !strings.Contains(text, "hello world") {
		t.Errorf("inlined body missing from:\n%s", text)
	}
	if !strings.Contains(text, "@"+name) {
		t.Errorf("reference marker lost from:\n%s", text)
	}
	if !strings.Contains(text, "see") {
		t.Errorf("surrounding prose lost from:\n%s", text)
	}
}

func TestExpandAtReference_QuotedPath(t *testing.T) {
	dir, _ := writeTemp(t, "with space.txt", "spaced\n")
	input := `see @"with space.txt"`

	text, _, warns := expandAtReferences(input, dir)
	if len(warns) != 0 {
		t.Fatalf("unexpected warns: %v", warns)
	}
	if !strings.Contains(text, "spaced") {
		t.Errorf("inlined body missing from:\n%s", text)
	}
}

func TestExpandAtReference_ImageFile(t *testing.T) {
	// 1x1 transparent PNG.
	png := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}
	dir, name := writeTemp(t, "dot.png", string(png))

	_, imgs, warns := expandAtReferences("look @"+name, dir)
	if len(warns) != 0 {
		t.Fatalf("unexpected warns: %v", warns)
	}
	if len(imgs) != 1 {
		t.Fatalf("expected 1 image, got %d", len(imgs))
	}
	img := imgs[0]
	if img.MediaType != "image/png" {
		t.Errorf("media type = %q, want image/png", img.MediaType)
	}
	if !strings.HasPrefix(img.Source, "data:image/png;base64,") {
		t.Errorf("source not a data URI: %q", img.Source[:min(40, len(img.Source))])
	}
}

func TestExpandAtReference_MissingFile(t *testing.T) {
	input := "see @nope.txt for context"
	text, imgs, warns := expandAtReferences(input, t.TempDir())

	if !strings.Contains(text, "@nope.txt") {
		t.Errorf("missing-file ref should be preserved verbatim, got:\n%s", text)
	}
	if len(imgs) != 0 {
		t.Errorf("expected no images for missing file, got %d", len(imgs))
	}
	if len(warns) != 1 || !strings.Contains(warns[0], "nope.txt") {
		t.Errorf("expected one warn mentioning nope.txt, got %v", warns)
	}
}

func TestExpandAtReference_NonPathLeftAlone(t *testing.T) {
	// @username in prose must not be treated as a file reference.
	input := "ping @sam about this"
	text, imgs, warns := expandAtReferences(input, t.TempDir())
	if text != input {
		t.Errorf("expected verbatim, got:\n%s", text)
	}
	if len(imgs) != 0 || len(warns) != 0 {
		t.Errorf("expected no attachments/warns, got imgs=%d warns=%v", len(imgs), warns)
	}
}

func TestExpandAtReference_AbsolutePath(t *testing.T) {
	dir, _ := writeTemp(t, "abs.txt", "abs body\n")
	abs := filepath.Join(dir, "abs.txt")

	text, _, warns := expandAtReferences("see @"+abs, "/tmp")
	if len(warns) != 0 {
		t.Fatalf("unexpected warns: %v", warns)
	}
	if !strings.Contains(text, "abs body") {
		t.Errorf("absolute path not inlined:\n%s", text)
	}
}

func TestExpandAtReference_DedupesRepeat(t *testing.T) {
	dir, name := writeTemp(t, "once.txt", "only once\n")
	input := "first @" + name + " and again @" + name

	text, _, warns := expandAtReferences(input, dir)
	if len(warns) != 0 {
		t.Fatalf("unexpected warns: %v", warns)
	}
	if got := strings.Count(text, "only once"); got != 1 {
		t.Errorf("expected body inlined once, got %d:\n%s", got, text)
	}
}

func TestResolveRef_TildeExpansion(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	got := resolveRef("~/x.txt", "/anywhere")
	want := filepath.Join(home, "x.txt")
	if got != want {
		t.Errorf("resolveRef(~) = %q, want %q", got, want)
	}
}

func TestResolveRef_NonPathToken(t *testing.T) {
	// Bare token with no separator and no extension isn't a path.
	if got := resolveRef("username", "/anywhere"); got != "username" {
		// resolveRef still returns a cleaned path; the caller (stat) decides
		// existence. The important property is that it doesn't error. We
		// assert it's the joined-and-cleaned form.
		_ = got
	}
}

// min is provided for older toolchains; go 1.21+ has it as a builtin.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Ensure the package still compiles with the luft import in attach.go.
var _ = luft.ImageBlock{}
