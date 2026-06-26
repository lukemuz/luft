package editor

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func mkEditor(t *testing.T) (*Editor, string) {
	t.Helper()
	dir := t.TempDir()
	e, err := New(Config{Root: dir})
	if err != nil {
		t.Fatal(err)
	}
	return e, dir
}

func call(t *testing.T, e *Editor, in editorInput) (string, error) {
	t.Helper()
	raw, _ := json.Marshal(in)
	return e.Handler()(context.Background(), raw)
}

func TestCreateAndView(t *testing.T) {
	e, dir := mkEditor(t)
	out, err := call(t, e, editorInput{Command: "create", Path: "hello.txt", FileText: "line1\nline2\n"})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if !strings.Contains(out, "created") {
		t.Fatalf("unexpected: %q", out)
	}
	body, _ := os.ReadFile(filepath.Join(dir, "hello.txt"))
	if string(body) != "line1\nline2\n" {
		t.Fatalf("file body: %q", body)
	}

	view, err := call(t, e, editorInput{Command: "view", Path: "hello.txt"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(view, "line1") || !strings.Contains(view, "line2") {
		t.Fatalf("view: %q", view)
	}
}

func TestMultiEditAppliesAllInOrder(t *testing.T) {
	e, dir := mkEditor(t)
	if _, err := call(t, e, editorInput{Command: "create", Path: "f.txt", FileText: "alpha beta gamma"}); err != nil {
		t.Fatal(err)
	}
	in := multiEditInput{Path: "f.txt"}
	in.Edits = append(in.Edits,
		struct {
			OldStr string `json:"old_str"`
			NewStr string `json:"new_str"`
		}{OldStr: "alpha", NewStr: "ALPHA"},
		struct {
			OldStr string `json:"old_str"`
			NewStr string `json:"new_str"`
		}{OldStr: "ALPHA beta", NewStr: "ALPHA-BETA"}, // depends on the first edit's result
	)
	out, err := e.multiEdit(in)
	if err != nil {
		t.Fatalf("multiEdit: %v", err)
	}
	if !strings.Contains(out, "2 replacements") {
		t.Fatalf("unexpected summary: %q", out)
	}
	body, _ := os.ReadFile(filepath.Join(dir, "f.txt"))
	if string(body) != "ALPHA-BETA gamma" {
		t.Fatalf("file body: %q", body)
	}
}

func TestMultiEditAtomicOnFailure(t *testing.T) {
	e, dir := mkEditor(t)
	if _, err := call(t, e, editorInput{Command: "create", Path: "g.txt", FileText: "one two three"}); err != nil {
		t.Fatal(err)
	}
	in := multiEditInput{Path: "g.txt"}
	in.Edits = append(in.Edits,
		struct {
			OldStr string `json:"old_str"`
			NewStr string `json:"new_str"`
		}{OldStr: "one", NewStr: "1"},
		struct {
			OldStr string `json:"old_str"`
			NewStr string `json:"new_str"`
		}{OldStr: "MISSING", NewStr: "x"}, // fails — whole op must abort
	)
	if _, err := e.multiEdit(in); err == nil {
		t.Fatal("expected error when an edit does not match")
	}
	body, _ := os.ReadFile(filepath.Join(dir, "g.txt"))
	if string(body) != "one two three" {
		t.Fatalf("file should be untouched, got: %q", body)
	}
}

// A CRLF file (the norm on Windows) edited with an LF multi-line old_str must
// still match: models and JSON args use LF, so without newline reconciliation
// every multi-line edit on such a file would spuriously fail.
func TestStrReplaceMatchesAcrossCRLF(t *testing.T) {
	e, dir := mkEditor(t)
	crlf := "func Add(a, b int) int {\r\n\treturn a - b\r\n}\r\n"
	if err := os.WriteFile(filepath.Join(dir, "c.go"), []byte(crlf), 0o644); err != nil {
		t.Fatal(err)
	}
	// old_str/new_str use LF, as a model would emit them.
	if _, err := call(t, e, editorInput{
		Command: "str_replace",
		Path:    "c.go",
		OldStr:  "func Add(a, b int) int {\n\treturn a - b\n}",
		NewStr:  "func Add(a, b int) int {\n\treturn a + b\n}",
	}); err != nil {
		t.Fatalf("str_replace across CRLF: %v", err)
	}
	body, _ := os.ReadFile(filepath.Join(dir, "c.go"))
	want := "func Add(a, b int) int {\r\n\treturn a + b\r\n}\r\n"
	if string(body) != want {
		t.Fatalf("CRLF convention not preserved or edit wrong:\n got %q\nwant %q", body, want)
	}
}

func TestMultiEditRejectsNonUnique(t *testing.T) {
	e, dir := mkEditor(t)
	if _, err := call(t, e, editorInput{Command: "create", Path: "h.txt", FileText: "x x x"}); err != nil {
		t.Fatal(err)
	}
	in := multiEditInput{Path: "h.txt"}
	in.Edits = append(in.Edits, struct {
		OldStr string `json:"old_str"`
		NewStr string `json:"new_str"`
	}{OldStr: "x", NewStr: "y"})
	if _, err := e.multiEdit(in); err == nil {
		t.Fatal("expected non-unique old_str to be rejected")
	}
	body, _ := os.ReadFile(filepath.Join(dir, "h.txt"))
	if string(body) != "x x x" {
		t.Fatalf("file should be untouched, got: %q", body)
	}
}

func TestCreateRefusesOverwrite(t *testing.T) {
	e, _ := mkEditor(t)
	if _, err := call(t, e, editorInput{Command: "create", Path: "a.txt", FileText: "x"}); err != nil {
		t.Fatal(err)
	}
	if _, err := call(t, e, editorInput{Command: "create", Path: "a.txt", FileText: "y"}); err == nil {
		t.Fatal("expected overwrite refusal")
	}
}

func TestStrReplace(t *testing.T) {
	e, dir := mkEditor(t)
	os.WriteFile(filepath.Join(dir, "f.go"), []byte("package main\n\nfunc Foo() {}\n"), 0o644)

	if _, err := call(t, e, editorInput{Command: "str_replace", Path: "f.go", OldStr: "Foo", NewStr: "Bar"}); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(filepath.Join(dir, "f.go"))
	if !strings.Contains(string(body), "Bar()") {
		t.Fatalf("body: %q", body)
	}
}

func TestStrReplaceRejectsNonUnique(t *testing.T) {
	e, dir := mkEditor(t)
	os.WriteFile(filepath.Join(dir, "f.go"), []byte("x\nx\n"), 0o644)
	if _, err := call(t, e, editorInput{Command: "str_replace", Path: "f.go", OldStr: "x", NewStr: "y"}); err == nil {
		t.Fatal("expected non-unique error")
	}
}

func TestInsert(t *testing.T) {
	e, dir := mkEditor(t)
	os.WriteFile(filepath.Join(dir, "f.txt"), []byte("a\nb\nc\n"), 0o644)
	at := 1
	if _, err := call(t, e, editorInput{Command: "insert", Path: "f.txt", InsertLine: &at, NewStr: "INS"}); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(filepath.Join(dir, "f.txt"))
	if !strings.Contains(string(body), "a\nINS\nb\n") {
		t.Fatalf("body: %q", body)
	}
}

func TestPathEscape(t *testing.T) {
	e, _ := mkEditor(t)
	if _, err := call(t, e, editorInput{Command: "view", Path: "../../etc/passwd"}); err == nil {
		t.Fatal("expected path escape rejection")
	}
}

func TestViewDirectory(t *testing.T) {
	e, dir := mkEditor(t)
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("x"), 0o644)
	os.Mkdir(filepath.Join(dir, "sub"), 0o755)
	out, err := call(t, e, editorInput{Command: "view", Path: "."})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "a.txt") || !strings.Contains(out, "sub/") {
		t.Fatalf("dir listing: %q", out)
	}
}
