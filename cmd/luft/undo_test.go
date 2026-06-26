package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/tools/editor"
)

// wrappedEditor builds a real editor.Toolset wrapped only with the
// editCheckpoint middleware, against a temp dir. Returns the toolset, the
// undo stack, and the dir so tests can seed and assert on-disk state.
func wrappedEditor(t *testing.T) (luft.Toolset, *undoStack, string) {
	t.Helper()
	dir := t.TempDir()
	ed, err := editor.New(editor.Config{Root: dir})
	if err != nil {
		t.Fatalf("editor.New: %v", err)
	}
	stack := newUndoStack(20)
	ts := luft.Tools(ed.Toolset().Bindings...).Wrap(editCheckpoint(stack, dir))
	return ts, stack, dir
}

func callEdit(t *testing.T, ts luft.Toolset, name string, input any) (string, error) {
	t.Helper()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	for _, b := range ts.Bindings {
		if b.Tool.Name == name {
			return b.Func(context.Background(), raw)
		}
	}
	t.Fatalf("tool %q not found in toolset", name)
	return "", nil
}

func writeFile(t *testing.T, path, content string, mode os.FileMode) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), mode); err != nil {
		t.Fatalf("write seed: %v", err)
	}
	if err := os.Chmod(path, mode); err != nil {
		t.Fatalf("chmod: %v", err)
	}
}

func mustNotExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected %q to not exist, got err=%v", path, err)
	}
}

func mustExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected %q to exist, got err=%v", path, err)
	}
}

func readFile(t *testing.T, path string) (string, os.FileMode) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %q: %v", path, err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %q: %v", path, err)
	}
	return string(data), info.Mode().Perm()
}

// TestUndoCreate: create a new file, undo, confirm it is deleted on disk.
func TestUndoCreate(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	stack.beginTurn()
	defer stack.endTurn()

	p := filepath.Join(dir, "a.txt")
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command":   "create",
		"path":      "a.txt",
		"file_text": "hello\n",
	}); err != nil {
		t.Fatalf("create: %v", err)
	}
	mustExist(t, p)

	restored, skipped, nothing, msg := stack.undo()
	if nothing {
		t.Fatalf("expected an undo, got nothing: %s", msg)
	}
	if restored != 1 || skipped != 0 {
		t.Fatalf("restored=%d skipped=%d, want 1/0", restored, skipped)
	}
	mustNotExist(t, p)

	// Stack is now empty.
	_, _, nothing, _ = stack.undo()
	if !nothing {
		t.Fatal("second undo should be nothing-to-undo")
	}
}

// TestUndoStrReplace: modify a pre-existing file (with a non-default mode),
// undo, confirm content AND mode are restored.
func TestUndoStrReplace(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	p := filepath.Join(dir, "b.txt")
	writeFile(t, p, "line one\nline two\n", 0o600)

	stack.beginTurn()
	defer stack.endTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command": "str_replace",
		"path":    "b.txt",
		"old_str": "line one",
		"new_str": "LINE ONE",
	}); err != nil {
		t.Fatalf("str_replace: %v", err)
	}
	got, mode := readFile(t, p)
	if got != "LINE ONE\nline two\n" {
		t.Fatalf("post-edit content = %q", got)
	}

	restored, skipped, nothing, _ := stack.undo()
	if nothing || restored != 1 || skipped != 0 {
		t.Fatalf("undo: restored=%d skipped=%d nothing=%v", restored, skipped, nothing)
	}
	got, mode = readFile(t, p)
	if want := "line one\nline two\n"; got != want {
		t.Fatalf("post-undo content = %q, want %q", got, want)
	}
	// Mode restoration is only meaningful on POSIX; Windows has no perm bits.
	if runtime.GOOS != "windows" && mode != 0o600 {
		t.Fatalf("post-undo mode = %o, want 0600", mode)
	}
}

// TestUndoInsert: insert lines, undo, confirm original restored.
func TestUndoInsert(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	p := filepath.Join(dir, "c.txt")
	writeFile(t, p, "a\nb\n", 0o644)

	stack.beginTurn()
	defer stack.endTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command":     "insert",
		"path":        "c.txt",
		"new_str":     "INSERTED",
		"insert_line": 1,
	}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	got, _ := readFile(t, p)
	if want := "a\nINSERTED\nb\n"; got != want {
		t.Fatalf("post-insert content = %q, want %q", got, want)
	}

	restored, _, nothing, _ := stack.undo()
	if nothing || restored != 1 {
		t.Fatalf("undo: restored=%d nothing=%v", restored, nothing)
	}
	got, _ = readFile(t, p)
	if want := "a\nb\n"; got != want {
		t.Fatalf("post-undo content = %q, want %q", got, want)
	}
}

// TestUndoMultiEdit: a single multi_edit produces one undo entry that reverts
// all applied edits.
func TestUndoMultiEdit(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	p := filepath.Join(dir, "d.txt")
	writeFile(t, p, "foo\nbar\nbaz\n", 0o644)

	stack.beginTurn()
	defer stack.endTurn()
	if _, err := callEdit(t, ts, "multi_edit", map[string]any{
		"path": "d.txt",
		"edits": []map[string]string{
			{"old_str": "foo", "new_str": "FOO"},
			{"old_str": "baz", "new_str": "BAZ"},
		},
	}); err != nil {
		t.Fatalf("multi_edit: %v", err)
	}
	got, _ := readFile(t, p)
	if want := "FOO\nbar\nBAZ\n"; got != want {
		t.Fatalf("post-multiedit content = %q, want %q", got, want)
	}

	restored, skipped, nothing, _ := stack.undo()
	if nothing || restored != 1 || skipped != 0 {
		t.Fatalf("undo: restored=%d skipped=%d nothing=%v (want 1 entry for the whole multi_edit)", restored, skipped, nothing)
	}
	got, _ = readFile(t, p)
	if want := "foo\nbar\nbaz\n"; got != want {
		t.Fatalf("post-undo content = %q, want %q", got, want)
	}
}

// TestUndoMultipleEditsOneTurn: two edits in one turn are reverted together.
func TestUndoMultipleEditsOneTurn(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	created := filepath.Join(dir, "e.txt")
	modified := filepath.Join(dir, "f.txt")
	writeFile(t, modified, "original\n", 0o644)

	stack.beginTurn()
	defer stack.endTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command":   "create",
		"path":      "e.txt",
		"file_text": "new file\n",
	}); err != nil {
		t.Fatalf("create e.txt: %v", err)
	}
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command": "str_replace",
		"path":    "f.txt",
		"old_str": "original",
		"new_str": "CHANGED",
	}); err != nil {
		t.Fatalf("str_replace f.txt: %v", err)
	}

	restored, skipped, nothing, _ := stack.undo()
	if nothing || restored != 2 || skipped != 0 {
		t.Fatalf("undo: restored=%d skipped=%d nothing=%v, want 2/0/false", restored, skipped, nothing)
	}
	mustNotExist(t, created)
	got, _ := readFile(t, modified)
	if want := "original\n"; got != want {
		t.Fatalf("f.txt after undo = %q, want %q", got, want)
	}
}

// TestMultiLevelUndo: two turns; undo reverts the second, then the first,
// then reports nothing.
func TestMultiLevelUndo(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	g := filepath.Join(dir, "g.txt")
	h := filepath.Join(dir, "h.txt")

	// Turn 1: create g.txt.
	stack.beginTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command":   "create",
		"path":      "g.txt",
		"file_text": "G\n",
	}); err != nil {
		t.Fatalf("create g.txt: %v", err)
	}
	stack.endTurn()

	// Turn 2: create h.txt.
	stack.beginTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command":   "create",
		"path":      "h.txt",
		"file_text": "H\n",
	}); err != nil {
		t.Fatalf("create h.txt: %v", err)
	}
	stack.endTurn()

	mustExist(t, g)
	mustExist(t, h)

	// Undo turn 2: h gone, g still there.
	restored, _, nothing, _ := stack.undo()
	if nothing || restored != 1 {
		t.Fatalf("undo1: restored=%d nothing=%v", restored, nothing)
	}
	mustNotExist(t, h)
	mustExist(t, g)

	// Undo turn 1: g gone.
	restored, _, nothing, _ = stack.undo()
	if nothing || restored != 1 {
		t.Fatalf("undo2: restored=%d nothing=%v", restored, nothing)
	}
	mustNotExist(t, g)

	_, _, nothing, _ = stack.undo()
	if !nothing {
		t.Fatal("third undo should be nothing-to-undo")
	}
}

// TestUndoClobberSafetySkip: a file modified on disk after the agent's edit
// (e.g. via bash) is NOT clobbered by undo.
func TestUndoClobberSafetySkip(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	p := filepath.Join(dir, "i.txt")

	stack.beginTurn()
	defer stack.endTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command":   "create",
		"path":      "i.txt",
		"file_text": "agent made this\n",
	}); err != nil {
		t.Fatalf("create: %v", err)
	}
	// Simulate a post-edit external modification (bash/user).
	if err := os.WriteFile(p, []byte("user clobbered this\n"), 0o644); err != nil {
		t.Fatalf("clobber: %v", err)
	}

	restored, skipped, nothing, msg := stack.undo()
	if nothing {
		t.Fatalf("expected a skip, got nothing: %s", msg)
	}
	if restored != 0 || skipped != 1 {
		t.Fatalf("restored=%d skipped=%d, want 0/1 (must not delete a file that changed underneath us)", restored, skipped)
	}
	if !strings.Contains(msg, "changed on disk") {
		t.Fatalf("undo msg should mention on-disk change, got: %s", msg)
	}
	// File is preserved with the user's content.
	got, _ := readFile(t, p)
	if want := "user clobbered this\n"; got != want {
		t.Fatalf("clobbered file content = %q, want %q", got, want)
	}
}

// TestUndoViewNotRecorded: a view command is read-only and must not produce
// an undo entry (the turn frame is empty and trimmed).
func TestUndoViewNotRecorded(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	p := filepath.Join(dir, "v.txt")
	writeFile(t, p, "content\n", 0o644)

	stack.beginTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command": "view",
		"path":    "v.txt",
	}); err != nil {
		t.Fatalf("view: %v", err)
	}
	stack.endTurn()

	_, _, nothing, _ := stack.undo()
	if !nothing {
		t.Fatal("view should not be recorded; undo should report nothing")
	}
}

// TestUndoFailedEditNotRecorded: an edit that fails (old_str not found) must
// not push an undo entry.
func TestUndoFailedEditNotRecorded(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	p := filepath.Join(dir, "fail.txt")
	writeFile(t, p, "unchanged\n", 0o644)

	stack.beginTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command": "str_replace",
		"path":    "fail.txt",
		"old_str": "DOES NOT EXIST",
		"new_str": "x",
	}); err == nil {
		t.Fatal("expected str_replace to fail on missing old_str")
	}
	stack.endTurn()

	_, _, nothing, _ := stack.undo()
	if !nothing {
		t.Fatal("failed edit should not be recorded; undo should report nothing")
	}
	// File untouched.
	got, _ := readFile(t, p)
	if want := "unchanged\n"; got != want {
		t.Fatalf("file content after failed edit = %q, want %q", got, want)
	}
}

// TestUndoEmptyStackNotice: a fresh stack with no turns reports nothing-to-undo.
func TestUndoEmptyStackNotice(t *testing.T) {
	_, stack, _ := wrappedEditor(t)
	_, _, nothing, msg := stack.undo()
	if !nothing {
		t.Fatal("fresh stack should report nothing-to-undo")
	}
	if !strings.Contains(msg, "nothing to undo") {
		t.Fatalf("msg should mention nothing-to-undo, got: %s", msg)
	}
}

// TestUndoPathNormalization: the trained tool emits a leading slash; the
// snapshot must resolve under the root so undo deletes the right file.
func TestUndoPathNormalization(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)
	sub := filepath.Join(dir, "sub")
	p := filepath.Join(sub, "x.txt")

	stack.beginTurn()
	defer stack.endTurn()
	if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
		"command":   "create",
		"path":      "/sub/x.txt", // leading slash, trained-tool style
		"file_text": "nested\n",
	}); err != nil {
		t.Fatalf("create: %v", err)
	}
	mustExist(t, p)

	restored, _, nothing, _ := stack.undo()
	if nothing || restored != 1 {
		t.Fatalf("undo: restored=%d nothing=%v", restored, nothing)
	}
	mustNotExist(t, p)
}

// TestUndoDepthBound: the stack trims to its max depth, dropping the oldest.
func TestUndoDepthBound(t *testing.T) {
	ts, stack, dir := wrappedEditor(t)

	// stack.max is 20; create 21 turns, each with one file.
	for i := 0; i < stack.max+1; i++ {
		stack.beginTurn()
		if _, err := callEdit(t, ts, "str_replace_based_edit_tool", map[string]any{
			"command":   "create",
			"path":      fmt.Sprintf("f%d.txt", i),
			"file_text": fmt.Sprintf("file %d\n", i),
		}); err != nil {
			t.Fatalf("create f%d: %v", i, err)
		}
		stack.endTurn()
	}

	// The oldest frame (f0) should have been trimmed; only the most recent
	// `max` turns remain. Undoing `max` times clears the stack; the file from
	// the trimmed turn (f0) is still on disk (unrevertible — by design).
	mustExist(t, filepath.Join(dir, "f0.txt"))
	for i := 0; i < stack.max; i++ {
		restored, _, nothing, _ := stack.undo()
		if nothing {
			t.Fatalf("undo %d: expected a revert, got nothing", i)
		}
		if restored != 1 {
			t.Fatalf("undo %d: restored=%d, want 1", i, restored)
		}
	}
	_, _, nothing, _ := stack.undo()
	if !nothing {
		t.Fatal("after depth-bound undos, stack should be empty")
	}
	mustExist(t, filepath.Join(dir, "f0.txt")) // trimmed frame, unrevertible
}
