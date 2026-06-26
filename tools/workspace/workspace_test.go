package workspace_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/lukemuz/luft/tools/workspace"
)

// setupTestDir creates a temporary directory with a small tree for tests.
//
//	root/
//	  hello.txt       "hello world\n"
//	  subdir/
//	    foo.go        "package foo\n\nfunc Foo() {}\n"
//	    bar.go        "package foo\n\nfunc Bar() {}\n"
func setupTestDir(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	must(os.WriteFile(filepath.Join(root, "hello.txt"), []byte("hello world\n"), 0644))
	must(os.MkdirAll(filepath.Join(root, "subdir"), 0755))
	must(os.WriteFile(filepath.Join(root, "subdir", "foo.go"), []byte("package foo\n\nfunc Foo() {}\n"), 0644))
	must(os.WriteFile(filepath.Join(root, "subdir", "bar.go"), []byte("package foo\n\nfunc Bar() {}\n"), 0644))
	return root
}

func newWS(t *testing.T, root string) *workspace.Workspace {
	t.Helper()
	ws, err := workspace.NewReadOnly(workspace.Config{Root: root})
	if err != nil {
		t.Fatalf("NewReadOnly: %v", err)
	}
	return ws
}

func callTool(t *testing.T, ws *workspace.Workspace, name string, args any) string {
	t.Helper()
	dispatch := ws.Dispatch()
	fn, ok := dispatch[name]
	if !ok {
		t.Fatalf("dispatch missing %q", name)
	}
	raw, _ := json.Marshal(args)
	out, err := fn(context.Background(), raw)
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", name, err)
	}
	return out
}

func TestNewReadOnlyRequiresRoot(t *testing.T) {
	_, err := workspace.NewReadOnly(workspace.Config{})
	if err == nil {
		t.Fatal("want error for empty root, got nil")
	}
}

func TestToolsetHasFiveBindings(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	ts := ws.Toolset()
	if len(ts.Bindings) != 5 {
		t.Errorf("want 5 bindings, got %d", len(ts.Bindings))
	}
}

func TestListDirectory(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	out := callTool(t, ws, "list_directory", map[string]any{"path": "."})
	var entries []struct {
		Name  string `json:"name"`
		IsDir bool   `json:"is_dir"`
	}
	if err := json.Unmarshal([]byte(out), &entries); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, out)
	}
	names := make(map[string]bool)
	for _, e := range entries {
		names[e.Name] = true
	}
	if !names["hello.txt"] {
		t.Error("expected hello.txt in listing")
	}
	if !names["subdir"] {
		t.Error("expected subdir in listing")
	}
}

func TestListDirectoryDefault(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	// empty path should default to root
	out := callTool(t, ws, "list_directory", map[string]any{})
	if !strings.Contains(out, "hello.txt") {
		t.Errorf("expected hello.txt in output, got: %s", out)
	}
}

func TestListDirectoryTraversalRejected(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	dispatch := ws.Dispatch()
	raw, _ := json.Marshal(map[string]any{"path": "../"})
	_, err := dispatch["list_directory"](context.Background(), raw)
	if err == nil {
		t.Fatal("want error for path traversal, got nil")
	}
}

func TestFindFiles(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	out := callTool(t, ws, "Glob", map[string]any{"pattern": "*.go"})
	var files []string
	if err := json.Unmarshal([]byte(out), &files); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(files) != 2 {
		t.Errorf("want 2 .go files, got %d: %v", len(files), files)
	}
}

func TestFindFilesNoMatch(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	out := callTool(t, ws, "Glob", map[string]any{"pattern": "*.rs"})
	var files []string
	json.Unmarshal([]byte(out), &files)
	if len(files) != 0 {
		t.Errorf("want 0 .rs files, got %d", len(files))
	}
}

func TestSearchText(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	out := callTool(t, ws, "Grep", map[string]any{
		"pattern": "func",
		"include": "*.go",
	})
	var matches []struct {
		File string `json:"file"`
		Line int    `json:"line"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal([]byte(out), &matches); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, out)
	}
	if len(matches) < 2 {
		t.Errorf("want at least 2 func matches, got %d", len(matches))
	}
}

func TestSearchTextInvalidPattern(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	dispatch := ws.Dispatch()
	raw, _ := json.Marshal(map[string]any{"pattern": "["}) // invalid regex
	out, err := dispatch["Grep"](context.Background(), raw)
	if err != nil {
		t.Fatalf("want literal fallback, got error: %v", err)
	}
	var matches []struct {
		File string `json:"file"`
	}
	if err := json.Unmarshal([]byte(out), &matches); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, out)
	}
	// No '[' characters exist in the setup fixtures, so the literal
	// search should match nothing — but must not error.
	if len(matches) != 0 {
		t.Errorf("want 0 matches for '[' in fixtures, got %d (%+v)", len(matches), matches)
	}
}

func TestReadFile(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	out := callTool(t, ws, "read_file", map[string]any{"path": "hello.txt"})
	if !strings.Contains(out, "hello world") {
		t.Errorf("expected 'hello world' in output, got: %s", out)
	}
}

func TestReadFileLineRange(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	// foo.go has 3 lines; request line 3 only (the func line)
	out := callTool(t, ws, "read_file", map[string]any{
		"path":       "subdir/foo.go",
		"start_line": 3,
		"end_line":   3,
	})
	if !strings.Contains(out, "Foo") {
		t.Errorf("expected Foo in line 3 output, got: %s", out)
	}
	if strings.Contains(out, "package") {
		t.Errorf("should not include line 1 (package), got: %s", out)
	}
}

func TestReadFileTraversalRejected(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	dispatch := ws.Dispatch()
	raw, _ := json.Marshal(map[string]any{"path": "../secret"})
	_, err := dispatch["read_file"](context.Background(), raw)
	if err == nil {
		t.Fatal("want error for traversal, got nil")
	}
}

func TestReadFileSizeLimit(t *testing.T) {
	root := t.TempDir()
	big := strings.Repeat("a", 200)
	os.WriteFile(filepath.Join(root, "big.txt"), []byte(big), 0644)

	ws, _ := workspace.NewReadOnly(workspace.Config{Root: root, MaxFileBytes: 100})
	out := callTool(t, ws, "read_file", map[string]any{"path": "big.txt"})
	if len(out) >= 200 {
		t.Errorf("expected truncation, got %d bytes", len(out))
	}
	if !strings.Contains(out, "truncated") {
		t.Errorf("expected truncation notice, got: %s", out)
	}
}

func TestFileInfo(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	out := callTool(t, ws, "file_info", map[string]any{"path": "hello.txt"})
	var info struct {
		Name  string `json:"name"`
		Size  int64  `json:"size"`
		IsDir bool   `json:"is_dir"`
	}
	if err := json.Unmarshal([]byte(out), &info); err != nil {
		t.Fatalf("unmarshal: %v (raw: %s)", err, out)
	}
	if info.Name != "hello.txt" {
		t.Errorf("want name hello.txt, got %q", info.Name)
	}
	if info.IsDir {
		t.Error("hello.txt should not be a directory")
	}
	if info.Size == 0 {
		t.Error("hello.txt size should not be 0")
	}
}

func TestFileInfoOnDir(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	out := callTool(t, ws, "file_info", map[string]any{"path": "subdir"})
	var info struct {
		IsDir bool `json:"is_dir"`
	}
	json.Unmarshal([]byte(out), &info)
	if !info.IsDir {
		t.Error("subdir should be a directory")
	}
}

func TestAbsolutePathRejected(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS(t, root)
	dispatch := ws.Dispatch()
	raw, _ := json.Marshal(map[string]any{"path": "/etc/passwd"})
	_, err := dispatch["read_file"](context.Background(), raw)
	if err == nil {
		t.Fatal("want error for absolute path, got nil")
	}
}

// ---- New (read-write) workspace tests ---------------------------------------

func newWS_RW(t *testing.T, root string) *workspace.Workspace {
	t.Helper()
	ws, err := workspace.New(workspace.Config{Root: root})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return ws
}

func TestNewHasSixBindings(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS_RW(t, root)
	ts := ws.Toolset()
	if len(ts.Bindings) != 6 {
		t.Errorf("want 6 bindings, got %d", len(ts.Bindings))
	}
}

func TestEditFileReplacesAllOccurrences(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "edit.txt"), []byte("hello world\nhello go\n"), 0644)
	ws := newWS_RW(t, root)

	out := callTool(t, ws, "edit_file", map[string]any{
		"path":       "edit.txt",
		"old_string": "hello",
		"new_string": "goodbye",
	})
	if !strings.Contains(out, "2") {
		t.Errorf("expected '2' in output, got: %s", out)
	}
	data, _ := os.ReadFile(filepath.Join(root, "edit.txt"))
	if strings.Contains(string(data), "hello") {
		t.Errorf("expected all occurrences replaced, file now: %s", data)
	}
	if !strings.Contains(string(data), "goodbye") {
		t.Errorf("expected replacement text present, file now: %s", data)
	}
}

func TestEditFileExpectedCountMatch(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "f.txt"), []byte("a b a"), 0644)
	ws := newWS_RW(t, root)

	out := callTool(t, ws, "edit_file", map[string]any{
		"path":           "f.txt",
		"old_string":     "a",
		"new_string":     "z",
		"expected_count": 2,
	})
	if !strings.Contains(out, "2") {
		t.Errorf("expected '2' in output, got: %s", out)
	}
}

func TestEditFileExpectedCountMismatch(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "f.txt"), []byte("a b a"), 0644)
	ws := newWS_RW(t, root)

	dispatch := ws.Dispatch()
	raw, _ := json.Marshal(map[string]any{
		"path":           "f.txt",
		"old_string":     "a",
		"new_string":     "z",
		"expected_count": 1,
	})
	_, err := dispatch["edit_file"](context.Background(), raw)
	if err == nil {
		t.Fatal("want error for expected_count mismatch, got nil")
	}
	if !strings.Contains(err.Error(), "expected 1") {
		t.Errorf("error should mention expected count, got: %v", err)
	}
}

func TestEditFileOldStringNotFound(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "f.txt"), []byte("hello world"), 0644)
	ws := newWS_RW(t, root)

	dispatch := ws.Dispatch()
	raw, _ := json.Marshal(map[string]any{
		"path":       "f.txt",
		"old_string": "missing",
		"new_string": "x",
	})
	_, err := dispatch["edit_file"](context.Background(), raw)
	if err == nil {
		t.Fatal("want error when old_string not found, got nil")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error should mention not found, got: %v", err)
	}
}

func TestEditFileTraversalRejected(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS_RW(t, root)

	dispatch := ws.Dispatch()
	raw, _ := json.Marshal(map[string]any{
		"path":       "../secret",
		"old_string": "x",
		"new_string": "y",
	})
	_, err := dispatch["edit_file"](context.Background(), raw)
	if err == nil {
		t.Fatal("want error for path traversal, got nil")
	}
}

func TestEditFilePreservesPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		// Windows doesn't implement POSIX mode bits — Go's chmod only honors the
		// 0200 (write) bit, so a 0600 file reports as 0666. The permission-
		// preservation contract this test checks is meaningful only on Unix.
		t.Skip("POSIX file modes are not supported on Windows")
	}
	root := t.TempDir()
	path := filepath.Join(root, "perm.txt")
	os.WriteFile(path, []byte("old content"), 0600)
	ws := newWS_RW(t, root)

	callTool(t, ws, "edit_file", map[string]any{
		"path":       "perm.txt",
		"old_string": "old",
		"new_string": "new",
	})
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode() != 0600 {
		t.Errorf("expected mode 0600, got %v", info.Mode())
	}
}

func TestEditFileMetadataFlags(t *testing.T) {
	root := setupTestDir(t)
	ws := newWS_RW(t, root)
	ts := ws.Toolset()
	for _, b := range ts.Bindings {
		if b.Tool.Name == "edit_file" {
			if b.Meta.ReadOnly {
				t.Error("edit_file should not be ReadOnly")
			}
			if !b.Meta.Destructive {
				t.Error("edit_file should be Destructive")
			}
			if !b.Meta.RequiresConfirmation {
				t.Error("edit_file should RequiresConfirmation")
			}
			return
		}
	}
	t.Fatal("edit_file binding not found in toolset")
}
