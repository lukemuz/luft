// Package editor implements the handlers for Anthropic's str_replace_based_edit_tool
// (text_editor_20250728) sandboxed to a workspace root.
//
// Pair the handler with anthropic.TextEditor20250728 to register a binding
// the model has been post-trained on:
//
//	ed, _ := editor.New(editor.Config{Root: "."})
//	binding := anthropic.TextEditor20250728(ed.Handler())
//	tools := luft.Tools(binding)
//
// Four commands are supported:
//
//   - view:        read a file (with optional [start,end] line range and
//     max_characters cap) or list a directory's entries
//   - str_replace: in-place exact-string replacement; rejects when old_str
//     is missing or non-unique
//   - create:      write a new file (refuses to overwrite existing files)
//   - insert:      insert new_str after a given line (0 = beginning)
//
// All paths are validated against the workspace root and rejected if they
// escape it or are absolute.
package editor

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/lukemuz/luft"
)

const (
	defaultMaxCharacters = 200_000
	defaultMaxFileBytes  = 5 << 20 // 5 MiB hard cap on files we'll touch
)

// Config controls Editor behaviour.
type Config struct {
	// Root is the directory all paths are resolved against. Required.
	Root string

	// MaxCharacters is the default cap on bytes returned by view when the
	// model does not supply max_characters. 0 uses 200_000.
	MaxCharacters int

	// MaxFileBytes caps how large a file the editor will read or modify.
	// 0 uses 5 MiB.
	MaxFileBytes int64
}

// Editor implements the str_replace_based_edit_tool commands.
type Editor struct {
	root         string
	maxChars     int
	maxFileBytes int64
}

// New constructs an Editor.
func New(cfg Config) (*Editor, error) {
	if cfg.Root == "" {
		return nil, fmt.Errorf("editor: Config.Root is required")
	}
	abs, err := filepath.Abs(cfg.Root)
	if err != nil {
		return nil, fmt.Errorf("editor: resolve root: %w", err)
	}
	maxChars := cfg.MaxCharacters
	if maxChars == 0 {
		maxChars = defaultMaxCharacters
	}
	maxBytes := cfg.MaxFileBytes
	if maxBytes == 0 {
		maxBytes = defaultMaxFileBytes
	}
	return &Editor{root: abs, maxChars: maxChars, maxFileBytes: maxBytes}, nil
}

// Toolset returns a single-binding toolset exposing the editor as a regular
// function-call tool (named "str_replace_based_edit_tool"). Use this for
// providers that don't support Anthropic's trained text_editor_20250728
// schema (e.g. OpenAI, OpenRouter).
func (e *Editor) Toolset() luft.Toolset {
	schema := luft.InputSchema{
		Type: "object",
		Properties: map[string]luft.SchemaProperty{
			"command":        {Type: "string", Description: "One of: view, str_replace, create, insert."},
			"path":           {Type: "string", Description: "File path relative to the workspace root."},
			"old_str":        {Type: "string", Description: "For str_replace: the exact existing substring to replace. Must be unique."},
			"new_str":        {Type: "string", Description: "For str_replace and insert: the new text."},
			"file_text":      {Type: "string", Description: "For create: the full contents of the new file."},
			"insert_line":    {Type: "integer", Description: "For insert: the 0-indexed line number after which to insert (0 = beginning)."},
			"view_range":     {Type: "array", Description: "For view: optional [start, end] 1-indexed inclusive line range."},
			"max_characters": {Type: "integer", Description: "For view: cap on bytes returned."},
		},
		Required: []string{"command", "path"},
	}
	desc := "View, create, or edit files in the workspace. Commands: view (read file or list directory), str_replace (in-place exact-string replacement; old_str must be unique), create (write new file; refuses to overwrite), insert (insert new_str after insert_line)."
	tool, fn := luft.NewTypedTool("str_replace_based_edit_tool", desc, schema, func(ctx context.Context, in editorInput) (string, error) {
		return e.dispatch(in)
	})
	single := luft.ToolBinding{Tool: tool, Func: fn, Meta: luft.ToolMetadata{RequiresConfirmation: true}}

	multiSchema := luft.InputSchema{
		Type: "object",
		Properties: map[string]luft.SchemaProperty{
			"path": {Type: "string", Description: "File path relative to the workspace root."},
			"edits": {
				Type:        "array",
				Description: "Ordered list of edits applied to the same file. Each is applied in turn to the evolving file contents.",
				Items: &luft.SchemaProperty{
					Type: "object",
					Properties: map[string]luft.SchemaProperty{
						"old_str": {Type: "string", Description: "Exact existing substring to replace. Must be unique in the file at the point this edit is applied."},
						"new_str": {Type: "string", Description: "Replacement text."},
					},
					Required: []string{"old_str", "new_str"},
				},
			},
		},
		Required: []string{"path", "edits"},
	}
	multiDesc := "Apply multiple str_replace edits to a single file in one atomic call. Edits apply in order to the evolving file; each old_str must match exactly once at the point it is applied. If any edit fails to match (or is non-unique), no edits are written. Prefer this over several str_replace_based_edit_tool calls when changing 2+ sites in the same file."
	multiTool, multiFn := luft.NewTypedTool("multi_edit", multiDesc, multiSchema, func(ctx context.Context, in multiEditInput) (string, error) {
		return e.multiEdit(in)
	})
	multi := luft.ToolBinding{Tool: multiTool, Func: multiFn, Meta: luft.ToolMetadata{RequiresConfirmation: true}}

	return luft.Tools(single, multi)
}

func (e *Editor) dispatch(in editorInput) (string, error) {
	switch in.Command {
	case "view":
		return e.view(in)
	case "str_replace":
		return e.strReplace(in)
	case "create":
		return e.create(in)
	case "insert":
		return e.insert(in)
	case "":
		return "", fmt.Errorf("editor: command is required")
	default:
		return "", fmt.Errorf("editor: unsupported command %q (want view|str_replace|create|insert)", in.Command)
	}
}

// Handler returns a ToolFunc suitable for anthropic.TextEditor20250728.
func (e *Editor) Handler() luft.ToolFunc {
	return func(ctx context.Context, raw json.RawMessage) (string, error) {
		var in editorInput
		if err := json.Unmarshal(raw, &in); err != nil {
			return "", fmt.Errorf("editor: parse input: %w", err)
		}
		switch in.Command {
		case "view":
			return e.view(in)
		case "str_replace":
			return e.strReplace(in)
		case "create":
			return e.create(in)
		case "insert":
			return e.insert(in)
		case "":
			return "", fmt.Errorf("editor: command is required")
		default:
			return "", fmt.Errorf("editor: unsupported command %q (want view|str_replace|create|insert)", in.Command)
		}
	}
}

// editorInput is the union of fields the model may emit. The active set
// depends on Command.
type editorInput struct {
	Command       string `json:"command"`
	Path          string `json:"path"`
	OldStr        string `json:"old_str,omitempty"`
	NewStr        string `json:"new_str,omitempty"`
	FileText      string `json:"file_text,omitempty"`
	InsertLine    *int   `json:"insert_line,omitempty"`
	ViewRange     []int  `json:"view_range,omitempty"`
	MaxCharacters int    `json:"max_characters,omitempty"`
}

func (e *Editor) safePath(rel string) (string, error) {
	if rel == "" {
		return "", fmt.Errorf("path is required")
	}
	// The trained tool emits absolute-looking paths sometimes; treat any
	// path as relative to root by stripping a leading slash.
	rel = strings.TrimPrefix(rel, "/")
	abs := filepath.Clean(filepath.Join(e.root, filepath.FromSlash(rel)))
	rootSep := e.root
	if !strings.HasSuffix(rootSep, string(filepath.Separator)) {
		rootSep += string(filepath.Separator)
	}
	if abs != e.root && !strings.HasPrefix(abs, rootSep) {
		return "", fmt.Errorf("path %q escapes the workspace root", rel)
	}
	return abs, nil
}

func (e *Editor) relPath(abs string) string {
	rel, err := filepath.Rel(e.root, abs)
	if err != nil {
		return abs
	}
	return filepath.ToSlash(rel)
}

func (e *Editor) view(in editorInput) (string, error) {
	abs, err := e.safePath(in.Path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("view: %w", err)
	}
	if info.IsDir() {
		entries, err := os.ReadDir(abs)
		if err != nil {
			return "", fmt.Errorf("view: %w", err)
		}
		var b strings.Builder
		fmt.Fprintf(&b, "%s/\n", e.relPath(abs))
		for _, ent := range entries {
			suffix := ""
			if ent.IsDir() {
				suffix = "/"
			}
			fmt.Fprintf(&b, "  %s%s\n", ent.Name(), suffix)
		}
		return strings.TrimRight(b.String(), "\n"), nil
	}
	if info.Size() > e.maxFileBytes {
		return "", fmt.Errorf("view: file is %d bytes, exceeds limit %d", info.Size(), e.maxFileBytes)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", fmt.Errorf("view: %w", err)
	}
	text := string(data)

	if len(in.ViewRange) == 2 {
		lines := strings.Split(text, "\n")
		start, end := in.ViewRange[0], in.ViewRange[1]
		if start < 1 {
			start = 1
		}
		if end == -1 || end > len(lines) {
			end = len(lines)
		}
		if start > end {
			return "", fmt.Errorf("view: invalid range [%d,%d]", in.ViewRange[0], in.ViewRange[1])
		}
		var b strings.Builder
		for i := start - 1; i < end; i++ {
			fmt.Fprintf(&b, "%6d\t%s\n", i+1, lines[i])
		}
		return strings.TrimRight(b.String(), "\n"), nil
	}

	cap := in.MaxCharacters
	if cap <= 0 {
		cap = e.maxChars
	}
	if len(text) > cap {
		text = text[:cap] + fmt.Sprintf("\n... [truncated %d more bytes; pass max_characters or view_range to see more]", len(text)-cap)
	}
	// Number lines for the model's benefit.
	lines := strings.Split(text, "\n")
	var b strings.Builder
	for i, line := range lines {
		fmt.Fprintf(&b, "%6d\t%s\n", i+1, line)
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

func (e *Editor) strReplace(in editorInput) (string, error) {
	if in.OldStr == "" {
		return "", fmt.Errorf("str_replace: old_str is required")
	}
	abs, err := e.safePath(in.Path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("str_replace: %w", err)
	}
	if info.Size() > e.maxFileBytes {
		return "", fmt.Errorf("str_replace: file is %d bytes, exceeds limit %d", info.Size(), e.maxFileBytes)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", fmt.Errorf("str_replace: %w", err)
	}
	text := string(data)
	oldStr, newStr := reconcileNewlines(text, in.OldStr, in.NewStr)
	count := strings.Count(text, oldStr)
	if count == 0 {
		return "", fmt.Errorf("str_replace: old_str not found in %q", in.Path)
	}
	if count > 1 {
		return "", fmt.Errorf("str_replace: old_str matches %d times in %q; make it unique by adding context", count, in.Path)
	}
	updated := strings.Replace(text, oldStr, newStr, 1)
	if err := os.WriteFile(abs, []byte(updated), info.Mode().Perm()); err != nil {
		return "", fmt.Errorf("str_replace: %w", err)
	}
	return fmt.Sprintf("edited %s (1 replacement)", e.relPath(abs)), nil
}

// reconcileNewlines adapts old_str/new_str to the file's newline convention so
// a multi-line edit doesn't spuriously fail on line-ending mismatch. Models and
// JSON tool arguments routinely use LF even when the on-disk file is CRLF (the
// norm on Windows), so an otherwise-correct multi-line old_str won't be found
// verbatim. If old_str already matches, both strings are returned untouched;
// otherwise both are translated to the file's dominant newline (CRLF if the
// file contains any, else LF), which makes the edit land and preserves the
// file's existing convention on write.
func reconcileNewlines(text, oldStr, newStr string) (string, string) {
	if strings.Contains(text, oldStr) {
		return oldStr, newStr
	}
	conv := func(s string) string {
		s = strings.ReplaceAll(s, "\r\n", "\n")
		if strings.Contains(text, "\r\n") {
			s = strings.ReplaceAll(s, "\n", "\r\n")
		}
		return s
	}
	return conv(oldStr), conv(newStr)
}

func (e *Editor) create(in editorInput) (string, error) {
	abs, err := e.safePath(in.Path)
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(abs); err == nil {
		return "", fmt.Errorf("create: %q already exists; use str_replace to modify it", in.Path)
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return "", fmt.Errorf("create: %w", err)
	}
	if err := os.WriteFile(abs, []byte(in.FileText), 0o644); err != nil {
		return "", fmt.Errorf("create: %w", err)
	}
	return fmt.Sprintf("created %s (%d bytes)", e.relPath(abs), len(in.FileText)), nil
}

func (e *Editor) insert(in editorInput) (string, error) {
	if in.InsertLine == nil {
		return "", fmt.Errorf("insert: insert_line is required")
	}
	abs, err := e.safePath(in.Path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("insert: %w", err)
	}
	if info.Size() > e.maxFileBytes {
		return "", fmt.Errorf("insert: file is %d bytes, exceeds limit %d", info.Size(), e.maxFileBytes)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", fmt.Errorf("insert: %w", err)
	}
	lines := strings.Split(string(data), "\n")
	at := *in.InsertLine
	if at < 0 || at > len(lines) {
		return "", fmt.Errorf("insert: insert_line %d is out of range [0,%d]", at, len(lines))
	}
	newLines := strings.Split(in.NewStr, "\n")
	out := make([]string, 0, len(lines)+len(newLines))
	out = append(out, lines[:at]...)
	out = append(out, newLines...)
	out = append(out, lines[at:]...)
	if err := os.WriteFile(abs, []byte(strings.Join(out, "\n")), info.Mode().Perm()); err != nil {
		return "", fmt.Errorf("insert: %w", err)
	}
	return fmt.Sprintf("inserted %d line(s) into %s after line %d", len(newLines), e.relPath(abs), at), nil
}

// multiEditInput is the input shape for the multi_edit tool: a single file
// and an ordered list of str_replace edits applied atomically.
type multiEditInput struct {
	Path  string `json:"path"`
	Edits []struct {
		OldStr string `json:"old_str"`
		NewStr string `json:"new_str"`
	} `json:"edits"`
}

// multiEdit applies all edits to one file in a single atomic operation. Each
// old_str must occur exactly once in the current buffer at the point it is
// applied (later edits see the results of earlier ones). Validation runs over
// an in-memory copy; the file is written only if every edit succeeds, so a
// failure mid-list leaves the file untouched.
func (e *Editor) multiEdit(in multiEditInput) (string, error) {
	if len(in.Edits) == 0 {
		return "", fmt.Errorf("multi_edit: edits is empty")
	}
	abs, err := e.safePath(in.Path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("multi_edit: %w", err)
	}
	if info.Size() > e.maxFileBytes {
		return "", fmt.Errorf("multi_edit: file is %d bytes, exceeds limit %d", info.Size(), e.maxFileBytes)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", fmt.Errorf("multi_edit: %w", err)
	}
	text := string(data)
	for i, ed := range in.Edits {
		if ed.OldStr == "" {
			return "", fmt.Errorf("multi_edit: edit %d: old_str is required (no edits applied)", i+1)
		}
		oldStr, newStr := reconcileNewlines(text, ed.OldStr, ed.NewStr)
		count := strings.Count(text, oldStr)
		if count == 0 {
			return "", fmt.Errorf("multi_edit: edit %d: old_str not found in %q (no edits applied)", i+1, in.Path)
		}
		if count > 1 {
			return "", fmt.Errorf("multi_edit: edit %d: old_str matches %d times in %q; add surrounding context to make it unique (no edits applied)", i+1, count, in.Path)
		}
		text = strings.Replace(text, oldStr, newStr, 1)
	}
	if err := os.WriteFile(abs, []byte(text), info.Mode().Perm()); err != nil {
		return "", fmt.Errorf("multi_edit: %w", err)
	}
	return fmt.Sprintf("edited %s (%d replacements)", e.relPath(abs), len(in.Edits)), nil
}
