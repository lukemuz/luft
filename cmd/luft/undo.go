package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/lukemuz/luft"
)

// undoEntry captures one file's state around a single successful editor-tool
// invocation, enough to revert it. priorData/priorMode restore a modified file;
// existed=false marks a newly created file for deletion on undo. postHash
// detects on-disk changes that happened after the edit (e.g. via bash) so undo
// can skip rather than clobber unexpected state.
type undoEntry struct {
	absPath   string
	existed   bool
	priorData []byte
	priorMode os.FileMode
	postHash  string
}

// turnFrame is the group of edits made during one REPL turn; /undo reverts
// the top frame as a unit.
type turnFrame struct {
	entries []undoEntry
}

// undoStack is the in-memory, per-session stack of turn frames. It is shared
// between the editCheckpoint middleware (pusher) and the /undo command
// (popper). Tool calls within a turn may run concurrently, so all access is
// mutex-guarded. The stack is bounded by max; the oldest frame is trimmed when
// the bound is exceeded. It is never persisted — a resumed session starts
// empty.
type undoStack struct {
	mu     sync.Mutex
	frames []turnFrame
	max    int
}

func newUndoStack(max int) *undoStack {
	return &undoStack{max: max}
}

// beginTurn opens a fresh frame for the current turn. Called at the top of
// runTurn; a turn that ends up making no edits is dropped by endTurn so /undo
// targets the previous turn instead of a no-op.
func (s *undoStack) beginTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.frames = append(s.frames, turnFrame{})
}

// endTurn closes the current turn: drops an empty top frame and trims to the
// depth bound. Non-empty frames are always retained, even on a turn that
// ended in error, so partial edits remain revertible.
func (s *undoStack) endTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.frames) == 0 {
		return
	}
	if len(s.frames[len(s.frames)-1].entries) == 0 {
		s.frames = s.frames[:len(s.frames)-1]
	}
	if len(s.frames) > s.max {
		s.frames = s.frames[len(s.frames)-s.max:]
	}
}

func (s *undoStack) push(e undoEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.frames) == 0 {
		return
	}
	s.frames[len(s.frames)-1].entries = append(s.frames[len(s.frames)-1].entries, e)
}

// undo reverts the most recent turn's edits. Files whose current on-disk
// content no longer matches the recorded post-edit hash are skipped (not
// clobbered). Returns counts and a human-readable summary.
func (s *undoStack) undo() (restored, skipped int, nothing bool, msg string) {
	s.mu.Lock()
	if len(s.frames) == 0 {
		s.mu.Unlock()
		return 0, 0, true, "(nothing to undo — edits made via the editor tools this turn are revertible; bash-made changes are not tracked)"
	}
	frame := s.frames[len(s.frames)-1]
	s.frames = s.frames[:len(s.frames)-1]
	s.mu.Unlock()

	var skippedPaths []string
	for i := len(frame.entries) - 1; i >= 0; i-- {
		e := frame.entries[i]
		if cur := fileHash(e.absPath); cur != e.postHash {
			skipped++
			skippedPaths = append(skippedPaths, e.absPath)
			continue
		}
		if !e.existed {
			// Created this turn: remove it. Already-gone is a no-op success.
			if err := os.Remove(e.absPath); err != nil && !os.IsNotExist(err) {
				skipped++
				skippedPaths = append(skippedPaths, e.absPath)
				continue
			}
			restored++
			continue
		}
		// Modified this turn: restore prior content + mode.
		if err := os.MkdirAll(filepath.Dir(e.absPath), 0o755); err != nil {
			skipped++
			skippedPaths = append(skippedPaths, e.absPath)
			continue
		}
		mode := e.priorMode.Perm()
		if mode == 0 {
			mode = 0o644
		}
		if err := os.WriteFile(e.absPath, e.priorData, mode); err != nil {
			skipped++
			skippedPaths = append(skippedPaths, e.absPath)
			continue
		}
		restored++
	}

	switch {
	case skipped == 0:
		msg = fmt.Sprintf("reverted %d editor edit(s) this turn", restored)
	case restored == 0:
		msg = fmt.Sprintf("skipped %d edit(s) — file(s) changed on disk since the edit: %s", skipped, strings.Join(skippedPaths, ", "))
	default:
		msg = fmt.Sprintf("reverted %d editor edit(s); skipped %d (file(s) changed on disk since the edit: %s)", restored, skipped, strings.Join(skippedPaths, ", "))
	}
	return restored, skipped, false, msg
}

// editCheckpoint returns middleware that snapshots a target file's pre-edit
// state before the editor func runs and, on success, pushes an undo entry onto
// the shared stack. It must be the innermost editor wrapper (listed last in
// Wrap) so the snapshot is taken immediately before the disk write, after
// confirmation and timeout have already passed.
//
// Only the structured editor tools are snapshotted: str_replace_based_edit_tool
// (create/str_replace/insert; view is read-only and skipped) and multi_edit.
// Arbitrary file mutations made via bash cannot be generically snapshotted and
// are intentionally out of scope.
func editCheckpoint(stack *undoStack, root string) luft.Middleware {
	return func(b luft.ToolBinding) luft.ToolFunc {
		name := b.Tool.Name
		return func(ctx context.Context, input json.RawMessage) (string, error) {
			rel, mutates, ok := decodeEditPath(name, input)
			if !ok || !mutates {
				return b.Func(ctx, input)
			}
			abs, err := resolveUnderRoot(root, rel)
			var pre undoEntry
			snapshotOK := false
			if err == nil {
				pre, snapshotOK = snapshotFile(abs)
			}
			out, err := b.Func(ctx, input)
			if err != nil {
				return out, err
			}
			if snapshotOK {
				pre.absPath = abs
				pre.postHash = fileHash(abs)
				stack.push(pre)
			}
			return out, nil
		}
	}
}

// decodeEditPath extracts the target file path from an editor-tool input and
// reports whether the call mutates a file. view returns mutates=false.
func decodeEditPath(toolName string, input json.RawMessage) (path string, mutates bool, ok bool) {
	switch toolName {
	case "str_replace_based_edit_tool":
		var in struct {
			Command string `json:"command"`
			Path    string `json:"path"`
		}
		if err := json.Unmarshal(input, &in); err != nil {
			return "", false, false
		}
		switch in.Command {
		case "str_replace", "create", "insert":
			return in.Path, true, true
		default:
			return in.Path, false, true
		}
	case "multi_edit":
		var in struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal(input, &in); err != nil {
			return "", false, false
		}
		return in.Path, true, true
	}
	return "", false, false
}

// resolveUnderRoot mirrors editor.Editor.safePath: a leading slash is stripped,
// the path is cleaned under root, and an escape attempt is rejected. Keeping
// this identical to the editor's own resolution guarantees the snapshot targets
// exactly the file the editor will write.
func resolveUnderRoot(root, rel string) (string, error) {
	if rel == "" {
		return "", fmt.Errorf("path is required")
	}
	rel = strings.TrimPrefix(rel, "/")
	abs := filepath.Clean(filepath.Join(root, filepath.FromSlash(rel)))
	rootSep := root
	if !strings.HasSuffix(rootSep, string(filepath.Separator)) {
		rootSep += string(filepath.Separator)
	}
	if abs != root && !strings.HasPrefix(abs, rootSep) {
		return "", fmt.Errorf("path %q escapes the workspace root", rel)
	}
	return abs, nil
}

// snapshotFile reads a file's current content and mode, or records that it does
// not exist (the pre-state for a create).
func snapshotFile(abs string) (undoEntry, bool) {
	info, err := os.Stat(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return undoEntry{existed: false}, true
		}
		return undoEntry{}, false
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return undoEntry{existed: true}, true
	}
	return undoEntry{existed: true, priorData: data, priorMode: info.Mode()}, true
}

// fileHash returns the sha256 of the file's current content, or "" if it does
// not exist. Used for clobber-safe undo: a mismatch with the recorded
// post-edit hash means the file changed underneath us after the edit.
func fileHash(abs string) string {
	data, err := os.ReadFile(abs)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
