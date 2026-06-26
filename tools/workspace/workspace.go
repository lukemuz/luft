// Package workspace provides a safe, sandboxed set of filesystem tools rooted
// at a configurable directory. All path arguments are resolved relative to the
// root and validated to prevent directory traversal.
//
// Use NewReadOnly for the five read-only tools (list_directory, Glob,
// Grep, read_file, file_info). Use New to also include edit_file; pair
// it with the luft.WithConfirmation middleware to gate writes.
package workspace

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/lukemuz/luft"
)

const (
	defaultMaxFileBytes = 1 << 20 // 1 MiB
	defaultMaxResults   = 100

	// binarySniffBytes is how many leading bytes we look at to detect a
	// binary file (presence of a NUL byte). 8 KiB matches what most
	// search tools use.
	binarySniffBytes = 8192
)

// defaultSkipDirs is the set of directory basenames that Glob and Grep
// skip during traversal by default. These are conventional bloat
// directories — version-control metadata, dependency vendors, build
// outputs, language-specific caches — that almost always waste tool
// budget without producing useful results in a coding context.
//
// Walks rooted directly at one of these directories are NOT pruned:
// the skip rule applies to descendants encountered during traversal,
// so a user who explicitly asks to search inside node_modules still
// gets results.
var defaultSkipDirs = map[string]bool{
	".git": true, ".hg": true, ".svn": true,
	"node_modules": true, "bower_components": true,
	"vendor":        true,
	"target":        true,
	"dist":          true,
	"build":         true,
	"out":           true,
	".next":         true,
	".nuxt":         true,
	".cache":        true,
	".parcel-cache": true,
	"__pycache__":   true,
	".venv":         true,
	"venv":          true,
	".tox":          true,
	".pytest_cache": true,
	".mypy_cache":   true,
	".gradle":       true,
	".idea":         true,
	".vscode":       true,
	"coverage":      true,
}

// binaryExts is a fast-path: files with these extensions are skipped by
// Grep without opening them. The NUL-byte sniff catches everything
// else; this just avoids the open() syscall on the obvious cases.
var binaryExts = map[string]bool{
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".bmp": true, ".webp": true, ".ico": true, ".tiff": true,
	".pdf": true,
	".zip": true, ".tar": true, ".gz": true, ".bz2": true, ".xz": true, ".7z": true, ".rar": true,
	".exe": true, ".dll": true, ".so": true, ".dylib": true, ".a": true, ".o": true,
	".class": true, ".jar": true, ".war": true,
	".mp3": true, ".mp4": true, ".wav": true, ".avi": true, ".mov": true, ".mkv": true, ".flac": true, ".ogg": true,
	".woff": true, ".woff2": true, ".ttf": true, ".otf": true, ".eot": true,
	".pyc": true, ".pyo": true,
	".bin": true, ".dat": true,
}

// Config controls Workspace tool behaviour.
type Config struct {
	// Root is the directory that all workspace tools are sandboxed to.
	// All relative paths supplied to tools are resolved against Root.
	// Required.
	Root string

	// MaxFileBytes caps how many bytes read_file will return. 0 uses the
	// default (1 MiB).
	MaxFileBytes int64

	// MaxResults caps the number of entries returned by Glob and
	// Grep. 0 uses the default (100).
	MaxResults int

	// SkipDirs overrides the default set of directory basenames that
	// Glob and Grep prune during traversal (.git, node_modules, vendor,
	// build, dist, target, .venv, etc.). nil uses the default set.
	// Pass an empty (non-nil) map to disable skipping entirely.
	SkipDirs map[string]bool
}

// Workspace exposes a set of safe, read-only filesystem tools sandboxed to
// a root directory.
//
// Usage:
//
//	ws, err := workspace.NewReadOnly(workspace.Config{Root: "."})
//	if err != nil { ... }
//
//	// Use the toolset directly:
//	result, err := client.Loop(ctx, system, history, ws.Toolset(), 10)
//
//	// Or compose with other toolsets:
//	toolset := luft.MustJoin(clockset, ws.Toolset())
type Workspace struct {
	root         string
	maxFileBytes int64
	maxResults   int
	skipDirs     map[string]bool
	bindings     []luft.ToolBinding
}

// New creates a Workspace with all six tools: the five read-only tools from
// NewReadOnly plus edit_file for in-place text replacement. The edit_file tool
// carries RequiresConfirmation: true metadata; pair the returned Toolset with
// luft.WithConfirmation to gate writes.
// Returns an error if cfg.Root is empty or cannot be resolved to an absolute path.
func New(cfg Config) (*Workspace, error) {
	w, err := NewReadOnly(cfg)
	if err != nil {
		return nil, err
	}
	w.bindings = append(w.bindings, w.buildEditBinding())
	return w, nil
}

// NewReadOnly creates a Workspace with the five core read-only tools:
// list_directory, Glob, Grep, read_file, and file_info.
// Returns an error if cfg.Root is empty or cannot be resolved to an
// absolute path.
func NewReadOnly(cfg Config) (*Workspace, error) {
	if cfg.Root == "" {
		return nil, fmt.Errorf("workspace: Config.Root is required")
	}
	abs, err := filepath.Abs(cfg.Root)
	if err != nil {
		return nil, fmt.Errorf("workspace: resolve root %q: %w", cfg.Root, err)
	}
	maxBytes := cfg.MaxFileBytes
	if maxBytes == 0 {
		maxBytes = defaultMaxFileBytes
	}
	maxRes := cfg.MaxResults
	if maxRes == 0 {
		maxRes = defaultMaxResults
	}
	skip := cfg.SkipDirs
	if skip == nil {
		skip = defaultSkipDirs
	}
	w := &Workspace{
		root:         abs,
		maxFileBytes: maxBytes,
		maxResults:   maxRes,
		skipDirs:     skip,
	}
	w.bindings = w.buildBindings()
	return w, nil
}

// Toolset returns all workspace bindings as a luft.Toolset.
func (w *Workspace) Toolset() luft.Toolset {
	return luft.Toolset{Bindings: w.bindings}
}

// Tools returns the model-facing Tool slice. Most callers should use
// Toolset() and pass it to Client.Loop directly; Tools() exists for
// inspection and for callers building a custom loop on top of the
// provider primitives.
func (w *Workspace) Tools() []luft.Tool {
	return w.Toolset().Tools()
}

// Dispatch returns the name→func map. See Tools() for when to use this.
func (w *Workspace) Dispatch() map[string]luft.ToolFunc {
	return w.Toolset().Dispatch()
}

// safePath resolves a caller-supplied relative path against the workspace
// root, rejects absolute paths and dot-dot traversal, and returns the
// cleaned absolute path. An empty rel resolves to the root itself.
func (w *Workspace) safePath(rel string) (string, error) {
	if filepath.IsAbs(rel) {
		return "", fmt.Errorf("absolute paths are not allowed; use a path relative to the workspace root")
	}
	abs := filepath.Join(w.root, filepath.FromSlash(rel))
	// filepath.Join calls Clean, which resolves all ".." components. Check
	// the result is still inside the root.
	rootWithSep := w.root
	if !strings.HasSuffix(rootWithSep, string(filepath.Separator)) {
		rootWithSep += string(filepath.Separator)
	}
	if abs != w.root && !strings.HasPrefix(abs, rootWithSep) {
		return "", fmt.Errorf("path %q escapes the workspace root", rel)
	}
	return abs, nil
}

// relPath converts an absolute path back to a root-relative slash path for
// display in tool results.
func (w *Workspace) relPath(abs string) string {
	rel, err := filepath.Rel(w.root, abs)
	if err != nil {
		return abs
	}
	return filepath.ToSlash(rel)
}

// ---- tool input types -------------------------------------------------------

type listDirInput struct {
	Path string `json:"path"`
}

type findFilesInput struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path"`
}

type searchTextInput struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path"`
	Include string `json:"include"`
}

type readFileInput struct {
	Path      string `json:"path"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
}

type fileInfoInput struct {
	Path string `json:"path"`
}

type editFileInput struct {
	Path          string `json:"path"`
	OldString     string `json:"old_string"`
	NewString     string `json:"new_string"`
	ExpectedCount int    `json:"expected_count"`
}

// ---- binding construction ---------------------------------------------------

func (w *Workspace) buildBindings() []luft.ToolBinding {
	meta := luft.ToolMetadata{
		Source:     "tools/workspace",
		ReadOnly:   true,
		Filesystem: true,
	}

	listDirTool := luft.NewTool(
		"list_directory",
		"Lists files and directories at a path relative to the workspace root. Non-recursive.",
		luft.Object(
			luft.String("path", `Directory to list, relative to the workspace root. Defaults to "." (the root).`),
		),
	)

	findFilesTool := luft.NewTool(
		"Glob",
		"Finds files matching a glob pattern under a path relative to the workspace root. "+
			"Supports `*` (any characters within a path segment), `?` (single char), `[abc]` (char class), "+
			"and `**` (any number of path segments — e.g. `**/*.go` for all Go files at any depth, "+
			"`pkg/**/*_test.go` for test files under pkg/). "+
			"Patterns without `/` or `**` match against the file's basename only (so `*.go` works as expected). "+
			"Common bloat directories (.git, node_modules, vendor, target, dist, build, .venv, __pycache__, etc.) "+
			"are skipped automatically. To search inside one, pass it explicitly as `path`. "+
			"Results are root-relative slash paths, capped at MaxResults.",
		luft.Object(
			luft.String("pattern", "Glob pattern. Examples: `*.go`, `**/*.ts`, `pkg/**/*_test.go`.", luft.Required()),
			luft.String("path", `Directory to search under, relative to the workspace root. Defaults to ".".`),
		),
	)

	searchTextTool := luft.NewTool(
		"Grep",
		"Searches file contents for lines matching a pattern. The pattern is treated as a Go regular expression — "+
			"plain literals (no regex metachars) take a fast non-regex path automatically. "+
			"If a pattern contains metacharacters but fails to compile as a regex (e.g. `Trim(` with an unbalanced paren), "+
			"it is searched for as a literal substring rather than producing an error. "+
			"Returns matches as {file, line, text} objects, sorted by file then line. "+
			"Skips common bloat directories (.git, node_modules, vendor, build, dist, target, .venv, __pycache__, etc.) "+
			"and binary files (by extension and by NUL-byte sniff). "+
			"Walks the tree in parallel; capped at MaxResults matches.",
		luft.Object(
			luft.String("pattern", "Pattern to search for. Plain literals are fastest; full Go regex syntax is supported.", luft.Required()),
			luft.String("path", `Directory to search under, relative to the workspace root. Defaults to ".".`),
			luft.String("include", `Optional glob filter, applied to each file's path relative to the search base (e.g. "*.go", "src/**/*.ts").`),
		),
	)

	readFileTool := luft.NewTool(
		"read_file",
		"Reads the contents of a file. Respects MaxFileBytes. "+
			"Optionally restricts output to a line range (1-indexed, inclusive). "+
			"When the file is an image (png, jpeg, gif, webp, bmp, tiff), the text payload "+
			"is a one-line metadata summary and the image bytes are attached to the result "+
			"so the model receives them as visual content.",
		luft.Object(
			luft.String("path", "File path relative to the workspace root.", luft.Required()),
			luft.Integer("start_line", "First line to include (1-indexed). 0 or omitted means start of file."),
			luft.Integer("end_line", "Last line to include (1-indexed, inclusive). 0 or omitted means end of file."),
		),
	)

	fileInfoTool := luft.NewTool(
		"file_info",
		"Returns metadata (name, size, modification time, is_dir, mode) for a path relative to the workspace root.",
		luft.Object(
			luft.String("path", "Path relative to the workspace root.", luft.Required()),
		),
	)

	return []luft.ToolBinding{
		{Tool: listDirTool, Func: luft.TypedToolFunc(w.listDirectory), Meta: meta},
		{Tool: findFilesTool, Func: luft.TypedToolFunc(w.findFiles), Meta: meta},
		{Tool: searchTextTool, Func: luft.TypedToolFunc(w.searchText), Meta: meta},
		{Tool: readFileTool, Func: luft.TypedToolFunc(w.readFile), Meta: meta},
		{Tool: fileInfoTool, Func: luft.TypedToolFunc(w.fileInfo), Meta: meta},
	}
}

func (w *Workspace) buildEditBinding() luft.ToolBinding {
	editMeta := luft.ToolMetadata{
		Source:               "tools/workspace",
		ReadOnly:             false,
		Destructive:          true,
		Filesystem:           true,
		RequiresConfirmation: true,
	}
	editFileTool := luft.NewTool(
		"edit_file",
		"Replaces all occurrences of old_string with new_string in a file. "+
			"Returns an error if old_string is not found. "+
			"Set expected_count to a positive integer to assert the exact number of replacements; "+
			"the edit is rejected when the actual count differs.",
		luft.Object(
			luft.String("path", "File path relative to the workspace root.", luft.Required()),
			luft.String("old_string", "Exact string to find and replace.", luft.Required()),
			luft.String("new_string", "Replacement string.", luft.Required()),
			luft.Integer("expected_count", "Expected number of occurrences. When non-zero, the edit is rejected if the actual count differs."),
		),
	)
	return luft.ToolBinding{Tool: editFileTool, Func: luft.TypedToolFunc(w.editFile), Meta: editMeta}
}

// ---- tool implementations ---------------------------------------------------

func (w *Workspace) listDirectory(_ context.Context, in listDirInput) (string, error) {
	rel := in.Path
	if rel == "" {
		rel = "."
	}
	abs, err := w.safePath(rel)
	if err != nil {
		return "", err
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return "", fmt.Errorf("list_directory: %w", err)
	}
	type entry struct {
		Name  string `json:"name"`
		IsDir bool   `json:"is_dir"`
	}
	out := make([]entry, len(entries))
	for i, e := range entries {
		out[i] = entry{Name: e.Name(), IsDir: e.IsDir()}
	}
	return luft.JSONResult(out)
}

func (w *Workspace) findFiles(_ context.Context, in findFilesInput) (string, error) {
	base := in.Path
	if base == "" {
		base = "."
	}
	absBase, err := w.safePath(base)
	if err != nil {
		return "", err
	}
	pattern := in.Pattern
	if pattern == "" {
		return "", fmt.Errorf("Glob: pattern is required")
	}

	var matches []string
	err = filepath.WalkDir(absBase, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil // skip unreadable dirs
		}
		if d.IsDir() {
			if p != absBase && w.skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		// Path relative to the search base, with forward slashes —
		// gives the model the natural "src/foo/bar.go" form to match
		// against patterns like "src/**/*.go".
		rel, _ := filepath.Rel(absBase, p)
		rel = filepath.ToSlash(rel)
		if globMatch(pattern, rel) {
			matches = append(matches, w.relPath(p))
			if len(matches) >= w.maxResults {
				return filepath.SkipAll
			}
		}
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("Glob: %w", err)
	}
	return luft.JSONResult(matches)
}

func (w *Workspace) searchText(ctx context.Context, in searchTextInput) (string, error) {
	base := in.Path
	if base == "" {
		base = "."
	}
	absBase, err := w.safePath(base)
	if err != nil {
		return "", err
	}
	if in.Pattern == "" {
		return "", fmt.Errorf("Grep: pattern is required")
	}
	m, err := compileMatcher(in.Pattern)
	if err != nil {
		return "", fmt.Errorf("Grep: invalid pattern: %w", err)
	}

	// Parallel scan: a walker goroutine collects candidate file paths
	// and feeds them to a worker pool. Workers open and scan files
	// independently. A single goroutine merges matches into a slice
	// and cancels the workerCtx once the cap is reached.
	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	pathsCh := make(chan string, 64)
	resultsCh := make(chan grepMatch, 64)

	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}
	if workers < 2 {
		workers = 2
	}

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range pathsCh {
				if workerCtx.Err() != nil {
					continue // drain
				}
				w.grepFile(workerCtx, p, m, resultsCh)
			}
		}()
	}

	// Walker.
	go func() {
		defer close(pathsCh)
		_ = filepath.WalkDir(absBase, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if workerCtx.Err() != nil {
				return filepath.SkipAll
			}
			if d.IsDir() {
				if p != absBase && w.skipDirs[d.Name()] {
					return filepath.SkipDir
				}
				return nil
			}
			if in.Include != "" {
				rel, _ := filepath.Rel(absBase, p)
				if !globMatch(in.Include, filepath.ToSlash(rel)) {
					return nil
				}
			}
			if binaryExts[strings.ToLower(filepath.Ext(p))] {
				return nil
			}
			select {
			case pathsCh <- p:
			case <-workerCtx.Done():
				return filepath.SkipAll
			}
			return nil
		})
	}()

	// Closer: when all workers exit, close the result channel so the
	// merger loop terminates.
	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// Merger.
	var matches []grepMatch
	for r := range resultsCh {
		if len(matches) < w.maxResults {
			matches = append(matches, r)
			if len(matches) >= w.maxResults {
				cancel()
			}
		}
	}

	if ctx.Err() != nil {
		return "", ctx.Err()
	}

	// Order is non-deterministic with parallel workers — sort for
	// stable output (file, then line).
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].File != matches[j].File {
			return matches[i].File < matches[j].File
		}
		return matches[i].Line < matches[j].Line
	})
	return luft.JSONResult(matches)
}

// grepMatch is one Grep result row.
type grepMatch struct {
	File string `json:"file"`
	Line int    `json:"line"`
	Text string `json:"text"`
}

// grepFile is the per-file scanner used by searchText. It opens the file,
// sniffs for a NUL byte to skip binaries that slipped past the extension
// filter, and emits matches on resultsCh until the workerCtx is cancelled.
func (w *Workspace) grepFile(ctx context.Context, p string, m matcher, resultsCh chan<- grepMatch) {
	f, err := os.Open(p)
	if err != nil {
		return
	}
	defer f.Close()

	br := bufio.NewReader(f)

	// Sniff first chunk for a NUL byte to detect binary files that
	// don't have a known extension.
	head, _ := br.Peek(binarySniffBytes)
	if bytes.IndexByte(head, 0) >= 0 {
		return
	}

	rel := w.relPath(p)
	lineNum := 0
	for {
		if ctx.Err() != nil {
			return
		}
		line, err := br.ReadSlice('\n')
		if len(line) > 0 {
			lineNum++
			// Trim the trailing newline (and CR) before matching/output.
			trimmed := line
			for len(trimmed) > 0 && (trimmed[len(trimmed)-1] == '\n' || trimmed[len(trimmed)-1] == '\r') {
				trimmed = trimmed[:len(trimmed)-1]
			}
			if m.match(trimmed) {
				select {
				case resultsCh <- grepMatch{File: rel, Line: lineNum, Text: string(trimmed)}:
				case <-ctx.Done():
					return
				}
			}
		}
		if err != nil {
			// io.EOF or bufio.ErrBufferFull — either way stop on this file.
			return
		}
	}
}

// matcher abstracts over the literal fast-path and full regex.
type matcher interface {
	match(line []byte) bool
}

type literalMatcher struct{ pat []byte }

func (l *literalMatcher) match(line []byte) bool { return bytes.Contains(line, l.pat) }

type regexMatcher struct{ re *regexp.Regexp }

func (r *regexMatcher) match(line []byte) bool { return r.re.Match(line) }

// regexMetachars is the set of characters that turn a string from a plain
// literal into a non-trivial regex. If a pattern contains none of these,
// we can search with bytes.Contains, which is dramatically faster than
// running the regex engine per line.
const regexMetachars = `\.+*?()|[]{}^$`

func compileMatcher(p string) (matcher, error) {
	if p != "" && !strings.ContainsAny(p, regexMetachars) {
		return &literalMatcher{pat: []byte(p)}, nil
	}
	re, err := regexp.Compile(p)
	if err != nil {
		// A pattern that contains regex metacharacters but fails to
		// compile (e.g. "Trim(" with an unbalanced paren) is almost
		// always meant as a literal search. Fall back to literal
		// matching rather than surfacing a confusing regex error.
		return &literalMatcher{pat: []byte(p)}, nil
	}
	return &regexMatcher{re: re}, nil
}

// globMatch reports whether path (relative, forward-slash) matches
// pattern. Supports *, ?, [...] per segment plus ** for cross-segment
// matching. If the pattern has no '/' or '**' it falls back to a
// basename-only match — preserving the historical "*.go" behavior.
func globMatch(pattern, p string) bool {
	if !strings.Contains(pattern, "/") && !strings.Contains(pattern, "**") {
		ok, _ := path.Match(pattern, path.Base(p))
		return ok
	}
	patSegs := strings.Split(pattern, "/")
	pathSegs := strings.Split(p, "/")
	return matchSegments(patSegs, pathSegs)
}

func matchSegments(pat, p []string) bool {
	if len(pat) == 0 {
		return len(p) == 0
	}
	if pat[0] == "**" {
		// ** matches zero or more path segments. Try every
		// possible split.
		for i := 0; i <= len(p); i++ {
			if matchSegments(pat[1:], p[i:]) {
				return true
			}
		}
		return false
	}
	if len(p) == 0 {
		return false
	}
	ok, _ := path.Match(pat[0], p[0])
	if !ok {
		return false
	}
	return matchSegments(pat[1:], p[1:])
}

func (w *Workspace) readFile(ctx context.Context, in readFileInput) (string, error) {
	if in.Path == "" {
		return "", fmt.Errorf("read_file: path is required")
	}
	abs, err := w.safePath(in.Path)
	if err != nil {
		return "", err
	}
	f, err := os.Open(abs)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	defer f.Close()

	// No line range: read up to maxFileBytes directly.
	if in.StartLine == 0 && in.EndLine == 0 {
		limited := io.LimitReader(f, w.maxFileBytes+1)
		data, err := io.ReadAll(limited)
		if err != nil {
			return "", fmt.Errorf("read_file: %w", err)
		}
		truncated := ""
		if int64(len(data)) > w.maxFileBytes {
			data = data[:w.maxFileBytes]
			truncated = fmt.Sprintf("\n[truncated: file exceeds %d bytes]", w.maxFileBytes)
		}
		if mime := detectImageMIME(data, abs); mime != "" {
			// Image bytes can't be safely truncated — a chopped data URI
			// decodes to a corrupt image. Bail rather than mislead the
			// model with a partial attachment.
			if truncated != "" {
				return "", fmt.Errorf("read_file: image %q exceeds %d byte cap", in.Path, w.maxFileBytes)
			}
			luft.AttachImage(ctx, luft.ImageBlock{
				Source:    "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data),
				MediaType: mime,
			})
			return fmt.Sprintf("image: %s (%s, %d B)", w.relPath(abs), mime, len(data)), nil
		}
		return string(data) + truncated, nil
	}

	// Line range requested: scan line by line.
	start := in.StartLine
	end := in.EndLine
	if start < 1 {
		start = 1
	}

	var sb strings.Builder
	var byteCount int64
	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		if lineNum < start {
			continue
		}
		if end > 0 && lineNum > end {
			break
		}
		line := scanner.Text() + "\n"
		byteCount += int64(len(line))
		if byteCount > w.maxFileBytes {
			sb.WriteString(fmt.Sprintf("[truncated: output exceeds %d bytes]", w.maxFileBytes))
			break
		}
		sb.WriteString(line)
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	return sb.String(), nil
}

// imageExtMIME maps the extensions read_file recognises as images to their
// canonical IANA media types. Keep this aligned with the extension entries
// in binaryExts so the Grep skipper and the read_file image detector
// agree on what counts as an image.
var imageExtMIME = map[string]string{
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".gif":  "image/gif",
	".webp": "image/webp",
	".bmp":  "image/bmp",
	".tiff": "image/tiff",
}

// detectImageMIME returns a non-empty media type when data looks like an
// image. It sniffs the leading bytes via http.DetectContentType first
// (catches PNG/JPEG/GIF/WEBP/BMP regardless of extension), then falls back
// to the file extension for formats DetectContentType is conservative about.
func detectImageMIME(data []byte, path string) string {
	sniffLen := len(data)
	if sniffLen > 512 {
		sniffLen = 512
	}
	ct := http.DetectContentType(data[:sniffLen])
	// http.DetectContentType returns the media type with optional
	// "; charset=..." for text — we only care about image/* here.
	if i := strings.Index(ct, ";"); i >= 0 {
		ct = ct[:i]
	}
	ct = strings.TrimSpace(ct)
	if strings.HasPrefix(ct, "image/") {
		return ct
	}
	if mime, ok := imageExtMIME[strings.ToLower(filepath.Ext(path))]; ok {
		return mime
	}
	return ""
}

type fileInfoResult struct {
	Name    string    `json:"name"`
	Size    int64     `json:"size"`
	ModTime time.Time `json:"mod_time"`
	IsDir   bool      `json:"is_dir"`
	Mode    string    `json:"mode"`
}

func (w *Workspace) fileInfo(_ context.Context, in fileInfoInput) (string, error) {
	if in.Path == "" {
		return "", fmt.Errorf("file_info: path is required")
	}
	abs, err := w.safePath(in.Path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("file_info: %w", err)
	}
	out := fileInfoResult{
		Name:    info.Name(),
		Size:    info.Size(),
		ModTime: info.ModTime().UTC(),
		IsDir:   info.IsDir(),
		Mode:    info.Mode().String(),
	}
	data, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("file_info: marshal result: %w", err)
	}
	return string(data), nil
}

func (w *Workspace) editFile(_ context.Context, in editFileInput) (string, error) {
	if in.Path == "" {
		return "", fmt.Errorf("edit_file: path is required")
	}
	if in.OldString == "" {
		return "", fmt.Errorf("edit_file: old_string is required")
	}
	abs, err := w.safePath(in.Path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}
	content := string(data)
	count := strings.Count(content, in.OldString)
	if count == 0 {
		return "", fmt.Errorf("edit_file: old_string not found in %q", in.Path)
	}
	if in.ExpectedCount > 0 && count != in.ExpectedCount {
		return "", fmt.Errorf("edit_file: expected %d occurrence(s) of old_string, found %d", in.ExpectedCount, count)
	}
	updated := strings.ReplaceAll(content, in.OldString, in.NewString)
	if err := os.WriteFile(abs, []byte(updated), info.Mode()); err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}
	return fmt.Sprintf("replaced %d occurrence(s) in %s", count, in.Path), nil
}
