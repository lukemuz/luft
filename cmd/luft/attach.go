package main

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/lukemuz/luft"
)

// atRefRe matches @-references of the form @path or @"path with spaces".
// The leading @ must be at a word boundary (start of line or preceded by
// whitespace) so it doesn't swallow email addresses or Twitter handles
// embedded in prose. Quoted form @"..." allows paths containing spaces.
var atRefRe = regexp.MustCompile(`(?:^|\s)@(?:` + `"([^"]+)"` + `|` + `(\S+)` + `)`)

// imageByExt maps a file extension to the IANA media type used when a
// referenced file is recognised as an image. Anything not listed here is
// inlined as text.
var imageByExt = map[string]string{
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".gif":  "image/gif",
	".webp": "image/webp",
	".bmp":  "image/bmp",
}

// maxAttachBytes caps the size of a single referenced file. Text beyond
// this is truncated with a marker; images are skipped with a warning so
// a huge screenshot can't silently blow the context budget.
const maxAttachBytes = 4 << 20 // 4 MiB

// expandAtReferences scans input for @path tokens, resolves each against
// cwd (relative paths; absolute paths and ~ are honoured), and returns:
//   - text: input with each @path replaced by an inlined, delimited block
//     containing the file contents (text files only)
//   - images: one ImageBlock per referenced image file (base64 data URI)
//   - warns: non-fatal problems (file missing, image too large, etc.),
//     already formatted for stderr
//
// Unrecognised @-tokens that aren't file paths (e.g. @username in prose)
// are left untouched. A @-token is only treated as a reference when the
// resolved path exists as a regular file.
func expandAtReferences(input, cwd string) (text string, images []luft.ImageBlock, warns []string) {
	var (
		out      strings.Builder
		seen     = map[string]bool{} // dedupe by resolved absolute path
		lastEnd  int
	)
	out.Grow(len(input))

	for _, m := range atRefRe.FindAllStringSubmatchIndex(input, -1) {
		// m = [allStart, allEnd, quotedStart, quotedEnd, bareStart, bareEnd]
		// Copy the text between the previous match and this one (including
		// the leading whitespace/newline that the regex consumed).
		out.WriteString(input[lastEnd:m[0]])
		// The leading @ is at m[0]+len(leading-ws); find it and drop it.
		at := indexByteFrom(input, '@', m[0])
		out.WriteString(input[m[0]:at]) // leading whitespace

		var ref string
		if m[2] >= 0 {
			ref = input[m[2]:m[3]] // quoted
		} else {
			ref = input[m[4]:m[5]] // bare
		}

		abs := resolveRef(ref, cwd)
		if abs == "" {
			// Not a resolvable path (e.g. @username). Emit verbatim.
			out.WriteString("@" + ref)
			lastEnd = m[1]
			continue
		}
		if seen[abs] {
			// Already inlined once; don't double up.
			out.WriteString("@" + ref)
			lastEnd = m[1]
			continue
		}
		seen[abs] = true

		info, err := os.Stat(abs)
		if err != nil || info.IsDir() {
			if looksLikePath(ref) {
				warns = append(warns, fmt.Sprintf("@%s: not a file", ref))
			}
			out.WriteString("@" + ref)
			lastEnd = m[1]
			continue
		}

		if mt, ok := imageByExt[strings.ToLower(filepath.Ext(abs))]; ok {
			img, w, ok := readImageAttachment(abs, mt, ref)
			if !ok {
				warns = append(warns, w)
				out.WriteString("@" + ref)
				lastEnd = m[1]
				continue
			}
			images = append(images, img)
			out.WriteString("@" + ref) // keep the mention in text so the model sees the marker
			lastEnd = m[1]
			continue
		}

		// Text file: inline under a delimited header.
		body, trunc, err := readTextAttachment(abs)
		if err != nil {
			warns = append(warns, fmt.Sprintf("@%s: %v", ref, err))
			out.WriteString("@" + ref)
			lastEnd = m[1]
			continue
		}
		out.WriteString(formatTextAttachment(ref, abs, body, trunc))
		lastEnd = m[1]
	}
	out.WriteString(input[lastEnd:])
	return out.String(), images, warns
}

// resolveRef turns an @-reference into an absolute path, or "" if it
// isn't a path-shaped token. It expands a leading ~ and resolves
// relative paths against cwd. A token with no path separators and no
// extension (e.g. "username") is treated as a non-reference so prose
// mentions aren't mistaken for files.
func resolveRef(ref, cwd string) string {
	if ref == "" || !looksLikePath(ref) {
		return ""
	}
	if strings.HasPrefix(ref, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		ref = filepath.Join(home, strings.TrimPrefix(ref, "~"))
	}
	if !filepath.IsAbs(ref) {
		ref = filepath.Join(cwd, ref)
	}
	// Cleaned but not yet stat'd; caller checks existence.
	return filepath.Clean(ref)
}

// looksLikePath reports whether ref is shaped like a file path: it has a
// path separator, a leading ~, or a file extension. Bare words like
// "username" or "sam" return false so prose mentions aren't mistaken for
// references.
func looksLikePath(ref string) bool {
	if ref == "" {
		return false
	}
	if strings.HasPrefix(ref, "~") {
		return true
	}
	if strings.ContainsAny(ref, `/\`) {
		return true
	}
	if i := strings.LastIndexByte(ref, '.'); i > 0 {
		// has an extension that isn't the whole token
		return true
	}
	return false
}

// readTextAttachment reads up to maxAttachBytes of a text file. If the
// file is larger, the body is truncated and trunc is true.
func readTextAttachment(abs string) (body string, trunc bool, err error) {
	f, err := os.Open(abs)
	if err != nil {
		return "", false, err
	}
	defer f.Close()
	b, err := readCap(f, maxAttachBytes+1)
	if err != nil {
		return "", false, err
	}
	if len(b) > maxAttachBytes {
		return string(b[:maxAttachBytes]), true, nil
	}
	return string(b), false, nil
}

// readImageAttachment reads an image file, base64-encodes it, and returns
// a data-URI ImageBlock. ok is false when the file is too large or can't
// be read; warn carries the reason.
func readImageAttachment(abs, mediaType, ref string) (img luft.ImageBlock, warn string, ok bool) {
	b, err := os.ReadFile(abs)
	if err != nil {
		return luft.ImageBlock{}, fmt.Sprintf("@%s: %v", ref, err), false
	}
	if len(b) > maxAttachBytes {
		return luft.ImageBlock{}, fmt.Sprintf("@%s: image %s, exceeds %s — skipped", ref, humanBytes(len(b)), humanBytes(maxAttachBytes)), false
	}
	return luft.ImageBlock{
		Source:    "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(b),
		MediaType: mediaType,
	}, "", true
}

// readCap reads up to n bytes from r.
func readCap(r interface{ Read([]byte) (int, error) }, n int) ([]byte, error) {
	buf := make([]byte, n)
	got, err := r.Read(buf)
	return buf[:got], err
}

// formatTextAttachment renders an inlined file as a delimited block.
func formatTextAttachment(ref, abs, body string, trunc bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, "@%s (%s):\n", ref, abs)
	b.WriteString("---\n")
	b.WriteString(strings.TrimRight(body, "\n"))
	b.WriteString("\n")
	if trunc {
		fmt.Fprintf(&b, "[… truncated at %s]\n", humanBytes(maxAttachBytes))
	}
	b.WriteString("---")
	return b.String()
}

// indexByteFrom is strings.IndexByte but starting at offset i.
func indexByteFrom(s string, c byte, from int) int {
	if from < 0 {
		from = 0
	}
	if from >= len(s) {
		return -1
	}
	if j := strings.IndexByte(s[from:], c); j >= 0 {
		return from + j
	}
	return -1
}
