package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// renderEditPreview returns a human-readable preview of an edit-tool call
// for the confirmation prompt. ok is false when the tool is not an editor
// or the input cannot be parsed — callers should fall back to JSON.
//
// Supported tool names match both the luft editor toolset and Anthropic's
// trained text_editor bindings.
func renderEditPreview(toolName string, input json.RawMessage) (string, bool) {
	switch toolName {
	case "multi_edit":
		return renderMultiEditPreview(input)
	case "str_replace_based_edit_tool", "str_replace_editor":
	default:
		return "", false
	}

	var in struct {
		Command    string `json:"command"`
		Path       string `json:"path"`
		OldStr     string `json:"old_str"`
		NewStr     string `json:"new_str"`
		FileText   string `json:"file_text"`
		InsertLine *int   `json:"insert_line"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", false
	}

	const maxBodyLines = 30

	var b strings.Builder
	switch in.Command {
	case "str_replace":
		fmt.Fprintf(&b, "  %s %s\n", grey("edit"), bold(in.Path))
		writeRemovedLines(&b, in.OldStr, maxBodyLines)
		writeAddedLines(&b, in.NewStr, maxBodyLines)
	case "create":
		stats := fmt.Sprintf("(new file, %s, %d lines)", humanBytes(len(in.FileText)), countLines(in.FileText))
		fmt.Fprintf(&b, "  %s %s %s\n", grey("create"), bold(in.Path), grey(stats))
		writeAddedLines(&b, in.FileText, maxBodyLines)
	case "insert":
		line := 0
		if in.InsertLine != nil {
			line = *in.InsertLine
		}
		fmt.Fprintf(&b, "  %s %s %s\n", grey("insert"), bold(in.Path), grey(fmt.Sprintf("(after line %d)", line)))
		writeAddedLines(&b, in.NewStr, maxBodyLines)
	default:
		// view (read-only) or unknown — let the caller fall back to JSON.
		return "", false
	}
	return b.String(), true
}

// renderMultiEditPreview renders a multi_edit call as a stacked diff, one
// removed/added block per edit, for the confirmation prompt.
func renderMultiEditPreview(input json.RawMessage) (string, bool) {
	var in struct {
		Path  string `json:"path"`
		Edits []struct {
			OldStr string `json:"old_str"`
			NewStr string `json:"new_str"`
		} `json:"edits"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", false
	}
	const maxBodyLines = 20
	var b strings.Builder
	fmt.Fprintf(&b, "  %s %s %s\n", grey("multi-edit"), bold(in.Path), grey(fmt.Sprintf("(%d edits)", len(in.Edits))))
	for i, ed := range in.Edits {
		fmt.Fprintf(&b, "  %s\n", grey(fmt.Sprintf("edit %d:", i+1)))
		writeRemovedLines(&b, ed.OldStr, maxBodyLines)
		writeAddedLines(&b, ed.NewStr, maxBodyLines)
	}
	return b.String(), true
}

func writeRemovedLines(b *strings.Builder, text string, maxLines int) {
	if text == "" {
		return
	}
	writePrefixedLines(b, text, "- ", red, maxLines)
}

func writeAddedLines(b *strings.Builder, text string, maxLines int) {
	if text == "" {
		return
	}
	writePrefixedLines(b, text, "+ ", green, maxLines)
}

func writePrefixedLines(b *strings.Builder, text, prefix string, paint func(string) string, maxLines int) {
	lines := strings.Split(text, "\n")
	n := len(lines)
	shown := n
	if n > maxLines {
		shown = maxLines
	}
	for i := 0; i < shown; i++ {
		fmt.Fprintf(b, "    %s\n", paint(prefix+lines[i]))
	}
	if n > shown {
		fmt.Fprintf(b, "    %s\n", grey(fmt.Sprintf("… %d more line(s)", n-shown)))
	}
}

func countLines(s string) int {
	if s == "" {
		return 0
	}
	n := strings.Count(s, "\n")
	if !strings.HasSuffix(s, "\n") {
		n++
	}
	return n
}
