package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
)

func confirmBinding(name string) luft.ToolBinding {
	return luft.ToolBinding{
		Tool: luft.Tool{Name: name},
		Meta: luft.ToolMetadata{RequiresConfirmation: true},
	}
}

func TestConfirmerDecisions(t *testing.T) {
	bash := confirmBinding("bash")
	raw := json.RawMessage(`{"command":"ls"}`)

	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{"yes", "y\n", true},
		{"yes word", "yes\n", true},
		{"yes no newline", "y", true},
		{"no", "n\n", false},
		{"empty declines", "\n", false},
		{"eof declines", "", false},
		{"garbage declines", "maybe\n", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			confirm := makeConfirmer(false, newSpinner(&out), strings.NewReader(tt.input), &out)
			ok, err := confirm(context.Background(), bash, raw)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ok != tt.want {
				t.Errorf("decision = %v, want %v", ok, tt.want)
			}
			// The prompt and its key legend must always reach the user.
			if !strings.Contains(out.String(), "approve") {
				t.Errorf("prompt not shown; out = %q", out.String())
			}
		})
	}
}

func TestConfirmerApproveAllPerTool(t *testing.T) {
	var out bytes.Buffer
	// One reader for the whole session: only the first call consumes input.
	confirm := makeConfirmer(false, newSpinner(&out), strings.NewReader("a\n"), &out)
	bash := confirmBinding("bash")
	editor := confirmBinding("str_replace_based_edit_tool")
	raw := json.RawMessage(`{}`)

	if ok, _ := confirm(context.Background(), bash, raw); !ok {
		t.Fatal("expected approval on 'a'")
	}
	// Second bash call needs no input — pre-approved for the session.
	if ok, _ := confirm(context.Background(), bash, raw); !ok {
		t.Error("expected bash to be pre-approved after 'a'")
	}
	// A different tool still prompts, and declines on the now-empty reader.
	if ok, _ := confirm(context.Background(), editor, raw); ok {
		t.Error("approve-all for bash must not approve the editor")
	}
}

func TestConfirmerApproveAllTools(t *testing.T) {
	var out bytes.Buffer
	confirm := makeConfirmer(false, newSpinner(&out), strings.NewReader("A\n"), &out)
	raw := json.RawMessage(`{}`)

	if ok, _ := confirm(context.Background(), confirmBinding("bash"), raw); !ok {
		t.Fatal("expected approval on 'A'")
	}
	// Everything is pre-approved now — no further input consumed.
	if ok, _ := confirm(context.Background(), confirmBinding("str_replace_based_edit_tool"), raw); !ok {
		t.Error("expected all tools pre-approved after 'A'")
	}
}

func TestConfirmerAutoYes(t *testing.T) {
	var out bytes.Buffer
	confirm := makeConfirmer(true, newSpinner(&out), strings.NewReader(""), &out)
	if ok, _ := confirm(context.Background(), confirmBinding("bash"), json.RawMessage(`{}`)); !ok {
		t.Error("expected auto-yes to approve")
	}
	if out.Len() != 0 {
		t.Errorf("auto-yes should not prompt; out = %q", out.String())
	}
}

func TestConfirmerSkipsNonConfirmTools(t *testing.T) {
	var out bytes.Buffer
	confirm := makeConfirmer(false, newSpinner(&out), strings.NewReader(""), &out)
	readonly := luft.ToolBinding{Tool: luft.Tool{Name: "read_file"}} // RequiresConfirmation false
	if ok, _ := confirm(context.Background(), readonly, json.RawMessage(`{}`)); !ok {
		t.Error("read-only tool should not require confirmation")
	}
}
