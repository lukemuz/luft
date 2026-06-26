package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
)

// discardLogger satisfies luft.Logger for tests.
type discardLogger struct{}

func (discardLogger) Info(string, ...any)  {}
func (discardLogger) Error(string, ...any) {}

// fakeMCPServerPath builds the fake MCP server (mcp/testdata/fakeserver)
// and returns the binary path. Skips the test if 'go build' is unavailable.
func fakeMCPServerPath(t *testing.T) string {
	t.Helper()
	out := filepath.Join(t.TempDir(), "fake-mcp-server")
	if runtime.GOOS == "windows" {
		out += ".exe" // Windows needs the extension to exec the built binary
	}
	// Build by import path so this doesn't depend on the test's working
	// directory (the previous "../mcp" relative dir resolved to cmd/mcp from
	// here and silently skipped the test).
	cmd := exec.Command("go", "build", "-o", out, "github.com/lukemuz/luft/mcp/testdata/fakeserver")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("cannot build fake MCP server: %v\n%s", err, b)
	}
	return out
}

func TestLoadMCPConfig(t *testing.T) {
	t.Run("valid two-server config", func(t *testing.T) {
		cfg := `{"mcpServers": {
			"demo": {"command": "npx", "args": ["-y", "@scope/server"], "env": {"TOKEN": "abc"}},
			"weather": {"command": "node", "args": ["server.js"]}
		}}`
		path := filepath.Join(t.TempDir(), "mcp.json")
		if err := os.WriteFile(path, []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
		got, err := loadMCPConfig(path)
		if err != nil {
			t.Fatalf("loadMCPConfig: %v", err)
		}
		if len(got.MCPServers) != 2 {
			t.Fatalf("expected 2 servers, got %d", len(got.MCPServers))
		}
		demo, ok := got.MCPServers["demo"]
		if !ok {
			t.Fatal("missing demo server")
		}
		if demo.Command != "npx" {
			t.Errorf("demo.Command = %q, want npx", demo.Command)
		}
		if len(demo.Args) != 2 || demo.Args[1] != "@scope/server" {
			t.Errorf("demo.Args = %v, want [-y @scope/server]", demo.Args)
		}
		if demo.Env["TOKEN"] != "abc" {
			t.Errorf("demo.Env[TOKEN] = %q, want abc", demo.Env["TOKEN"])
		}
	})

	t.Run("empty mcpServers object", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "mcp.json")
		if err := os.WriteFile(path, []byte(`{"mcpServers": {}}`), 0o644); err != nil {
			t.Fatal(err)
		}
		got, err := loadMCPConfig(path)
		if err != nil {
			t.Fatalf("loadMCPConfig: %v", err)
		}
		if len(got.MCPServers) != 0 {
			t.Errorf("expected 0 servers, got %d", len(got.MCPServers))
		}
	})

	t.Run("missing file", func(t *testing.T) {
		_, err := loadMCPConfig(filepath.Join(t.TempDir(), "nope.json"))
		if err == nil {
			t.Fatal("expected error for missing file")
		}
	})

	t.Run("malformed json", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "mcp.json")
		if err := os.WriteFile(path, []byte(`{not json`), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := loadMCPConfig(path)
		if err == nil {
			t.Fatal("expected error for malformed JSON")
		}
	})

	t.Run("empty path returns empty config", func(t *testing.T) {
		got, err := loadMCPConfig("")
		if err != nil {
			t.Fatalf("loadMCPConfig(\"\"): %v", err)
		}
		if len(got.MCPServers) != 0 {
			t.Errorf("expected 0 servers, got %d", len(got.MCPServers))
		}
	})
}

func TestResolveMCPConfigPath(t *testing.T) {
	t.Run("explicit missing file errors", func(t *testing.T) {
		_, err := resolveMCPConfigPath(filepath.Join(t.TempDir(), "nope.json"))
		if err == nil {
			t.Fatal("expected error for explicit missing file")
		}
	})
	t.Run("empty explicit with no defaults returns empty", func(t *testing.T) {
		// Run in a temp dir with no .mcp.json and HOME pointed elsewhere
		// so no default file is found.
		dir := t.TempDir()
		t.Setenv("HOME", dir)
		old, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		defer os.Chdir(old)
		if err := os.Chdir(dir); err != nil {
			t.Fatal(err)
		}
		got, err := resolveMCPConfigPath("")
		if err != nil {
			t.Fatalf("resolveMCPConfigPath: %v", err)
		}
		if got != "" {
			t.Errorf("expected empty path, got %q", got)
		}
	})
	t.Run("explicit existing file wins", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "mcp.json")
		if err := os.WriteFile(path, []byte(`{"mcpServers": {}}`), 0o644); err != nil {
			t.Fatal(err)
		}
		got, err := resolveMCPConfigPath(path)
		if err != nil {
			t.Fatalf("resolveMCPConfigPath: %v", err)
		}
		if got != path {
			t.Errorf("expected %q, got %q", path, got)
		}
	})
}

func TestConnectMCP(t *testing.T) {
	serverBin := fakeMCPServerPath(t)

	cfg := mcpServersConfig{MCPServers: map[string]mcpServerConfig{
		"demo": {Command: serverBin},
	}}

	// A confirmer that auto-approves, so the wrapped tool can be invoked.
	confirm := func(context.Context, luft.ToolBinding, json.RawMessage) (bool, error) {
		return true, nil
	}

	ctx := context.Background()
	var warn bytes.Buffer
	tools, servers, conns := connectMCP(ctx, cfg, confirm, discardLogger{}, &warn)
	defer func() {
		for _, c := range conns {
			_ = c.Close()
		}
	}()

	if warn.Len() > 0 {
		t.Errorf("unexpected warnings: %s", warn.String())
	}
	if len(servers) != 1 {
		t.Fatalf("expected 1 server, got %d (warnings: %s)", len(servers), warn.String())
	}
	if servers[0].Name != "demo" {
		t.Errorf("server name = %q, want demo", servers[0].Name)
	}
	if len(servers[0].Tools) != 2 {
		t.Errorf("expected 2 tools, got %d", len(servers[0].Tools))
	}

	// The fake server exposes "echo" and "fail"; both must be present and
	// require confirmation.
	names := map[string]bool{}
	for _, b := range tools.Bindings {
		names[b.Tool.Name] = true
		if !b.Meta.RequiresConfirmation {
			t.Errorf("tool %q: RequiresConfirmation not set", b.Tool.Name)
		}
		if b.Meta.Source != "mcp" {
			t.Errorf("tool %q: Source = %q, want mcp", b.Tool.Name, b.Meta.Source)
		}
	}
	if !names["echo"] {
		t.Error("missing echo tool")
	}
	if !names["fail"] {
		t.Error("missing fail tool")
	}

	// End-to-end: invoking "echo" through the wrapped binding returns the
	// echoed message (the confirmer auto-approves).
	dispatch := tools.Dispatch()
	fn, ok := dispatch["echo"]
	if !ok {
		t.Fatal("echo not in dispatch map")
	}
	input, _ := json.Marshal(map[string]string{"message": "hello mcp"})
	out, err := fn(ctx, input)
	if err != nil {
		t.Fatalf("echo call: %v", err)
	}
	if out != "hello mcp" {
		t.Errorf("echo output = %q, want %q", out, "hello mcp")
	}
}

func TestConnectMCPSkipsBadServer(t *testing.T) {
	cfg := mcpServersConfig{MCPServers: map[string]mcpServerConfig{
		"broken": {Command: "/nonexistent/binary-that-does-not-exist"},
	}}
	ctx := context.Background()
	var warn bytes.Buffer
	tools, servers, conns := connectMCP(ctx, cfg, nil, discardLogger{}, &warn)
	defer func() {
		for _, c := range conns {
			_ = c.Close()
		}
	}()

	if len(servers) != 0 {
		t.Errorf("expected 0 servers, got %d", len(servers))
	}
	if len(tools.Bindings) != 0 {
		t.Errorf("expected 0 tools, got %d", len(tools.Bindings))
	}
	if !strings.Contains(warn.String(), "broken") {
		t.Errorf("expected warning mentioning 'broken', got: %s", warn.String())
	}
}

func TestConnectMCPSkipsNoCommand(t *testing.T) {
	cfg := mcpServersConfig{MCPServers: map[string]mcpServerConfig{
		"empty": {},
	}}
	ctx := context.Background()
	var warn bytes.Buffer
	_, servers, conns := connectMCP(ctx, cfg, nil, discardLogger{}, &warn)
	defer func() {
		for _, c := range conns {
			_ = c.Close()
		}
	}()

	if len(servers) != 0 {
		t.Errorf("expected 0 servers for empty command, got %d", len(servers))
	}
	if !strings.Contains(warn.String(), "empty") {
		t.Errorf("expected warning mentioning 'empty', got: %s", warn.String())
	}
}

func TestEnvMapToSlice(t *testing.T) {
	t.Run("nil map returns nil", func(t *testing.T) {
		if got := envMapToSlice(nil); got != nil {
			t.Errorf("expected nil, got %v", got)
		}
	})
	t.Run("empty map returns nil", func(t *testing.T) {
		if got := envMapToSlice(map[string]string{}); got != nil {
			t.Errorf("expected nil, got %v", got)
		}
	})
	t.Run("entries are KEY=VALUE", func(t *testing.T) {
		got := envMapToSlice(map[string]string{"TOKEN": "abc", "FOO": "bar"})
		if len(got) != 2 {
			t.Fatalf("expected 2 entries, got %d", len(got))
		}
		want := map[string]bool{"TOKEN=abc": true, "FOO=bar": true}
		for _, e := range got {
			if !want[e] {
				t.Errorf("unexpected entry %q", e)
			}
		}
	})
}
