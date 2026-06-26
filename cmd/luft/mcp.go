package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/mcp"
)

// mcpServerConfig is one server entry from the MCP config file.
type mcpServerConfig struct {
	Command string            `json:"command"`
	Args    []string          `json:"args"`
	Env     map[string]string `json:"env"`
}

// mcpServersConfig is the parsed config file shape: the conventional
// {"mcpServers": {name: {command, args, env}}} object.
type mcpServersConfig struct {
	MCPServers map[string]mcpServerConfig `json:"mcpServers"`
}

// mcpServerInfo is the session-side metadata for the /mcp listing: the
// server name and a snapshot of its tools (name + description + schema),
// captured at startup. It holds no live ToolFuncs.
type mcpServerInfo struct {
	Name  string
	Tools []luft.Tool
}

// resolveMCPConfigPath returns the MCP config file path per the lookup
// precedence:
//  1. the explicit -mcp flag, if set;
//  2. ./.mcp.json in the working directory;
//  3. ~/.config/luft/mcp.json.
//
// The first existing file wins. When no default file is present, MCP is
// off: it returns ("", nil). An explicit flag pointing at a missing file
// is a hard error (the user asked for something specific).
func resolveMCPConfigPath(explicit string) (string, error) {
	if explicit != "" {
		if _, err := os.Stat(explicit); err != nil {
			return "", fmt.Errorf("mcp: config file %q not found: %w", explicit, err)
		}
		return explicit, nil
	}
	for _, p := range []string{".mcp.json"} {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		p := filepath.Join(home, ".config", "luft", "mcp.json")
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", nil
}

// loadMCPConfig reads and parses the MCP config file at path. A path of ""
// yields an empty config (MCP off). Pure: file path in, config out, no
// subprocesses.
func loadMCPConfig(path string) (mcpServersConfig, error) {
	if path == "" {
		return mcpServersConfig{}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return mcpServersConfig{}, fmt.Errorf("mcp: read config %q: %w", path, err)
	}
	var cfg mcpServersConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return mcpServersConfig{}, fmt.Errorf("mcp: parse config %q: %w", path, err)
	}
	return cfg, nil
}

// connectMCP connects every server in cfg (best-effort: failures log a
// warning to out and are skipped), fetches each Toolset, marks every
// binding RequiresConfirmation so the shared confirmer prompts on each
// call, wraps them with the standard tool middleware stack, and returns
// the joined Toolset plus per-server info and the live connections (for
// the caller to Close on exit). Returns an empty Toolset and nil error
// if no servers connected — MCP simply off, exactly as today.
func connectMCP(
	ctx context.Context,
	cfg mcpServersConfig,
	confirm func(context.Context, luft.ToolBinding, json.RawMessage) (bool, error),
	logger luft.Logger,
	out io.Writer,
) (tools luft.Toolset, servers []mcpServerInfo, conns []*mcp.Server) {
	names := make([]string, 0, len(cfg.MCPServers))
	for name := range cfg.MCPServers {
		names = append(names, name)
	}
	sort.Strings(names)

	stack := []luft.Middleware{
		luft.WithConfirmation(confirm),
		luft.WithTimeout(60 * time.Second),
		luft.WithResultLimit(64 * 1024),
		luft.WithLogging(logger),
	}

	seen := map[string]bool{}
	for _, name := range names {
		sc := cfg.MCPServers[name]
		if sc.Command == "" {
			fmt.Fprintf(out, "  %s mcp server %q: no command set, skipping\n", yellow("⚠"), name)
			continue
		}
		// Connect under the long-lived ctx so the server process lives for the
		// whole session: mcp.Connect ties the child to its context via
		// exec.CommandContext, so a cancelled or timeout context would KILL the
		// server right after connecting and break every later tool call. The
		// startup timeout is enforced with a select, NOT via the process's
		// context, so a hung handshake can't stall startup yet a server that
		// connects fine keeps running.
		type connectResult struct {
			srv *mcp.Server
			err error
		}
		resCh := make(chan connectResult, 1)
		go func() {
			srv, err := mcp.Connect(ctx, mcp.Config{
				Command: sc.Command,
				Args:    sc.Args,
				Env:     envMapToSlice(sc.Env),
			})
			resCh <- connectResult{srv, err}
		}()
		var srv *mcp.Server
		select {
		case res := <-resCh:
			if res.err != nil {
				fmt.Fprintf(out, "  %s mcp server %q: connect failed: %v\n", yellow("⚠"), name, res.err)
				continue
			}
			srv = res.srv
		case <-time.After(10 * time.Second):
			fmt.Fprintf(out, "  %s mcp server %q: connect timed out after 10s, skipping\n", yellow("⚠"), name)
			continue
		}
		ts, err := srv.Toolset(ctx)
		if err != nil {
			fmt.Fprintf(out, "  %s mcp server %q: toolset failed: %v\n", yellow("⚠"), name, err)
			_ = srv.Close()
			continue
		}

		// MCP tools are external and may have side effects: require
		// confirmation on each binding so the shared confirmer prompts
		// (it auto-approves bindings without this flag).
		for i := range ts.Bindings {
			ts.Bindings[i].Meta.RequiresConfirmation = true
		}
		wrapped := ts.Wrap(stack...)

		// De-duplicate tool names defensively: a later server (or a
		// built-in) exposing the same name is skipped with a warning
		// rather than crashing MustJoin in main.
		var deduped []luft.ToolBinding
		for _, b := range wrapped.Bindings {
			if seen[b.Tool.Name] {
				fmt.Fprintf(out, "  %s mcp server %q: duplicate tool %q, skipping\n", yellow("⚠"), name, b.Tool.Name)
				continue
			}
			seen[b.Tool.Name] = true
			deduped = append(deduped, b)
		}

		tools.Bindings = append(tools.Bindings, deduped...)
		servers = append(servers, mcpServerInfo{Name: name, Tools: wrapped.Tools()})
		conns = append(conns, srv)
	}
	return tools, servers, conns
}

// envMapToSlice converts the config's KEY→VALUE map into the "KEY=VALUE"
// slice that exec.Cmd expects. A nil/empty map yields nil, so mcp.Connect
// inherits the parent environment.
func envMapToSlice(env map[string]string) []string {
	if len(env) == 0 {
		return nil
	}
	out := make([]string, 0, len(env))
	for k, v := range env {
		out = append(out, k+"="+v)
	}
	return out
}
