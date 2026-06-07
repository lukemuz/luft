// Package mcp provides a client for the Model Context Protocol (MCP).
// It connects to an MCP server over stdio, lists the tools it exposes,
// and adapts them into ordinary luft.Tool and luft.ToolFunc values so
// they compose naturally with any Toolset or dispatch map.
//
// Usage:
//
//	srv, err := mcp.Connect(ctx, mcp.Config{
//		Command: "my-mcp-server",
//		Args:    []string{"--stdio"},
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer srv.Close()
//
//	toolset, err := srv.Toolset(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	result, err := client.Loop(ctx, system, history, toolset, 10)
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"sync/atomic"

	"github.com/lukemuz/luft"
)

// Config holds the parameters for connecting to an MCP server.
type Config struct {
	// Command is the executable to run (e.g. "npx", "python3").
	Command string
	// Args are additional arguments passed to Command.
	Args []string
	// Env sets extra environment variables for the child process.
	// Each entry should be in "KEY=VALUE" form.
	Env []string
}

// Server is a live connection to an MCP server process. It is safe for
// concurrent use. Call Close when done to terminate the child process.
type Server struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner

	mu      sync.Mutex
	pending map[int64]chan jsonRPCResponse

	nextID atomic.Int64
	closed chan struct{}
}

// Connect starts the MCP server described by cfg and performs the
// MCP initialization handshake. The returned *Server is ready to use.
func Connect(ctx context.Context, cfg Config) (*Server, error) {
	if cfg.Command == "" {
		return nil, fmt.Errorf("mcp: Config.Command is required")
	}

	cmd := exec.CommandContext(ctx, cfg.Command, cfg.Args...)
	if len(cfg.Env) > 0 {
		cmd.Env = append(cmd.Environ(), cfg.Env...)
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: open stdin pipe: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: open stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("mcp: start server %q: %w", cfg.Command, err)
	}

	srv := &Server{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  bufio.NewScanner(stdoutPipe),
		pending: make(map[int64]chan jsonRPCResponse),
		closed:  make(chan struct{}),
	}

	go srv.readLoop()

	if err := srv.initialize(ctx); err != nil {
		_ = srv.Close()
		return nil, err
	}
	return srv, nil
}

// Close terminates the MCP server process and releases resources.
func (s *Server) Close() error {
	select {
	case <-s.closed:
		return nil
	default:
		close(s.closed)
	}
	_ = s.stdin.Close()
	return s.cmd.Process.Kill()
}

// Toolset queries the MCP server for its tool list and returns an
// luft.Toolset containing one ToolBinding per MCP tool. Each binding's
// ToolFunc forwards calls to the MCP server via the tools/call method.
// The returned Toolset is a snapshot; call Toolset again to refresh.
func (s *Server) Toolset(ctx context.Context) (luft.Toolset, error) {
	resp, err := s.call(ctx, "tools/list", nil)
	if err != nil {
		return luft.Toolset{}, fmt.Errorf("mcp: list tools: %w", err)
	}

	var payload struct {
		Tools []mcpTool `json:"tools"`
	}
	if err := json.Unmarshal(resp, &payload); err != nil {
		return luft.Toolset{}, fmt.Errorf("mcp: decode tools/list response: %w", err)
	}

	bindings := make([]luft.ToolBinding, 0, len(payload.Tools))
	for _, mt := range payload.Tools {
		t, err := mcpToolToAgent(mt)
		if err != nil {
			return luft.Toolset{}, err
		}
		fn := s.makeToolFunc(mt.Name)
		bindings = append(bindings, luft.ToolBinding{
			Tool: t,
			Func: fn,
			Meta: luft.ToolMetadata{Source: "mcp"},
		})
	}
	return luft.Toolset{Bindings: bindings}, nil
}

// mcpTool is the wire representation of an MCP tool from tools/list.
type mcpTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// mcpToolToAgent converts an MCP wire tool into a luft.Tool.
// The inputSchema from MCP is already JSON Schema, so we pass it through
// as raw bytes rather than re-serialising through luft.InputSchema.
func mcpToolToAgent(mt mcpTool) (luft.Tool, error) {
	schema := mt.InputSchema
	if len(schema) == 0 {
		schema = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return luft.Tool{
		Name:        mt.Name,
		Description: mt.Description,
		InputSchema: schema,
	}, nil
}

// makeToolFunc returns a luft.ToolFunc that calls the named MCP tool.
func (s *Server) makeToolFunc(toolName string) luft.ToolFunc {
	return func(ctx context.Context, input json.RawMessage) (string, error) {
		params := map[string]interface{}{
			"name":      toolName,
			"arguments": input,
		}
		raw, err := s.call(ctx, "tools/call", params)
		if err != nil {
			return "", fmt.Errorf("mcp: call tool %q: %w", toolName, err)
		}

		// MCP tools/call returns { content: [{type, text}], isError: bool }
		var result struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		}
		if err := json.Unmarshal(raw, &result); err != nil {
			return "", fmt.Errorf("mcp: decode tools/call response for %q: %w", toolName, err)
		}
		var text string
		for _, c := range result.Content {
			if c.Type == "text" {
				text += c.Text
			}
		}
		if result.IsError {
			return "", fmt.Errorf("%s", text)
		}
		return text, nil
	}
}

// ---- JSON-RPC 2.0 transport ------------------------------------------------

type jsonRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	ID     int64           `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  *jsonRPCError   `json:"error"`
}

type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *jsonRPCError) Error() string {
	return fmt.Sprintf("mcp: rpc error %d: %s", e.Code, e.Message)
}

// call sends a JSON-RPC request and waits for the matching response.
func (s *Server) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := s.nextID.Add(1)
	ch := make(chan jsonRPCResponse, 1)

	s.mu.Lock()
	s.pending[id] = ch
	s.mu.Unlock()

	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}
	line, err := json.Marshal(req)
	if err != nil {
		s.mu.Lock()
		delete(s.pending, id)
		s.mu.Unlock()
		return nil, fmt.Errorf("mcp: marshal request: %w", err)
	}
	line = append(line, '\n')

	s.mu.Lock()
	_, writeErr := s.stdin.Write(line)
	s.mu.Unlock()
	if writeErr != nil {
		s.mu.Lock()
		delete(s.pending, id)
		s.mu.Unlock()
		return nil, fmt.Errorf("mcp: write request: %w", writeErr)
	}

	select {
	case <-ctx.Done():
		s.mu.Lock()
		delete(s.pending, id)
		s.mu.Unlock()
		return nil, ctx.Err()
	case <-s.closed:
		return nil, fmt.Errorf("mcp: server closed")
	case resp := <-ch:
		if resp.Error != nil {
			return nil, resp.Error
		}
		return resp.Result, nil
	}
}

// readLoop continuously reads lines from the server's stdout and delivers
// each JSON-RPC response to the waiting caller. It runs until the server
// closes or the pipe is broken.
func (s *Server) readLoop() {
	for s.stdout.Scan() {
		line := s.stdout.Bytes()
		var resp jsonRPCResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			continue // skip malformed lines (e.g. server log output)
		}
		s.mu.Lock()
		ch, ok := s.pending[resp.ID]
		if ok {
			delete(s.pending, resp.ID)
		}
		s.mu.Unlock()
		if ok {
			ch <- resp
		}
	}
	// Drain any pending callers so they don't block forever.
	s.mu.Lock()
	for id, ch := range s.pending {
		ch <- jsonRPCResponse{ID: id, Error: &jsonRPCError{Code: -1, Message: "server stdout closed"}}
		delete(s.pending, id)
	}
	s.mu.Unlock()
}

// initialize performs the MCP initialization handshake.
func (s *Server) initialize(ctx context.Context) error {
	params := map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]interface{}{},
		"clientInfo": map[string]interface{}{
			"name":    "luft",
			"version": "0.2.0",
		},
	}
	_, err := s.call(ctx, "initialize", params)
	if err != nil {
		return fmt.Errorf("mcp: initialize: %w", err)
	}
	// Send the initialized notification (no response expected).
	notif := struct {
		JSONRPC string `json:"jsonrpc"`
		Method  string `json:"method"`
	}{JSONRPC: "2.0", Method: "notifications/initialized"}
	line, _ := json.Marshal(notif)
	line = append(line, '\n')
	s.mu.Lock()
	_, _ = s.stdin.Write(line)
	s.mu.Unlock()
	return nil
}
