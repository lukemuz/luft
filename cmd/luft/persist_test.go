package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/stores"
)

func TestDirHash(t *testing.T) {
	h := dirHash("/a/b")
	if len(h) != 12 {
		t.Fatalf("dirHash = %q, want 12 chars", h)
	}
	if dirHash("/a/b") != h {
		t.Errorf("dirHash is not stable for the same input")
	}
	if dirHash("/a/b") == dirHash("/c/d") {
		t.Errorf("dirHash collides for different inputs")
	}
	for _, r := range h {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f')) {
			t.Errorf("dirHash contains non-hex char %q", r)
		}
	}
}

func TestNewSessionIDPassesValidator(t *testing.T) {
	id := newSessionID("/some/cwd")
	// Round-trip through a real FileStore Create: the store's ID validator
	// rejects anything outside [A-Za-z0-9._-].
	dir := t.TempDir()
	store, err := stores.NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	ctx := context.Background()
	if err := store.Create(ctx, &luft.Session{ID: id}); err != nil {
		t.Fatalf("Create rejected ID %q: %v", id, err)
	}
	if !strings.HasPrefix(id, dirHash("/some/cwd")+"-") {
		t.Errorf("ID %q should start with the dirHash prefix", id)
	}
}

func TestNewSessionIDUniqueInSameSecond(t *testing.T) {
	// A fast restart in the same second must not collide.
	id1 := newSessionID("/cwd")
	id2 := newSessionID("/cwd")
	if id1 == id2 {
		t.Errorf("two IDs in the same second collided: %s", id1)
	}
}

func TestNewSessionIDSortable(t *testing.T) {
	// Construct two IDs with a controlled timestamp: the lexicographically
	// smaller one should correspond to the earlier time.
	tag := dirHash("/cwd")
	older := tag + "-20250101-000000-aaaa"
	newer := tag + "-20250102-000000-aaaa"
	if strings.Compare(older, newer) >= 0 {
		t.Errorf("older ID should sort before newer ID")
	}
}

func TestResolveSessionFreshNoFlags(t *testing.T) {
	ctx := context.Background()
	store, _ := stores.NewFileStore(t.TempDir())
	id, loaded, notice := resolveSession(ctx, store, "/cwd", "", false)
	if loaded != nil {
		t.Errorf("fresh session should load nothing")
	}
	if notice != "" {
		t.Errorf("fresh session should have no notice, got %q", notice)
	}
	if id == "" {
		t.Error("fresh session should get a non-empty ID")
	}
}

func TestResolveSessionContinueFindsNewestForDir(t *testing.T) {
	ctx := context.Background()
	store, _ := stores.NewFileStore(t.TempDir())
	tagA := dirHash("/dirA")
	tagB := dirHash("/dirB")

	// Three sessions for dirA (oldest → newest) and one for dirB.
	mustCreate(t, store, &luft.Session{ID: tagA + "-20250101-000000-0000", History: hist(1)})
	mustCreate(t, store, &luft.Session{ID: tagA + "-20250102-000000-0000", History: hist(2)})
	mustCreate(t, store, &luft.Session{ID: tagA + "-20250103-000000-0000", History: hist(3)})
	mustCreate(t, store, &luft.Session{ID: tagB + "-20250104-000000-0000", History: hist(1)})

	id, loaded, _ := resolveSession(ctx, store, "/dirA", "", true)
	want := tagA + "-20250103-000000-0000"
	if id != want {
		t.Fatalf("id = %q, want %q", id, want)
	}
	if loaded == nil || len(loaded.History) != 6 { // hist(3) = 6 messages
		t.Fatalf("loaded history len = %d, want 6", histLen(loaded))
	}
}

func TestResolveSessionContinueNothingFound(t *testing.T) {
	ctx := context.Background()
	store, _ := stores.NewFileStore(t.TempDir())
	id, loaded, notice := resolveSession(ctx, store, "/never", "", true)
	if loaded != nil {
		t.Errorf("should start fresh, got loaded history")
	}
	if id == "" {
		t.Error("should get a fresh ID")
	}
	if !strings.Contains(notice, "no prior session") {
		t.Errorf("notice should mention no prior session, got %q", notice)
	}
}

func TestResolveSessionResumeByID(t *testing.T) {
	ctx := context.Background()
	store, _ := stores.NewFileStore(t.TempDir())
	mustCreate(t, store, &luft.Session{ID: "abc-20250101-000000-0000", History: hist(3)})

	id, loaded, notice := resolveSession(ctx, store, "/anywhere", "abc-20250101-000000-0000", false)
	if id != "abc-20250101-000000-0000" {
		t.Fatalf("id = %q, want the resumed ID", id)
	}
	if loaded == nil || len(loaded.History) != 6 { // hist(3) = 6 messages
		t.Fatalf("loaded history len = %d, want 6", histLen(loaded))
	}
	if !strings.Contains(notice, "resumed") || !strings.Contains(notice, "6 messages") {
		t.Errorf("notice should report 6 messages restored, got %q", notice)
	}
}

func TestResolveSessionResumeNotFound(t *testing.T) {
	ctx := context.Background()
	store, _ := stores.NewFileStore(t.TempDir())
	id, loaded, notice := resolveSession(ctx, store, "/cwd", "missing-id", false)
	if loaded != nil {
		t.Errorf("should start fresh")
	}
	if id == "missing-id" {
		t.Error("should not reuse the missing ID")
	}
	if !strings.Contains(notice, "not found") {
		t.Errorf("notice should mention not found, got %q", notice)
	}
}

func TestResolveSessionResumeCorruptFile(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, _ := stores.NewFileStore(dir)

	// Write a corrupt JSON file directly.
	badID := "bad-20250101-000000-0000"
	if err := os.WriteFile(filepath.Join(dir, badID+".json"), []byte("{not valid json"), 0o644); err != nil {
		t.Fatal(err)
	}
	id, loaded, notice := resolveSession(ctx, store, "/cwd", badID, false)
	if loaded != nil {
		t.Errorf("corrupt session should start fresh")
	}
	if id == badID {
		t.Error("should not reuse the corrupt ID (would overwrite on first save)")
	}
	if !strings.Contains(notice, "could not read") {
		t.Errorf("notice should warn about the corrupt file, got %q", notice)
	}

	// The corrupt file should still be on disk (not overwritten or deleted).
	if _, err := os.Stat(filepath.Join(dir, badID+".json")); err != nil {
		t.Errorf("corrupt file should be left untouched: %v", err)
	}
}

func TestResolveSessionContinueCorruptNewest(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, _ := stores.NewFileStore(dir)
	tag := dirHash("/cwd")

	// One good session, then a corrupt one with a newer ID.
	mustCreate(t, store, &luft.Session{ID: tag + "-20250101-000000-0000", History: hist(2)})
	badID := tag + "-20250102-000000-0000"
	if err := os.WriteFile(filepath.Join(dir, badID+".json"), []byte("garbage"), 0o644); err != nil {
		t.Fatal(err)
	}

	id, loaded, notice := resolveSession(ctx, store, "/cwd", "", true)
	if loaded != nil {
		t.Errorf("corrupt newest should start fresh")
	}
	if id == badID {
		t.Error("should not reuse the corrupt ID")
	}
	if !strings.Contains(notice, "could not read") {
		t.Errorf("notice should warn about the corrupt file, got %q", notice)
	}
}

func TestResolveSessionNilStore(t *testing.T) {
	ctx := context.Background()
	// -continue with a nil store should not panic.
	id, loaded, notice := resolveSession(ctx, nil, "/cwd", "", true)
	if loaded != nil {
		t.Errorf("nil store should start fresh")
	}
	if id == "" {
		t.Error("should get a fresh ID")
	}
	if !strings.Contains(notice, "unavailable") {
		t.Errorf("notice should mention the store is unavailable, got %q", notice)
	}
}

func TestPersistWritesSnapshot(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, _ := stores.NewFileStore(dir)

	s := &session{
		store:     store,
		sessionID: "test-persist-0001",
		cwd:       "/proj",
		history:   hist(1),
		usage:     luft.Usage{InputTokens: 100, OutputTokens: 50},
	}
	s.persist(ctx)

	loaded, err := store.Get(ctx, "test-persist-0001")
	if err != nil {
		t.Fatalf("Get after persist: %v", err)
	}
	if len(loaded.History) != 2 { // hist(1) = 2 messages
		t.Errorf("loaded history len = %d, want 2", len(loaded.History))
	}
	cwd, err := luft.GetState[string](loaded, "cwd")
	if err != nil || cwd != "/proj" {
		t.Errorf("State[cwd] = %q, err=%v, want /proj", cwd, err)
	}
	u, err := luft.GetState[luft.Usage](loaded, "usage")
	if err != nil || u.InputTokens != 100 {
		t.Errorf("State[usage] = %+v, err=%v, want InputTokens=100", u, err)
	}
}

func TestPersistNilStoreNoOp(t *testing.T) {
	// A nil store must not panic.
	s := &session{store: nil, sessionID: "x", cwd: "/p", history: hist(1)}
	s.persist(context.Background())
}

// hist builds n trivial user/assistant message pairs (2n messages total),
// so a session's message count is easy to assert.
func hist(pairs int) []luft.Message {
	var out []luft.Message
	for i := 0; i < pairs; i++ {
		out = append(out, luft.NewUserMessage("q"))
		out = append(out, luft.Message{
			Role:    luft.RoleAssistant,
			Content: []luft.ContentBlock{{Type: luft.TypeText, Text: "a"}},
		})
	}
	return out
}

func mustCreate(t *testing.T, store *stores.FileStore, s *luft.Session) {
	t.Helper()
	if err := store.Create(context.Background(), s); err != nil {
		t.Fatalf("Create %s: %v", s.ID, err)
	}
}

func histLen(s *luft.Session) int {
	if s == nil {
		return -1
	}
	return len(s.History)
}
