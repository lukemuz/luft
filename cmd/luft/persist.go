package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/lukemuz/luft"
	"github.com/lukemuz/luft/stores"
)

// resolveStoreDir returns ~/.config/luft/store, creating it if needed. It
// mirrors resolveLogPath's home-dir logic but keeps the JSON session store in
// its own subdir, separate from the JSONL event logs.
func resolveStoreDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".config", "luft", "store")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return dir, nil
}

// dirHash returns the first 12 hex chars of sha256(absCwd). It is stable and
// filesystem-safe, and serves as a session-ID prefix so the FileStore can list
// sessions for a given working directory via a scoped prefix query instead of
// loading every session's JSON.
func dirHash(absCwd string) string {
	sum := sha256.Sum256([]byte(absCwd))
	return hex.EncodeToString(sum[:])[:12]
}

// newSessionID builds a stable, sortable ID for a fresh REPL session:
//
//	<dirHash>-<timestamp>-<rand>
//
// The dirHash prefix scopes it to the working directory; the timestamp is
// zero-padded so lexicographic order matches chronological order; the random
// suffix breaks same-second collisions on a fast restart. All characters are
// in [0-9a-f-], which passes the FileStore's ID validator.
func newSessionID(absCwd string) string {
	return dirHash(absCwd) + "-" + time.Now().Format("20060102-150405") + "-" + randHex(4)
}

func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand should not fail in practice; fall back to a low-entropy
		// value so the ID is still valid rather than crashing startup.
		for i := range b {
			b[i] = byte(time.Now().UnixNano() >> uint(i*8))
		}
	}
	return hex.EncodeToString(b)
}

// resolveSession determines which session ID to use at startup and, on resume,
// loads the prior history. It never returns an error: every failure mode
// (missing store, not-found, corrupt JSON) degrades to a fresh session with a
// human-readable notice printed to stderr. It returns the ID, the loaded
// session (nil when fresh), and a notice line (empty when starting fresh with
// no flags). The store is the concrete *stores.FileStore so the -continue path
// can use ListIDs — which reads only directory entries, not full histories —
// to find the newest session for this directory.
func resolveSession(ctx context.Context, store *stores.FileStore, absCwd, resumeID string, continueLast bool) (string, *luft.Session, string) {
	if resumeID == "" && !continueLast {
		return newSessionID(absCwd), nil, ""
	}
	if store == nil {
		return newSessionID(absCwd), nil, grey("(session store unavailable — starting fresh)")
	}

	if resumeID != "" {
		sess, err := store.Get(ctx, resumeID)
		if err != nil {
			if errors.Is(err, luft.ErrSessionNotFound) {
				return newSessionID(absCwd), nil, fmt.Sprintf("%s session %s not found — %s",
					yellow("⚠"), bold(resumeID), grey("starting fresh"))
			}
			return newSessionID(absCwd), nil, fmt.Sprintf("%s could not read session %s: %v — %s",
				yellow("⚠"), bold(resumeID), err, grey("starting fresh"))
		}
		return resumeID, sess, fmt.Sprintf("%s session %s %s",
			green("↻ resumed"), bold(resumeID), grey(fmt.Sprintf("(%d messages restored)", len(sess.History))))
	}

	// -continue: find the most recent session for this working directory.
	// ListIDs scopes by the dirHash prefix and returns IDs sorted ascending;
	// the last ID is the newest. It reads only directory entries, not full
	// session JSON, so resuming doesn't load every prior history.
	ids, err := store.ListIDs(ctx, dirHash(absCwd)+"-", 0)
	if err != nil || len(ids) == 0 {
		return newSessionID(absCwd), nil, grey("(no prior session for this directory — starting fresh)")
	}
	latest := ids[len(ids)-1]
	sess, err := store.Get(ctx, latest)
	if err != nil {
		// Corrupt or deleted between list and get: fresh ID so the first save
		// doesn't overwrite the bad file.
		return newSessionID(absCwd), nil, fmt.Sprintf("%s could not read session %s: %v — %s",
			yellow("⚠"), bold(latest), err, grey("starting fresh"))
	}
	return latest, sess, fmt.Sprintf("%s session %s %s",
		green("↻ resumed"), bold(latest), grey(fmt.Sprintf("(%d messages restored)", len(sess.History))))
}
