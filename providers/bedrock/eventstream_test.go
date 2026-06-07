package bedrock

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

func TestEventStream_RoundTrip(t *testing.T) {
	frame := encodeEventStreamMessage(
		map[string]string{":message-type": "event", ":event-type": "contentBlockDelta"},
		[]byte(`{"delta":{"text":"hello"}}`),
	)
	r := newEventStreamReader(bytes.NewReader(frame))
	msg, err := r.Read()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if msg.Headers[":event-type"] != "contentBlockDelta" {
		t.Errorf("event-type: %q", msg.Headers[":event-type"])
	}
	if string(msg.Payload) != `{"delta":{"text":"hello"}}` {
		t.Errorf("payload: %s", msg.Payload)
	}
}

func TestEventStream_MultipleFrames(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(encodeEventStreamMessage(
		map[string]string{":event-type": "messageStart"},
		[]byte(`{"role":"assistant"}`),
	))
	buf.Write(encodeEventStreamMessage(
		map[string]string{":event-type": "messageStop"},
		[]byte(`{"stopReason":"end_turn"}`),
	))

	r := newEventStreamReader(&buf)
	got := []string{}
	for {
		msg, err := r.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		got = append(got, msg.Headers[":event-type"])
	}
	if len(got) != 2 || got[0] != "messageStart" || got[1] != "messageStop" {
		t.Errorf("events: %v", got)
	}
}

func TestEventStream_PreludeCRCMismatch(t *testing.T) {
	frame := encodeEventStreamMessage(
		map[string]string{":event-type": "x"},
		[]byte("y"),
	)
	frame[8] ^= 0xff // corrupt prelude CRC

	r := newEventStreamReader(bytes.NewReader(frame))
	if _, err := r.Read(); err == nil {
		t.Fatal("expected prelude CRC error")
	}
}
