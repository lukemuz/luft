package bedrock

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

// AWS Event Stream (vnd.amazon.eventstream) binary frame decoder.
// Spec: https://docs.aws.amazon.com/transcribe/latest/dg/event-stream.html
//
// Frame layout:
//   [4]  total length (big-endian uint32, includes everything from this byte)
//   [4]  headers length (big-endian uint32)
//   [4]  prelude CRC (CRC32 of first 8 bytes)
//   [N]  headers
//   [P]  payload (total - 16 - N)
//   [4]  message CRC (CRC32 of everything except itself)
//
// Headers are key-typed-value triples:
//   [1]  name length (uint8)
//   [N]  name (UTF-8, no null terminator)
//   [1]  value type
//   [V]  value (length depends on type)

const (
	eventHeaderTypeBoolTrue  = 0
	eventHeaderTypeBoolFalse = 1
	eventHeaderTypeInt8      = 2
	eventHeaderTypeInt16     = 3
	eventHeaderTypeInt32     = 4
	eventHeaderTypeInt64     = 5
	eventHeaderTypeBytes     = 6
	eventHeaderTypeString    = 7
	eventHeaderTypeTimestamp = 8
	eventHeaderTypeUUID      = 9
)

// eventStreamMessage is one decoded frame.
type eventStreamMessage struct {
	Headers map[string]string // string-typed headers, captured as strings
	Payload []byte
}

// eventStreamReader reads framed messages from an underlying byte stream.
type eventStreamReader struct {
	r io.Reader
}

func newEventStreamReader(r io.Reader) *eventStreamReader {
	return &eventStreamReader{r: r}
}

// Read returns the next message or io.EOF on a clean end of stream.
func (e *eventStreamReader) Read() (eventStreamMessage, error) {
	var prelude [12]byte
	if _, err := io.ReadFull(e.r, prelude[:]); err != nil {
		return eventStreamMessage{}, err
	}
	totalLen := binary.BigEndian.Uint32(prelude[0:4])
	headersLen := binary.BigEndian.Uint32(prelude[4:8])
	preludeCRC := binary.BigEndian.Uint32(prelude[8:12])

	if got := crc32.ChecksumIEEE(prelude[:8]); got != preludeCRC {
		return eventStreamMessage{}, fmt.Errorf("eventstream: prelude crc mismatch: got %#x want %#x", got, preludeCRC)
	}
	if totalLen < 16 {
		return eventStreamMessage{}, fmt.Errorf("eventstream: total length too small: %d", totalLen)
	}
	if uint64(headersLen)+16 > uint64(totalLen) {
		return eventStreamMessage{}, fmt.Errorf("eventstream: headers length %d exceeds body in frame of %d", headersLen, totalLen)
	}

	rest := make([]byte, totalLen-12)
	if _, err := io.ReadFull(e.r, rest); err != nil {
		return eventStreamMessage{}, err
	}

	headersBytes := rest[:headersLen]
	payload := rest[headersLen : len(rest)-4]

	headers, err := parseEventStreamHeaders(headersBytes)
	if err != nil {
		return eventStreamMessage{}, err
	}
	return eventStreamMessage{Headers: headers, Payload: payload}, nil
}

func parseEventStreamHeaders(b []byte) (map[string]string, error) {
	out := make(map[string]string)
	i := 0
	for i < len(b) {
		if i+1 > len(b) {
			return nil, errors.New("eventstream: header truncated at name length")
		}
		nameLen := int(b[i])
		i++
		if i+nameLen+1 > len(b) {
			return nil, errors.New("eventstream: header name truncated")
		}
		name := string(b[i : i+nameLen])
		i += nameLen
		valueType := b[i]
		i++
		switch valueType {
		case eventHeaderTypeString, eventHeaderTypeBytes:
			if i+2 > len(b) {
				return nil, errors.New("eventstream: header value length truncated")
			}
			vl := int(binary.BigEndian.Uint16(b[i : i+2]))
			i += 2
			if i+vl > len(b) {
				return nil, errors.New("eventstream: header value bytes truncated")
			}
			out[name] = string(b[i : i+vl])
			i += vl
		case eventHeaderTypeBoolTrue:
			out[name] = "true"
		case eventHeaderTypeBoolFalse:
			out[name] = "false"
		case eventHeaderTypeInt8:
			if i+1 > len(b) {
				return nil, errors.New("eventstream: header int8 truncated")
			}
			out[name] = fmt.Sprintf("%d", int8(b[i]))
			i++
		case eventHeaderTypeInt16:
			if i+2 > len(b) {
				return nil, errors.New("eventstream: header int16 truncated")
			}
			out[name] = fmt.Sprintf("%d", int16(binary.BigEndian.Uint16(b[i:i+2])))
			i += 2
		case eventHeaderTypeInt32:
			if i+4 > len(b) {
				return nil, errors.New("eventstream: header int32 truncated")
			}
			out[name] = fmt.Sprintf("%d", int32(binary.BigEndian.Uint32(b[i:i+4])))
			i += 4
		case eventHeaderTypeInt64, eventHeaderTypeTimestamp:
			if i+8 > len(b) {
				return nil, errors.New("eventstream: header int64 truncated")
			}
			out[name] = fmt.Sprintf("%d", int64(binary.BigEndian.Uint64(b[i:i+8])))
			i += 8
		case eventHeaderTypeUUID:
			if i+16 > len(b) {
				return nil, errors.New("eventstream: header uuid truncated")
			}
			out[name] = fmt.Sprintf("%x", b[i:i+16])
			i += 16
		default:
			return nil, fmt.Errorf("eventstream: unknown header value type %d", valueType)
		}
	}
	return out, nil
}

// encodeEventStreamMessage builds a single framed message; used only by tests
// to feed deterministic frames into the reader.
func encodeEventStreamMessage(headers map[string]string, payload []byte) []byte {
	var hb []byte
	for k, v := range headers {
		hb = append(hb, byte(len(k)))
		hb = append(hb, k...)
		hb = append(hb, eventHeaderTypeString)
		hb = binary.BigEndian.AppendUint16(hb, uint16(len(v)))
		hb = append(hb, v...)
	}
	totalLen := uint32(12 + len(hb) + len(payload) + 4)
	out := make([]byte, 0, totalLen)
	out = binary.BigEndian.AppendUint32(out, totalLen)
	out = binary.BigEndian.AppendUint32(out, uint32(len(hb)))
	preludeCRC := crc32.ChecksumIEEE(out[:8])
	out = binary.BigEndian.AppendUint32(out, preludeCRC)
	out = append(out, hb...)
	out = append(out, payload...)
	msgCRC := crc32.ChecksumIEEE(out)
	out = binary.BigEndian.AppendUint32(out, msgCRC)
	return out
}
