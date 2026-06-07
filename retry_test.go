package luft

import (
	"net/http"
	"testing"
	"time"
)

func TestParseRetryAfter(t *testing.T) {
	future := time.Now().Add(120 * time.Second).UTC().Format(http.TimeFormat)
	past := time.Now().Add(-120 * time.Second).UTC().Format(http.TimeFormat)

	tests := []struct {
		name  string
		value string
		set   bool
		check func(time.Duration) bool
	}{
		{name: "absent", set: false, check: func(d time.Duration) bool { return d == 0 }},
		{name: "delta seconds", value: "30", set: true, check: func(d time.Duration) bool { return d == 30*time.Second }},
		{name: "zero seconds", value: "0", set: true, check: func(d time.Duration) bool { return d == 0 }},
		{name: "negative seconds", value: "-5", set: true, check: func(d time.Duration) bool { return d == 0 }},
		{name: "garbage", value: "soon", set: true, check: func(d time.Duration) bool { return d == 0 }},
		{name: "http-date future", value: future, set: true, check: func(d time.Duration) bool { return d > 110*time.Second && d <= 121*time.Second }},
		{name: "http-date past", value: past, set: true, check: func(d time.Duration) bool { return d == 0 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := http.Header{}
			if tt.set {
				h.Set("Retry-After", tt.value)
			}
			if got := ParseRetryAfter(h); !tt.check(got) {
				t.Errorf("ParseRetryAfter(%q) = %v", tt.value, got)
			}
		})
	}
}

func TestIsRetryableStatus(t *testing.T) {
	for _, c := range []int{429, 500, 502, 503, 504, 529} {
		if !isRetryableStatus(c) {
			t.Errorf("status %d should be retryable", c)
		}
	}
	for _, c := range []int{200, 400, 401, 403, 404, 409, 422, 501} {
		if isRetryableStatus(c) {
			t.Errorf("status %d should not be retryable", c)
		}
	}
}
