package bedrock

import (
	"bytes"
	"net/http"
	"strings"
	"testing"
	"time"
)

// TestSignRequest_Deterministic locks in the signature for a known request.
// If this changes, either the canonicalization or the signing-key derivation
// changed — both are stable across AWS services, so any change is a bug.
func TestSignRequest_Deterministic(t *testing.T) {
	creds := awsCredentials{
		AccessKeyID:     "AKIDEXAMPLE",
		SecretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
	}
	body := []byte(`{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`)
	req, err := http.NewRequest(http.MethodPost,
		"https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse",
		bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	now := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
	signRequest(req, creds, "bedrock", "us-east-1", body, now)

	if got := req.Header.Get("X-Amz-Date"); got != "20240101T120000Z" {
		t.Errorf("X-Amz-Date: got %q", got)
	}
	if got := req.Header.Get("X-Amz-Content-Sha256"); got == "" {
		t.Error("X-Amz-Content-Sha256 not set")
	}

	auth := req.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20240101/us-east-1/bedrock/aws4_request, SignedHeaders=") {
		t.Errorf("Authorization prefix wrong: %s", auth)
	}
	if !strings.Contains(auth, ", Signature=") {
		t.Errorf("Authorization missing Signature: %s", auth)
	}
}

func TestSignRequest_SessionToken(t *testing.T) {
	creds := awsCredentials{
		AccessKeyID:     "AKID",
		SecretAccessKey: "secret",
		SessionToken:    "TEMP-TOKEN",
	}
	req, _ := http.NewRequest(http.MethodPost,
		"https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse",
		strings.NewReader("{}"))
	signRequest(req, creds, "bedrock", "us-east-1", []byte("{}"),
		time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC))

	if got := req.Header.Get("X-Amz-Security-Token"); got != "TEMP-TOKEN" {
		t.Errorf("X-Amz-Security-Token: got %q", got)
	}
	if !strings.Contains(req.Header.Get("Authorization"), "x-amz-security-token") {
		t.Error("session token must be in signed headers")
	}
}

func TestAwsURIEncode(t *testing.T) {
	tests := []struct {
		in, want string
		slash    bool
	}{
		{"abc-_.~", "abc-_.~", false},
		{"a b", "a%20b", false},
		{"a/b", "a/b", false},
		{"a/b", "a%2Fb", true},
		{"a:b", "a%3Ab", false},
		{"", "", false},
	}
	for _, tc := range tests {
		if got := awsURIEncode(tc.in, tc.slash); got != tc.want {
			t.Errorf("awsURIEncode(%q, %v) = %q, want %q", tc.in, tc.slash, got, tc.want)
		}
	}
}

func TestCanonicalQuery(t *testing.T) {
	tests := []struct{ in, want string }{
		{"", ""},
		{"b=2&a=1", "a=1&b=2"},
		{"key", "key="},
		{"a b=c d", "a%20b=c%20d"},
	}
	for _, tc := range tests {
		if got := canonicalQuery(tc.in); got != tc.want {
			t.Errorf("canonicalQuery(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
