package bedrock

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// AWS Signature Version 4 signing, implemented against
// https://docs.aws.amazon.com/general/latest/gr/sigv4_signing.html
// Hand-written so the bedrock package preserves luft's zero-dependency core.

const (
	signAlgorithm = "AWS4-HMAC-SHA256"
	awsRequestTag = "aws4_request"
)

// awsCredentials carries the IAM identity used to sign a request.
// SessionToken is non-empty for temporary credentials (STS, IAM role, SSO).
type awsCredentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

// signRequest applies SigV4 to req in place. service is the AWS service name
// (e.g. "bedrock"); region is the AWS region. payload must be the exact bytes
// of the request body (the same bytes that will be sent). now is the signing
// timestamp; pass time.Now().UTC() in production.
//
// On return, req carries the Host, X-Amz-Date, X-Amz-Content-Sha256,
// X-Amz-Security-Token (if applicable), and Authorization headers.
func signRequest(req *http.Request, creds awsCredentials, service, region string, payload []byte, now time.Time) {
	amzDate := now.Format("20060102T150405Z")
	dateStamp := now.Format("20060102")
	credentialScope := dateStamp + "/" + region + "/" + service + "/" + awsRequestTag

	host := req.Host
	if host == "" {
		host = req.URL.Host
	}
	req.Header.Set("Host", host)
	req.Header.Set("X-Amz-Date", amzDate)
	if creds.SessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", creds.SessionToken)
	}

	payloadHash := sha256Hex(payload)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)

	canonHeaders, signedHeaders := canonicalHeaders(req.Header, host)
	// SigV4 for non-S3 services requires double URI-encoding: the wire path
	// is single-encoded, and the canonical URI re-encodes that (so a `%`
	// in the wire form becomes `%25`). EscapedPath() gives the wire form;
	// canonicalURI encodes again, producing the double-encoded canonical URI
	// that AWS expects.
	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI(req.URL.EscapedPath()),
		canonicalQuery(req.URL.RawQuery),
		canonHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	stringToSign := strings.Join([]string{
		signAlgorithm,
		amzDate,
		credentialScope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")

	signingKey := deriveSigningKey(creds.SecretAccessKey, dateStamp, region, service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, stringToSign))

	auth := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		signAlgorithm, creds.AccessKeyID, credentialScope, signedHeaders, signature)
	req.Header.Set("Authorization", auth)
}

func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}

func deriveSigningKey(secret, dateStamp, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secret), dateStamp)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	return hmacSHA256(kService, awsRequestTag)
}

func canonicalURI(path string) string {
	if path == "" {
		return "/"
	}
	parts := strings.Split(path, "/")
	for i, p := range parts {
		parts[i] = awsURIEncode(p, false)
	}
	return strings.Join(parts, "/")
}

func canonicalQuery(raw string) string {
	if raw == "" {
		return ""
	}
	pairs := strings.Split(raw, "&")
	encoded := make([]string, 0, len(pairs))
	for _, p := range pairs {
		eq := strings.IndexByte(p, '=')
		var k, v string
		if eq >= 0 {
			k, v = p[:eq], p[eq+1:]
		} else {
			k = p
		}
		encoded = append(encoded, awsURIEncode(k, true)+"="+awsURIEncode(v, true))
	}
	sort.Strings(encoded)
	return strings.Join(encoded, "&")
}

// canonicalHeaders builds the canonical headers block and the signed-headers
// list. Every header except Authorization participates in the signature; this
// is the broadest-signing variant of SigV4 and the easiest to keep correct.
func canonicalHeaders(h http.Header, host string) (string, string) {
	type kv struct{ k, v string }
	pairs := make([]kv, 0, len(h)+1)
	seenHost := false
	for k, vs := range h {
		lk := strings.ToLower(k)
		if lk == "authorization" {
			continue
		}
		if lk == "host" {
			seenHost = true
		}
		pairs = append(pairs, kv{lk, strings.Join(vs, ",")})
	}
	if !seenHost && host != "" {
		pairs = append(pairs, kv{"host", host})
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].k < pairs[j].k })

	var cb, sb strings.Builder
	for i, p := range pairs {
		cb.WriteString(p.k)
		cb.WriteByte(':')
		cb.WriteString(collapseSpaces(strings.TrimSpace(p.v)))
		cb.WriteByte('\n')
		if i > 0 {
			sb.WriteByte(';')
		}
		sb.WriteString(p.k)
	}
	return cb.String(), sb.String()
}

func collapseSpaces(s string) string {
	if !strings.Contains(s, "  ") {
		return s
	}
	var b strings.Builder
	prevSpace := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == ' ' {
			if prevSpace {
				continue
			}
			prevSpace = true
		} else {
			prevSpace = false
		}
		b.WriteByte(c)
	}
	return b.String()
}

// awsURIEncode percent-encodes per the SigV4 rules:
// unreserved characters (A-Z a-z 0-9 - _ . ~) pass through; in paths '/' is
// also preserved; everything else is %XX-encoded.
func awsURIEncode(s string, encodeSlash bool) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case 'A' <= c && c <= 'Z',
			'a' <= c && c <= 'z',
			'0' <= c && c <= '9',
			c == '-', c == '_', c == '.', c == '~':
			b.WriteByte(c)
		case c == '/' && !encodeSlash:
			b.WriteByte(c)
		default:
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}
