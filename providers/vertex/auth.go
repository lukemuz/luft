package vertex

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// Google service-account JWT → OAuth2 access token, written from scratch so
// the vertex package preserves luft's zero-dependency core.
// Flow per https://developers.google.com/identity/protocols/oauth2/service-account#httprest

const (
	tokenURL    = "https://oauth2.googleapis.com/token"
	jwtScope    = "https://www.googleapis.com/auth/cloud-platform"
	jwtAudience = tokenURL
	jwtGrantTyp = "urn:ietf:params:oauth:grant-type:jwt-bearer"
)

// ServiceAccount is the subset of a Google service-account JSON we need.
// Load it from disk with LoadServiceAccountFile, or build it manually.
type ServiceAccount struct {
	Type         string `json:"type"`
	ProjectID    string `json:"project_id"`
	PrivateKeyID string `json:"private_key_id"`
	PrivateKey   string `json:"private_key"` // PEM-encoded PKCS8 or PKCS1
	ClientEmail  string `json:"client_email"`
	TokenURI     string `json:"token_uri,omitempty"` // optional override
}

// LoadServiceAccountFile reads and parses a service-account JSON file.
func LoadServiceAccountFile(path string) (ServiceAccount, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ServiceAccount{}, fmt.Errorf("luft: vertex: read service account: %w", err)
	}
	var sa ServiceAccount
	if err := json.Unmarshal(data, &sa); err != nil {
		return ServiceAccount{}, fmt.Errorf("luft: vertex: parse service account: %w", err)
	}
	if sa.Type != "" && sa.Type != "service_account" {
		return ServiceAccount{}, fmt.Errorf("luft: vertex: unsupported credentials type %q", sa.Type)
	}
	if sa.ClientEmail == "" || sa.PrivateKey == "" {
		return ServiceAccount{}, errors.New("luft: vertex: service account missing client_email or private_key")
	}
	return sa, nil
}

// TokenSource yields valid bearer access tokens, refreshing as needed.
// The zero value is not usable; build via newServiceAccountTokenSource or
// supply a static token via newStaticTokenSource.
type TokenSource interface {
	Token(ctx context.Context) (string, error)
}

// newServiceAccountTokenSource returns a TokenSource backed by a Google
// service account. now is normally time.Now; tests override for determinism.
// httpClient is used for the token exchange; nil uses http.DefaultClient.
func newServiceAccountTokenSource(sa ServiceAccount, httpClient *http.Client, now func() time.Time) (*serviceAccountTokenSource, error) {
	key, err := parsePrivateKey(sa.PrivateKey)
	if err != nil {
		return nil, err
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	if now == nil {
		now = time.Now
	}
	tokURL := sa.TokenURI
	if tokURL == "" {
		tokURL = tokenURL
	}
	return &serviceAccountTokenSource{
		sa:         sa,
		key:        key,
		httpClient: httpClient,
		now:        now,
		tokenURL:   tokURL,
	}, nil
}

type serviceAccountTokenSource struct {
	sa         ServiceAccount
	key        *rsa.PrivateKey
	httpClient *http.Client
	now        func() time.Time
	tokenURL   string

	mu      sync.Mutex
	token   string
	expires time.Time
}

func (s *serviceAccountTokenSource) Token(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.token != "" && s.now().Before(s.expires.Add(-60*time.Second)) {
		return s.token, nil
	}
	tok, expIn, err := s.exchange(ctx)
	if err != nil {
		return "", err
	}
	s.token = tok
	s.expires = s.now().Add(time.Duration(expIn) * time.Second)
	return s.token, nil
}

func (s *serviceAccountTokenSource) exchange(ctx context.Context) (string, int, error) {
	assertion, err := s.signJWT()
	if err != nil {
		return "", 0, err
	}
	form := url.Values{}
	form.Set("grant_type", jwtGrantTyp)
	form.Set("assertion", assertion)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("luft: vertex: token exchange http: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("luft: vertex: token exchange failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var parsed struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
		TokenType   string `json:"token_type"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", 0, fmt.Errorf("luft: vertex: parse token response: %w", err)
	}
	if parsed.AccessToken == "" {
		return "", 0, errors.New("luft: vertex: token exchange returned empty access_token")
	}
	return parsed.AccessToken, parsed.ExpiresIn, nil
}

func (s *serviceAccountTokenSource) signJWT() (string, error) {
	now := s.now().Unix()
	headerJSON, _ := json.Marshal(map[string]string{"alg": "RS256", "typ": "JWT"})
	claimsJSON, _ := json.Marshal(map[string]any{
		"iss":   s.sa.ClientEmail,
		"scope": jwtScope,
		"aud":   jwtAudience,
		"iat":   now,
		"exp":   now + 3600,
	})

	hdr := base64URLEncode(headerJSON)
	body := base64URLEncode(claimsJSON)
	signingInput := hdr + "." + body

	sum := sha256.Sum256([]byte(signingInput))
	sig, err := rsa.SignPKCS1v15(rand.Reader, s.key, crypto.SHA256, sum[:])
	if err != nil {
		return "", fmt.Errorf("luft: vertex: sign jwt: %w", err)
	}
	return signingInput + "." + base64URLEncode(sig), nil
}

// staticTokenSource returns a fixed token. Useful in tests, or when the
// caller manages refresh externally (gcloud auth print-access-token, workload
// identity, ADC via an external library).
type staticTokenSource struct{ tok string }

func (s staticTokenSource) Token(_ context.Context) (string, error) { return s.tok, nil }

func parsePrivateKey(pemStr string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(pemStr))
	if block == nil {
		return nil, errors.New("luft: vertex: private key not PEM-encoded")
	}
	if k, err := x509.ParsePKCS1PrivateKey(block.Bytes); err == nil {
		return k, nil
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("luft: vertex: parse private key: %w", err)
	}
	rsaKey, ok := parsed.(*rsa.PrivateKey)
	if !ok {
		return nil, fmt.Errorf("luft: vertex: private key is %T, want *rsa.PrivateKey", parsed)
	}
	return rsaKey, nil
}

func base64URLEncode(b []byte) string {
	return base64.RawURLEncoding.EncodeToString(b)
}
