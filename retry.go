package luft

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"net"
	"net/url"
	"time"
)

// RetryConfig controls automatic retry behaviour for transient API errors.
// The zero value enables retrying with sensible defaults.
type RetryConfig struct {
	MaxRetries  int           // max attempts after the first; 0 → use default (3)
	InitialWait time.Duration // first back-off interval; 0 → use default (1s)
	MaxWait     time.Duration // back-off ceiling; 0 → use default (30s)
	Disabled    bool          // set true to disable all retrying

	// OnRetry, when non-nil, is called before each retry sleep with the
	// 1-based retry attempt number and the computed backoff duration.
	// Use this to log retries or reset streaming state (see StreamBuffer).
	OnRetry func(attempt int, wait time.Duration)
}

// defaults applied when the corresponding RetryConfig field is zero.
const (
	defaultMaxRetries  = 3
	defaultInitialWait = 1 * time.Second
	defaultMaxWait     = 30 * time.Second
)

// applyDefaults fills in zero-value fields with package defaults.
func (r RetryConfig) applyDefaults() RetryConfig {
	if r.MaxRetries == 0 {
		r.MaxRetries = defaultMaxRetries
	}
	if r.InitialWait == 0 {
		r.InitialWait = defaultInitialWait
	}
	if r.MaxWait == 0 {
		r.MaxWait = defaultMaxWait
	}
	return r
}

// callWithRetry calls fn, retrying on retryable errors with exponential
// back-off + jitter. It respects a Retry-After header value embedded in
// APIError when present.
//
// Retryable conditions: *APIError with a transient StatusCode (429, 529, or
// 5xx — see isRetryableStatus), any error satisfying isTemporary, and
// network-level *url.Error / *net.OpError values.
//
// Non-retryable conditions return immediately: *APIError with StatusCode 400,
// 401, 403, or 404; ErrMissingTool; and context cancellation / deadline.
//
// When all retries are exhausted the last error is wrapped in a
// RetryExhaustedError.
func callWithRetry(ctx context.Context, cfg RetryConfig, fn func() (ProviderResponse, error)) (ProviderResponse, error) {
	// A single call with no retry logic when Disabled is set.
	if cfg.Disabled {
		return fn()
	}

	cfg = cfg.applyDefaults()

	var lastErr error
	for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
		// Honour context cancellation before each attempt.
		if err := ctx.Err(); err != nil {
			if lastErr != nil {
				// We already tried at least once; surface the context error
				// directly so callers see a cancellation rather than a retry
				// exhaustion error.
				return ProviderResponse{}, err
			}
			return ProviderResponse{}, err
		}

		resp, err := fn()
		if err == nil {
			return resp, nil
		}
		lastErr = err

		// Never retry context cancellation or deadline exceeded.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return ProviderResponse{}, err
		}

		// Never retry ErrMissingTool — it is a programming error, not transient.
		if errors.Is(err, ErrMissingTool) {
			return ProviderResponse{}, err
		}

		// Check for a hard non-retryable API status.
		var apiErr *APIError
		if errors.As(err, &apiErr) {
			switch apiErr.StatusCode {
			case 400, 401, 403, 404:
				return ProviderResponse{}, err
			}
		}

		// If this was the last allowed attempt, stop here — do not sleep.
		if attempt == cfg.MaxRetries {
			break
		}

		// Determine whether the error is worth retrying at all.
		if !isRetryable(err) {
			return ProviderResponse{}, err
		}

		// Compute how long to wait before the next attempt.
		wait := backoffWait(cfg, attempt, apiErr)

		if cfg.OnRetry != nil {
			cfg.OnRetry(attempt+1, wait)
		}

		select {
		case <-ctx.Done():
			return ProviderResponse{}, ctx.Err()
		case <-time.After(wait):
		}
	}

	return ProviderResponse{}, &RetryExhaustedError{
		Attempts: cfg.MaxRetries + 1,
		Cause:    lastErr,
	}
}

// isRetryable reports whether err should trigger a retry attempt.
func isRetryable(err error) bool {
	// Retryable API status codes.
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return isRetryableStatus(apiErr.StatusCode)
	}

	// Network-level errors.
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}

	// Errors implementing the Temporary() bool interface.
	return isTemporary(err)
}

// isRetryableStatus reports whether an HTTP status code is worth retrying.
// 429 (rate limited) and 529 (Anthropic "overloaded") are explicit transient
// signals; 500/502/503/504 are server and gateway errors that are typically
// transient too. The hard 4xx authz/validation codes (400/401/403/404) are
// rejected earlier in callWithRetry and never reach here.
func isRetryableStatus(code int) bool {
	switch code {
	case 429, 500, 502, 503, 504, 529:
		return true
	default:
		return false
	}
}

// backoffWait computes the sleep duration for the given attempt index.
// If apiErr is non-nil and carries a RetryAfter hint, that value is used
// directly (still capped to MaxWait). Otherwise an exponential back-off with
// up to 20 % random jitter is applied.
func backoffWait(cfg RetryConfig, attempt int, apiErr *APIError) time.Duration {
	// Honour an explicit Retry-After from a 429 response.
	if apiErr != nil && apiErr.RetryAfter > 0 {
		if apiErr.RetryAfter > cfg.MaxWait {
			return cfg.MaxWait
		}
		return apiErr.RetryAfter
	}

	// Exponential back-off: InitialWait * 2^attempt, capped at MaxWait.
	exp := math.Pow(2, float64(attempt))
	wait := time.Duration(float64(cfg.InitialWait) * exp)
	if wait > cfg.MaxWait {
		wait = cfg.MaxWait
	}

	// Add up to 20 % jitter to spread out concurrent retries.
	jitter := time.Duration(rand.Float64() * 0.2 * float64(wait))
	return wait + jitter
}

// isTemporary reports whether err (or any error in its chain) implements the
// Temporary() bool interface and returns true from that method.
func isTemporary(err error) bool {
	type temporary interface {
		Temporary() bool
	}
	var t temporary
	if errors.As(err, &t) {
		return t.Temporary()
	}
	return false
}
