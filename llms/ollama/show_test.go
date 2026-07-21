package ollama

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/tmc/langchaingo/llms"
)

// TestNewProbesShow verifies that New() issues an /api/show request to the
// configured server and that the request body matches the ShowRequest schema.
func TestNewProbesShow(t *testing.T) {
	t.Parallel()

	var (
		showCalls atomic.Int32
		lastBody  atomic.Value // string
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		lastBody.Store(string(body))

		switch r.URL.Path {
		case "/api/show":
			showCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"capabilities":["thinking","completion"]}`))
		case "/api/chat":
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte(`{"model":"gemma3:1b","created_at":"2025-01-01T00:00:00Z","message":{"role":"assistant","content":"4"},"done":true}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	llm, err := New(
		WithServerURL(srv.URL),
		WithModel("gemma3:1b"),
	)
	require.NoError(t, err)
	require.NotNil(t, llm)

	assert.Equal(t, int32(1), showCalls.Load(), "New should call /api/show exactly once")
	assert.True(t, llm.SupportsReasoning(), "model advertising the thinking capability should be detected as reasoning-capable")

	// Verify request body matches ShowRequest schema: {"model":"..."}
	gotBody, _ := lastBody.Load().(string)
	var sent struct {
		Model string `json:"model"`
	}
	require.NoError(t, json.Unmarshal([]byte(gotBody), &sent))
	assert.Equal(t, "gemma3:1b", sent.Model, "/api/show body should carry the configured model name")
}

// TestSupportsReasoningFromCapabilities verifies that SupportsReasoning() reflects
// the /api/show response. We run two cases: with and without the "thinking"
// capability, and assert that think is sent (or not) accordingly.
func TestSupportsReasoningFromCapabilities(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name            string
		capabilities    string
		wantSupports    bool
		wantThinkInBody bool
	}{
		{
			name:            "thinking capability present",
			capabilities:    `["thinking"]`,
			wantSupports:    true,
			wantThinkInBody: true,
		},
		{
			name:            "no thinking capability",
			capabilities:    `["completion"]`,
			wantSupports:    false,
			wantThinkInBody: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				chatBodies atomic.Value
				showCalls  atomic.Int32
			)

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					showCalls.Add(1)
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"capabilities":` + tc.capabilities + `}`))
				case "/api/chat":
					body, _ := io.ReadAll(r.Body)
					chatBodies.Store(string(body))
					w.Header().Set("Content-Type", "application/x-ndjson")
					_, _ = w.Write([]byte(`{"model":"gemma3:1b","created_at":"2025-01-01T00:00:00Z","message":{"role":"assistant","content":"ok"},"done":true}`))
				default:
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()

			llm, err := New(
				WithServerURL(srv.URL),
				WithModel("gemma3:1b"),
				WithThink(true),
			)
			require.NoError(t, err)
			assert.Equal(t, tc.wantSupports, llm.SupportsReasoning())
			assert.Equal(t, int32(1), showCalls.Load())

			resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
				{
					Role:  llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{llms.TextContent{Text: "ping"}},
				},
			})
			require.NoError(t, err)
			require.NotEmpty(t, resp.Choices)

			body, _ := chatBodies.Load().(string)
			hasThink := strings.Contains(body, `"think"`)
			assert.Equal(t, tc.wantThinkInBody, hasThink,
				"think presence in chat body: got body=%s", body)
		})
	}
}

// TestWithThinkLevelSendsStringValue verifies WithThinkLevel produces a
// non-bool string value in the request body when the model supports it.
func TestWithThinkLevelSendsStringValue(t *testing.T) {
	t.Parallel()

	var chatBody atomic.Value

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"capabilities":["thinking"]}`))
		case "/api/chat":
			body, _ := io.ReadAll(r.Body)
			chatBody.Store(string(body))
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte(`{"model":"gemma3:1b","created_at":"2025-01-01T00:00:00Z","message":{"role":"assistant","content":"ok"},"done":true}`))
		}
	}))
	defer srv.Close()

	llm, err := New(
		WithServerURL(srv.URL),
		WithModel("gemma3:1b"),
		WithThinkLevel("high"),
	)
	require.NoError(t, err)
	require.True(t, llm.SupportsReasoning())

	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextContent{Text: "ping"}},
		},
	})
	require.NoError(t, err)

	body, _ := chatBody.Load().(string)
	assert.Contains(t, body, `"think":"high"`, "WithThinkLevel should serialize the level string verbatim")
}

// TestNoThinkOmitsField verifies that when WithThink is not used and the
// model supports thinking, the request still omits "think" entirely so the
// server treats the call as "unspecified".
func TestNoThinkOmitsField(t *testing.T) {
	t.Parallel()

	var chatBody atomic.Value

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"capabilities":["thinking"]}`))
		case "/api/chat":
			body, _ := io.ReadAll(r.Body)
			chatBody.Store(string(body))
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte(`{"model":"gemma3:1b","created_at":"2025-01-01T00:00:00Z","message":{"role":"assistant","content":"ok"},"done":true}`))
		}
	}))
	defer srv.Close()

	llm, err := New(
		WithServerURL(srv.URL),
		WithModel("gemma3:1b"),
	)
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextContent{Text: "ping"}},
		},
	})
	require.NoError(t, err)

	body, _ := chatBody.Load().(string)
	assert.NotContains(t, body, `"think"`, "WithThink not used: think field should be omitted from the request")
}

// TestSampleRequestShapes prints concrete /api/chat request bodies for each
// combination of capability + think option so reviewers can eyeball the JSON.
func TestSampleRequestShapes(t *testing.T) {
	t.Parallel()

	type tc struct {
		name       string
		capability string
		thinkOpt   Option
	}
	cases := []tc{
		{"no_think_unspecified", "[\"thinking\"]", nil},
		{"no_think_disabled", "[\"thinking\"]", WithThink(false)},
		{"think_enabled", "[\"thinking\"]", WithThink(true)},
		{"think_level_high", "[\"thinking\"]", WithThinkLevel("high")},
		{"think_level_max", "[\"thinking\"]", WithThinkLevel("max")},
		{"unsupported_think_enabled_omitted", "[\"completion\"]", WithThink(true)},
		{"unsupported_think_level_omitted", "[\"completion\"]", WithThinkLevel("low")},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var (
				chatBody atomic.Value
				showBody atomic.Value
			)

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					b, _ := io.ReadAll(r.Body)
					showBody.Store(string(b))
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"capabilities":` + c.capability + `}`))
				case "/api/chat":
					b, _ := io.ReadAll(r.Body)
					chatBody.Store(string(b))
					w.Header().Set("Content-Type", "application/x-ndjson")
					_, _ = w.Write([]byte(`{"model":"demo","created_at":"2025-01-01T00:00:00Z","message":{"role":"assistant","content":"ok"},"done":true}`))
				}
			}))
			defer srv.Close()

			opts := []Option{WithServerURL(srv.URL), WithModel("demo")}
			if c.thinkOpt != nil {
				opts = append(opts, c.thinkOpt)
			}
			llm, err := New(opts...)
			require.NoError(t, err)

			_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
				{
					Role:  llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{llms.TextContent{Text: "hello"}},
				},
			})
			require.NoError(t, err)

			// Pretty-print the captured JSON for readability.
			raw, _ := chatBody.Load().(string)
			var parsed map[string]any
			require.NoError(t, json.Unmarshal([]byte(raw), &parsed))
			out, _ := json.MarshalIndent(parsed, "", "  ")

			t.Logf("capabilities=%s supportsReasoning=%v\n--- /api/show request body ---\n%s\n--- /api/chat request body ---\n%s",
				c.capability, llm.SupportsReasoning(),
				showBody.Load().(string), string(out))
		})
	}
}
