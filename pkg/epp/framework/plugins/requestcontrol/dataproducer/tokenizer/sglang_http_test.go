/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tokenizer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
	"github.com/stretchr/testify/require"
)

func TestSGLangTokenizePreservesChatTemplateInputs(t *testing.T) {
	var got fwkrh.PayloadMap
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/tokenize", r.URL.Path)
		require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
		_, _ = w.Write([]byte(`{"tokens":[10,20,30],"count":3}`))
	}))
	defer server.Close()
	tk, err := newSGLangHTTPTokenizer(&sglangConfig{URL: server.URL}, "served-model")
	require.NoError(t, err)
	original := fwkrh.PayloadMap{"model": "client-model", "messages": []any{map[string]any{"role": "user", "content": "hello"}}, "chat_template_kwargs": map[string]any{"enable_thinking": false}, "tools": []any{map[string]any{"type": "function"}}, "tool_choice": "auto", "reasoning_effort": "low", "continue_final_message": true}
	body := &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{}, Payload: original}
	result, err := tk.produce(context.Background(), body)
	require.NoError(t, err)
	require.Equal(t, [][]uint32{{10, 20, 30}}, result.PerPromptTokens)
	require.Equal(t, "served-model", got["model"])
	for _, key := range []string{"messages", "chat_template_kwargs", "tools", "tool_choice", "reasoning_effort", "continue_final_message"} {
		require.Equal(t, original[key], got[key])
	}
	require.Equal(t, "client-model", original["model"])
}

func TestSGLangTokenizeStreamingInferenceUsesNonstreamingRequest(t *testing.T) {
	for _, kind := range []string{"completion", "chat"} {
		t.Run(kind, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var got fwkrh.PayloadMap
				require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
				if got["stream"] == true {
					http.Error(w, "OpenAIServingTokenize does not support streaming requests", http.StatusNotImplemented)
					return
				}
				require.NotContains(t, got, "stream_options")
				require.Equal(t, map[string]any{"enable_thinking": false}, got["chat_template_kwargs"])
				_, _ = w.Write([]byte(`{"tokens":[10,20,30]}`))
			}))
			defer server.Close()
			tk, err := newSGLangHTTPTokenizer(&sglangConfig{URL: server.URL}, "m")
			require.NoError(t, err)
			original := fwkrh.PayloadMap{"model": "m", "stream": true, "stream_options": map[string]any{"include_usage": true}, "chat_template_kwargs": map[string]any{"enable_thinking": false}}
			body := &fwkrh.InferenceRequestBody{Payload: original}
			if kind == "completion" {
				original["prompt"] = "hello"
				body.Completions = &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}}
			} else {
				original["messages"] = []any{map[string]any{"role": "user", "content": "hello"}}
				body.ChatCompletions = &fwkrh.ChatCompletionsRequest{}
			}
			result, err := tk.produce(context.Background(), body)
			require.NoError(t, err)
			require.Equal(t, [][]uint32{{10, 20, 30}}, result.PerPromptTokens)
			require.Equal(t, true, original["stream"])
			require.Equal(t, map[string]any{"include_usage": true}, original["stream_options"])
		})
	}
}

func TestSGLangTokenizeBatchAndFailure(t *testing.T) {
	for _, test := range []struct {
		name, response string
		status         int
		want           [][]uint32
		wantErr        bool
	}{
		{"batch", `{"tokens":[[1,2],[3,4]]}`, 200, [][]uint32{{1, 2}, {3, 4}}, false},
		{"missing tokens", `{}`, 200, nil, true},
		{"invalid tokens", `{"tokens":[-1]}`, 200, nil, true},
		{"http error", `{"error":"bad template"}`, 400, nil, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(test.response))
			}))
			defer server.Close()
			tk, err := newSGLangHTTPTokenizer(&sglangConfig{URL: server.URL}, "m")
			require.NoError(t, err)
			result, err := tk.produce(context.Background(), &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Strings: []string{"a", "b"}}}})
			if test.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Equal(t, test.want, result.PerPromptTokens)
			}
		})
	}
}

func TestSGLangTokenizeRejectsMultimodal(t *testing.T) {
	tk, err := newSGLangHTTPTokenizer(&sglangConfig{URL: "http://unused"}, "m")
	require.NoError(t, err)
	_, err = tk.produce(context.Background(), &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: []fwkrh.Message{{Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{{Type: "image_url"}}}}}}})
	require.ErrorContains(t, err, "text content only")
}

func TestSGLangBackendSelection(t *testing.T) {
	handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
	for _, params := range []string{`{"sglang":{"url":"http://unused"}}`, `{"modelName":"m","sglang":{"url":"http://unused"},"vllm":{}}`, `{"modelName":"m","sglang":{"url":"http://unused"},"estimate":{}}`} {
		_, err := PluginFactory("tok", json.NewDecoder(strings.NewReader(params)), handle)
		require.Error(t, err)
	}
	p, err := PluginFactory("tok", json.NewDecoder(strings.NewReader(`{"modelName":"m","sglang":{"url":"http://unused","timeout":"2s"}}`)), handle)
	require.NoError(t, err)
	require.IsType(t, &sglangHTTPTokenizer{}, p.(*Plugin).backend)
	require.Equal(t, 2*time.Second+sglangFallbackGrace, p.(*Plugin).ProduceTimeout())
}

func TestSGLangTokenizationFailureAllowsLoadRouting(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
	defer server.Close()
	p, err := NewPlugin(context.Background(), "tok", &tokenizerPluginConfig{ModelName: "m", SGLang: &sglangConfig{URL: server.URL}})
	require.NoError(t, err)
	req := &fwksched.InferenceRequest{Body: &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}}}}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.Nil(t, req.Body.TokenizedPrompt)
	req.Body = &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: []fwkrh.Message{{Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{{Type: "image_url"}}}}}}}
	require.NoError(t, p.Produce(context.Background(), req, nil))
	require.Nil(t, req.Body.TokenizedPrompt)
}
