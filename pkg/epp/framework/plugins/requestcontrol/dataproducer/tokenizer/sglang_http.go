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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net/http"
	"strings"
	"time"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

type sglangConfig struct {
	URL     string `json:"url"`
	Timeout string `json:"timeout,omitempty"`
}

const sglangFallbackGrace = 100 * time.Millisecond

type sglangHTTPTokenizer struct {
	client  *http.Client
	baseURL string
	model   string
	timeout time.Duration
}

func newSGLangHTTPTokenizer(cfg *sglangConfig, model string) (*sglangHTTPTokenizer, error) {
	if cfg.URL == "" {
		return nil, errors.New("sglang.url is required")
	}
	timeout, err := parseHTTPDuration(cfg.Timeout, defaultHTTPRenderTimeout)
	if err != nil {
		return nil, fmt.Errorf("invalid sglang.timeout: %w", err)
	}
	if timeout > time.Duration(math.MaxInt64)-sglangFallbackGrace {
		return nil, errors.New("sglang.timeout is too large")
	}
	return &sglangHTTPTokenizer{client: &http.Client{Transport: newRenderTransport()}, baseURL: strings.TrimRight(cfg.URL, "/"), model: model, timeout: timeout}, nil
}

// Allow the HTTP deadline to return a load fallback before the director aborts the producer.
func (s *sglangHTTPTokenizer) produceTimeout() time.Duration { return s.timeout + sglangFallbackGrace }

func (s *sglangHTTPTokenizer) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedPrompt, error) {
	var payload fwkrh.PayloadMap
	switch {
	case body.Completions != nil:
		if ids := body.Completions.Prompt.TokenIDs; len(ids) > 0 {
			return &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{ids}}, nil
		}
		payload, _ = completionsPayload(body).AsMap()
	case body.ChatCompletions != nil:
		for _, message := range body.ChatCompletions.Messages {
			for _, part := range message.Content.Structured {
				if part.Type != "text" {
					return nil, errors.New("SGLang cache tokenization supports text content only")
				}
			}
		}
		payload, _ = chatPayload(body).AsMap()
	case body.Generate != nil:
		return &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{body.Generate.TokenIDs}, MultiModalFeatures: convertMMFeaturesToUpstream(body.Generate.Features)}, nil
	default:
		return nil, errors.New("unsupported SGLang tokenization request")
	}
	payload = maps.Clone(payload)
	delete(payload, "stream")
	delete(payload, "stream_options")
	payload["model"] = s.model
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodPost, s.baseURL+"/v1/tokenize", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("SGLang tokenize: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodySnippetBytes))
		return nil, fmt.Errorf("SGLang tokenize status %d: %s", resp.StatusCode, snippet)
	}
	var wire struct {
		Tokens json.RawMessage `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&wire); err != nil {
		return nil, err
	}
	var tokens [][]uint32
	if err := json.Unmarshal(wire.Tokens, &tokens); err != nil {
		var single []uint32
		if err := json.Unmarshal(wire.Tokens, &single); err != nil {
			return nil, fmt.Errorf("invalid SGLang token IDs: %w", err)
		}
		tokens = [][]uint32{single}
	}
	if len(tokens) == 0 || len(tokens[0]) == 0 {
		return nil, errors.New("SGLang tokenize returned no tokens")
	}
	return &fwkrh.TokenizedPrompt{PerPromptTokens: tokens}, nil
}
