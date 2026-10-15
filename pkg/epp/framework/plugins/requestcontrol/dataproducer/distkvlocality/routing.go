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

package distkvlocality

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	controlv1 "github.com/llm-d/llm-d-router/pkg/distkvproto"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/cachecost"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type cachedRoutingLookup struct {
	matches []*controlv1.RoutingMatch
	err     error
	at      time.Time
}

const routingBatchSize = 32

func (p *Producer) produceRouting(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	p.writeMatchInfo(endpoints, 0, 0, 1)
	result := "unsupported_tokens"
	defer func() { routingRequests.WithLabelValues(p.typedName.Name, result).Inc() }()
	if request == nil || request.Body == nil {
		return nil
	}
	model := p.routingModel
	if model == "" && p.routingMode != "sglang" {
		model = request.TargetModel
	}
	if model == "" || (request.TargetModel != "" && request.TargetModel != model) {
		result = "unsupported_model"
		return nil
	}
	if request.Body.Payload != nil {
		if payload, ok := request.Body.Payload.AsMap(); ok {
			if payloadModel, ok := payload["model"].(string); ok && payloadModel != "" && payloadModel != model {
				result = "unsupported_model"
				return nil
			}
			if payload["lora_path"] != nil {
				result = "unsupported_lora"
				return nil
			}
			if p.routingMode != "sglang" && payload["prompt_logprobs"] != nil {
				result = "unsupported_logprobs"
				return nil
			}
			if start, ok := payload["logprob_start_len"].(float64); ok && start >= 0 {
				result = "unsupported_logprobs"
				return nil
			}
			if echo, ok := payload["echo"].(bool); ok && echo {
				if p.routingMode != "sglang" {
					field := "logprobs"
					if request.Body.ChatCompletions != nil {
						field = "top_logprobs"
					}
					if payload[field] != nil {
						result = "unsupported_logprobs"
						return nil
					}
				}
				if count, ok := payload["logprobs"].(float64); ok && count != 0 {
					result = "unsupported_logprobs"
					return nil
				}
			}
		}
	}
	if tokenproducer.CacheSaltFromBody(request.Body) != "" {
		result = "unsupported_salt"
		return nil
	}
	if request.Body.ChatCompletions != nil {
		for _, message := range request.Body.ChatCompletions.Messages {
			for _, part := range message.Content.Structured {
				if part.Type != "text" {
					result = "unsupported_multimodal"
					return nil
				}
			}
		}
	}
	tp := request.Body.TokenizedPrompt
	if tp == nil || tp.Estimated || len(tp.PerPromptTokens) == 0 || len(tp.PerPromptTokens[0]) == 0 {
		return nil
	}
	if len(tp.PerPromptTokens) != 1 {
		result = "unsupported_batch"
		return nil
	}
	if tp.CacheSalt != "" {
		result = "unsupported_salt"
		return nil
	}
	if len(tp.MultiModalFeatures) != 0 {
		result = "unsupported_multimodal"
		return nil
	}
	tokens := tp.PerPromptTokens[0]
	p.writeMatchInfo(endpoints, 0, len(tokens), 1)
	candidates := make([]*controlv1.RoutingCandidate, 0, len(endpoints))
	for _, ep := range endpoints {
		devicePrefix := p.devicePrefixTokens(ep, len(tokens))
		ep.Put(p.dk.String(), attrprefix.NewPrefixCacheMatchInfo(devicePrefix, len(tokens), 1))
		ep.Put(p.costKey.String(), &cachecost.MatchInfo{TotalTokens: uint32(len(tokens)), DeviceTokens: uint32(devicePrefix)})
		md := ep.GetMetadata()
		if md != nil && md.PodName != "" && md.NodeName != "" {
			candidate := &controlv1.RoutingCandidate{InstanceId: routingInstanceID(ep), NodeId: md.NodeName, DevicePrefixTokens: uint32(devicePrefix)}
			if p.routingMode != "sglang" && p.devicePrefixProducer != "" {
				key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(p.devicePrefixProducer).String()
				if raw, ok := ep.Get(key); ok {
					if info, ok := raw.(*attrprefix.PrefixCacheMatchInfo); ok && info != nil && info.GroupedDeviceCache() {
						candidate.GroupedDeviceCache = true
						candidate.DeviceBlockSize = uint32(info.BlockSizeTokens())
						for _, g := range info.DeviceCacheGroups() {
							group := &controlv1.RoutingDeviceGroup{GroupIdx: g.GroupIdx}
							for _, r := range g.Ranges {
								group.Ranges = append(group.Ranges, &controlv1.RoutingDeviceRange{StartToken: r.Start, EndToken: r.End})
							}
							candidate.DeviceGroups = append(candidate.DeviceGroups, group)
						}
					}
				}
			}
			candidates = append(candidates, candidate)
		}
	}
	if len(candidates) == 0 {
		result = "unsupported_candidates"
		return nil
	}
	req := &controlv1.LookupRoutingRequest{Model: model, TokenIds: tokens, Candidates: candidates}
	if p.routingMode != "sglang" {
		for _, pool := range p.pools {
			req.Pools = append(req.Pools, pool.Pool)
		}
	}
	matches, err := p.lookupRoutingCached(ctx, req)
	if err != nil && len(matches) == 0 {
		result = "lookup_error"
		return nil
	}
	result = "miss"
	scores := make(map[string]int, len(matches))
	costs := make(map[string]*cachecost.MatchInfo, len(matches))
	valid := make(map[string]uint32, len(candidates))
	for _, c := range candidates {
		scores[c.GetInstanceId()+"\x00"+c.GetNodeId()] = int(c.GetDevicePrefixTokens())
		valid[c.GetInstanceId()+"\x00"+c.GetNodeId()] = c.GetDevicePrefixTokens()
		if c.GetGroupedDeviceCache() && p.routingMode != "sglang" {
			valid[c.GetInstanceId()+"\x00"+c.GetNodeId()] = uint32(max(0, len(tokens)-1))
		}
	}
	for _, match := range matches {
		key := match.GetInstanceId() + "\x00" + match.GetNodeId()
		devicePrefix, known := valid[key]
		if !known || match.GetDevicePrefixTokens() > devicePrefix || match.GetReplayTokens() > match.GetMatchedTokens() {
			continue
		}
		if uint64(match.GetDevicePrefixTokens())+uint64(match.GetMatchedTokens()) > uint64(len(tokens)) {
			continue
		}
		score := uint64(scores[key])
		if match.GetStatus() == controlv1.Status_STATUS_OK {
			score = uint64(match.GetDevicePrefixTokens()) + uint64(match.GetMatchedTokens()) - uint64(match.GetReplayTokens())
		}
		if score > uint64(len(tokens)) {
			continue
		}
		if match.GetStatus() == controlv1.Status_STATUS_OK {
			cost, valid := routingCostInfo(len(tokens), match)
			if !valid {
				continue
			}
			costs[key] = cost
		}
		scores[key] = int(score)
		log.FromContext(ctx).V(logging.DEBUG).Info("DistKV candidate cache match", "instance", match.GetInstanceId(), "node", match.GetNodeId(), "status", match.GetStatus().String(), "deviceTokens", match.GetDevicePrefixTokens(), "externalTokens", match.GetMatchedTokens(), "replayTokens", match.GetReplayTokens(), "scoreTokens", score)
	}
	for _, ep := range endpoints {
		if md := ep.GetMetadata(); md != nil {
			key := routingInstanceID(ep) + "\x00" + md.NodeName
			score, queried := scores[key]
			if !queried {
				continue
			}
			if score > 0 {
				result = "hit"
			}
			ep.Put(p.dk.String(), attrprefix.NewPrefixCacheMatchInfo(score, len(tokens), 1))
			if cost := costs[key]; cost != nil {
				ep.Put(p.costKey.String(), cost)
			}
		}
	}
	if err != nil {
		result = "partial_" + result
	}
	return nil
}

func (p *Producer) lookupRoutingCached(ctx context.Context, req *controlv1.LookupRoutingRequest) ([]*controlv1.RoutingMatch, error) {
	if p.lookupTTL <= 0 {
		return p.lookupRouting(ctx, req)
	}
	encoded, err := proto.MarshalOptions{Deterministic: true}.Marshal(req)
	if err != nil {
		return nil, err
	}
	key := string(encoded)
	p.mu.Lock()
	if c, ok := p.routingLookups[key]; ok && time.Since(c.at) < p.lookupTTL {
		p.mu.Unlock()
		return c.matches, c.err
	}
	p.mu.Unlock()
	matches, err := p.lookupRouting(ctx, req)
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || (err != nil && len(matches) > 0) {
		return matches, err
	}
	p.mu.Lock()
	if len(p.routingLookups) >= p.lookupMax {
		for k, c := range p.routingLookups {
			if time.Since(c.at) >= p.lookupTTL {
				delete(p.routingLookups, k)
			}
		}
		if len(p.routingLookups) >= p.lookupMax {
			p.routingLookups = make(map[string]cachedRoutingLookup)
		}
	}
	p.routingLookups[key] = cachedRoutingLookup{matches: matches, err: err, at: time.Now()}
	p.mu.Unlock()
	return matches, err
}

func (p *Producer) lookupRouting(ctx context.Context, req *controlv1.LookupRoutingRequest) ([]*controlv1.RoutingMatch, error) {
	ctx, cancel := context.WithTimeout(ctx, p.lookupTimeout)
	defer cancel()
	var matches []*controlv1.RoutingMatch
	var lookupErr error
	for start := 0; start < len(req.Candidates); start += routingBatchSize {
		if err := ctx.Err(); err != nil {
			return matches, errors.Join(lookupErr, err)
		}
		batch := &controlv1.LookupRoutingRequest{Model: req.Model, TokenIds: req.TokenIds, Pools: req.Pools, Candidates: req.Candidates[start:min(start+routingBatchSize, len(req.Candidates))]}
		observed, err := p.client.lookupRouting(ctx, batch, p.lookupTimeout, p.routingMode != "sglang")
		matches = append(matches, observed...)
		lookupErr = errors.Join(lookupErr, err)
	}
	return matches, errors.Join(lookupErr, ctx.Err())
}

func (c *Client) lookupRouting(ctx context.Context, req *controlv1.LookupRoutingRequest, timeout time.Duration, vllm bool) ([]*controlv1.RoutingMatch, error) {
	rctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	var response *controlv1.LookupRoutingResponse
	var err error
	if vllm {
		response, err = c.stub.LookupVLLMRouting(rctx, req)
	} else {
		response, err = c.stub.LookupRouting(rctx, req)
	}
	if err != nil {
		return nil, fmt.Errorf("lookup routing: %w", err)
	}
	if response.GetHeader().GetStatus() != controlv1.Status_STATUS_OK {
		return nil, fmt.Errorf("lookup routing status %s", response.GetHeader().GetStatus())
	}
	return response.GetMatches(), nil
}

func (p *Producer) devicePrefixTokens(endpoint fwksched.Endpoint, total int) int {
	if p.devicePrefixProducer == "" {
		return 0
	}
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(p.devicePrefixProducer).String()
	raw, ok := endpoint.Get(key)
	if !ok {
		return 0
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info == nil || info.DeviceCachedBlockCount() <= 0 || info.BlockSizeTokens() <= 0 {
		return 0
	}
	if info.GroupedDeviceCache() {
		return 0
	}
	length := uint64(info.DeviceCachedBlockCount()) * uint64(info.BlockSizeTokens())
	if length > uint64(total) {
		return 0
	}
	if p.routingMode != "sglang" {
		return min(int(length), max(0, total-1))
	}
	limit := max(0, total-1) / info.BlockSizeTokens() * info.BlockSizeTokens()
	return min(int(length), limit)
}

func routingCostInfo(total int, match *controlv1.RoutingMatch) (*cachecost.MatchInfo, bool) {
	info := &cachecost.MatchInfo{TotalTokens: uint32(total), DeviceTokens: match.GetDevicePrefixTokens(), ExternalTokens: match.GetMatchedTokens(), ReplayTokens: match.GetReplayTokens(), BytesByMedium: make(map[string]uint64)}
	sources := make(map[string]bool)
	for _, r := range match.GetRanges() {
		if r.GetEndToken() <= r.GetStartToken() || uint64(r.GetEndToken()) > uint64(total) || (r.GetMedium() != "memory" && r.GetMedium() != "file") || r.GetBytes() > math.MaxUint64-info.BytesByMedium[r.GetMedium()] {
			return nil, false
		}
		info.BytesByMedium[r.GetMedium()] += r.GetBytes()
		if r.GetSourceNodeId() != "" && r.GetSourceNodeId() != match.GetNodeId() {
			if r.GetMedium() != "memory" || r.GetTransport() == "" {
				return nil, false
			}
			if info.RemoteBytesByTransport == nil {
				info.RemoteBytesByTransport = make(map[string]uint64)
			}
			if r.GetBytes() > math.MaxUint64-info.RemoteBytesByTransport[r.GetTransport()] {
				return nil, false
			}
			info.RemoteBytesByTransport[r.GetTransport()] += r.GetBytes()
			sources[r.GetSourceNodeId()] = true
		} else if r.GetTransport() != "" {
			return nil, false
		}
	}
	info.RemoteSources = uint32(len(sources))
	return info, true
}

func routingInstanceID(endpoint fwksched.Endpoint) string {
	md := endpoint.GetMetadata()
	if md == nil {
		return ""
	}
	if md.NamespacedName.Namespace == "" {
		return md.PodName
	}
	return md.NamespacedName.Namespace + "/" + md.PodName
}
