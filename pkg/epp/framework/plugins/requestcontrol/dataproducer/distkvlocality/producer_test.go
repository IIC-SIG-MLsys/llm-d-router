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
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	controlv1 "github.com/llm-d/llm-d-router/pkg/distkvproto"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/cachecost"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	prefixscorer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/types"
)

// fakeController is a DistKV CacheServerController stub for routing queries.
type fakeController struct {
	controlv1.UnimplementedCacheServerControllerServer
	lookups    int32
	vllmLookup func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error)
	vllmCalls  int32
}

func (f *fakeController) LookupVLLMRouting(ctx context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
	atomic.AddInt32(&f.vllmCalls, 1)
	if f.vllmLookup != nil {
		return f.vllmLookup(ctx, req)
	}
	return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}}, nil
}

func (f *fakeController) GlobalLookup(ctx context.Context,
	req *controlv1.GlobalLookupRequest,
) (*controlv1.GlobalLookupResponse, error) {
	atomic.AddInt32(&f.lookups, 1)
	return &controlv1.GlobalLookupResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}}, nil
}

func startFakeController(t *testing.T, fc *fakeController) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gs := grpc.NewServer()
	controlv1.RegisterCacheServerControllerServer(gs, fc)
	go func() { _ = gs.Serve(lis) }()
	t.Cleanup(gs.Stop)
	return lis.Addr().String()
}

func testRequest() *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:   "req-1",
		TargetModel: "m1",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
			},
		},
	}
}

func endpoint(node, addr string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: addr, Namespace: "default"},
		PodName:        addr,
		NodeName:       node,
		Address:        addr,
		Port:           "8000",
	}, nil, nil)
}

func matchInfo(ep fwksched.Endpoint, key string) *attrprefix.PrefixCacheMatchInfo {
	raw, ok := ep.Get(key)
	if !ok {
		return nil
	}
	info, _ := raw.(*attrprefix.PrefixCacheMatchInfo)
	return info
}

func newVLLMProducer(t *testing.T, fc *fakeController, config PluginConfig) *Producer {
	t.Helper()
	config.ControllerAddr = startFakeController(t, fc)
	p, err := New(context.Background(), "distkv", config)
	require.NoError(t, err)
	t.Cleanup(func() { _ = p.client.Close() })
	return p
}

func TestVLLMRetainsEveryCandidateAndCombinesItsDevicePrefix(t *testing.T) {
	fc := &fakeController{vllmLookup: func(_ context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		require.Equal(t, "m1", req.GetModel())
		require.Equal(t, []string{"p1", "p2"}, req.GetPools())
		require.EqualValues(t, 6, req.Candidates[0].DevicePrefixTokens)
		require.EqualValues(t, 2, req.Candidates[1].DevicePrefixTokens)
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}, Matches: []*controlv1.RoutingMatch{
			{InstanceId: "default/pod-a", NodeId: "n1", Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: 6, MatchedTokens: 2},
			{InstanceId: "default/pod-b", NodeId: "n1", Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: 2, MatchedTokens: 6},
			{InstanceId: "default/pod-c", NodeId: "n2", Status: controlv1.Status_STATUS_OK, MatchedTokens: 4},
		}}, nil
	}}
	p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pools: []PoolConfig{{Pool: "p1"}, {Pool: "p2"}}, DevicePrefixProducerName: "gpu"})
	require.Zero(t, p.lookupTTL)
	eps := []fwksched.Endpoint{endpoint("n1", "pod-a"), endpoint("n1", "pod-b"), endpoint("n2", "pod-c")}
	gpuKey := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String()
	eps[0].Put(gpuKey, attrprefix.NewPrefixCacheMatchInfo(3, 5, 2).WithDeviceCachedBlockCount(3))
	eps[1].Put(gpuKey, attrprefix.NewPrefixCacheMatchInfo(1, 5, 2).WithDeviceCachedBlockCount(1))
	require.NoError(t, p.Produce(context.Background(), testRequest(), eps))
	for i, want := range []int{8, 8, 4} {
		info := matchInfo(eps[i], p.dk.String())
		require.Equal(t, want, info.MatchBlocks())
		require.Equal(t, 10, info.TotalBlocks())
		require.Equal(t, 1, info.BlockSizeTokens())
	}
	require.Zero(t, atomic.LoadInt32(&fc.lookups))
}

func TestVLLMUnknownOrFailedQueryPreservesGPUAndCostBaseline(t *testing.T) {
	fc := &fakeController{}
	p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pool: "p1", DevicePrefixProducerName: "gpu"})
	ep := endpoint("n1", "pod")
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(2, 5, 2).WithDeviceCachedBlockCount(2))
	for _, status := range []controlv1.Status{controlv1.Status_STATUS_OK, controlv1.Status_STATUS_ERROR} {
		fc.vllmLookup = func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
			return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: status}}, nil
		}
		require.NoError(t, p.Produce(context.Background(), testRequest(), []fwksched.Endpoint{ep}))
		require.Equal(t, 4, matchInfo(ep, p.dk.String()).MatchBlocks())
		raw, ok := ep.Get(p.costKey.String())
		require.True(t, ok)
		require.Equal(t, &cachecost.MatchInfo{TotalTokens: 10, DeviceTokens: 4}, raw)
	}
}

func TestVLLMGroupedDeviceEvidenceOnlyCreditsValidatedResponse(t *testing.T) {
	for _, mode := range []string{"verified", "missing metadata", "old controller", "offline"} {
		t.Run(mode, func(t *testing.T) {
			fc := &fakeController{vllmLookup: func(_ context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
				candidate := req.Candidates[0]
				require.Zero(t, candidate.DevicePrefixTokens)
				require.True(t, candidate.GroupedDeviceCache)
				require.EqualValues(t, 2, candidate.DeviceBlockSize)
				require.EqualValues(t, 4, candidate.DeviceGroups[0].Ranges[0].EndToken)
				if mode == "offline" {
					return nil, errors.New("controller unavailable")
				}
				response := &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}}
				if mode != "missing metadata" {
					device := uint32(0)
					if mode == "verified" {
						device = 4
					}
					response.Matches = []*controlv1.RoutingMatch{{InstanceId: candidate.InstanceId, NodeId: candidate.NodeId, Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: device}}
				}
				return response, nil
			}}
			p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pool: "p1", DevicePrefixProducerName: "gpu"})
			ep := endpoint("node", "pod")
			info := attrprefix.NewPrefixCacheMatchInfo(2, 5, 2).WithDeviceCachedBlockCount(2).WithDeviceCacheGroups([]attrprefix.DeviceCacheGroup{{GroupIdx: 0, Ranges: []attrprefix.DeviceTokenRange{{Start: 0, End: 4}}}}, true)
			ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), info)
			require.NoError(t, p.Produce(context.Background(), testRequest(), []fwksched.Endpoint{ep}))
			want := 0
			if mode == "verified" {
				want = 4
			}
			require.Equal(t, want, matchInfo(ep, p.dk.String()).MatchBlocks())
			raw, _ := ep.Get(p.costKey.String())
			require.EqualValues(t, want, raw.(*cachecost.MatchInfo).DeviceTokens)
		})
	}
}

func TestVLLMPromptLogprobsDisablesCacheRoutingIncludingZero(t *testing.T) {
	for _, payload := range []fwkrh.PayloadMap{
		{"prompt_logprobs": float64(-1)}, {"prompt_logprobs": float64(0)}, {"prompt_logprobs": float64(1)}, {"echo": true, "logprobs": float64(0)}, {"echo": true, "top_logprobs": float64(0)},
	} {
		fc := &fakeController{}
		p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pool: "p1", DevicePrefixProducerName: "gpu"})
		ep := endpoint("node", "pod")
		ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(2, 5, 2).WithDeviceCachedBlockCount(2))
		request := testRequest()
		request.Body.Payload = payload
		if payload["top_logprobs"] != nil {
			request.Body.ChatCompletions = &fwkrh.ChatCompletionsRequest{}
		}
		require.NoError(t, p.Produce(context.Background(), request, []fwksched.Endpoint{ep}))
		require.Zero(t, matchInfo(ep, p.dk.String()).MatchBlocks())
		require.Zero(t, atomic.LoadInt32(&fc.vllmCalls))
	}
}

func TestVLLMCostIncludesEveryRequiredRankAndDeviceOverlap(t *testing.T) {
	fc := &fakeController{vllmLookup: func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}, Matches: []*controlv1.RoutingMatch{{
			InstanceId: "default/pod", NodeId: "n1", Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: 4, MatchedTokens: 4,
			Ranges: []*controlv1.RoutingCacheRange{{StartToken: 0, EndToken: 8, Medium: "memory", Bytes: 100}, {StartToken: 0, EndToken: 8, Medium: "memory", Bytes: 200}, {StartToken: 4, EndToken: 8, Medium: "file", Bytes: 50}},
		}}}, nil
	}}
	p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pool: "p1", DevicePrefixProducerName: "gpu"})
	ep := endpoint("n1", "pod")
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(2, 5, 2).WithDeviceCachedBlockCount(2))
	require.NoError(t, p.Produce(context.Background(), testRequest(), []fwksched.Endpoint{ep}))
	raw, _ := ep.Get(p.costKey.String())
	require.Equal(t, &cachecost.MatchInfo{TotalTokens: 10, DeviceTokens: 4, ExternalTokens: 4, BytesByMedium: map[string]uint64{"memory": 300, "file": 50}}, raw)
	require.Contains(t, p.Produces(), p.costKey)
}

func TestVLLMPoolsOnlyConfigBindsRequestModelAndCachesAllInputs(t *testing.T) {
	fc := &fakeController{}
	p := newVLLMProducer(t, fc, PluginConfig{Pools: []PoolConfig{{Model: "weight-path", Pool: "p1"}}, LookupCacheTTL: "10s"})
	req := testRequest()
	ep := endpoint("n1", "pod")
	for range 2 {
		require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
	}
	require.EqualValues(t, 1, atomic.LoadInt32(&fc.vllmCalls))
	query := &controlv1.LookupRoutingRequest{Model: "m1", TokenIds: []uint32{1, 2}, Pools: []string{"p1"}, Candidates: []*controlv1.RoutingCandidate{{InstanceId: "default/pod", NodeId: "n1"}}}
	_, err := p.lookupRoutingCached(context.Background(), query)
	require.NoError(t, err)
	query.Pools[0] = "p2"
	_, err = p.lookupRoutingCached(context.Background(), query)
	require.NoError(t, err)
	require.EqualValues(t, 3, atomic.LoadInt32(&fc.vllmCalls))
}

func TestVLLMCombinedPrefixAndLoadCanChooseEitherReachableCandidate(t *testing.T) {
	fc := &fakeController{vllmLookup: func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}, Matches: []*controlv1.RoutingMatch{{InstanceId: "default/external", NodeId: "n2", Status: controlv1.Status_STATUS_OK, MatchedTokens: 8}}}, nil
	}}
	p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pool: "p1", DevicePrefixProducerName: "gpu"})
	eps := []fwksched.Endpoint{endpoint("n1", "device"), endpoint("n2", "external")}
	eps[0].Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(3, 3, 3).WithDeviceCachedBlockCount(3))
	request := testRequest()
	require.NoError(t, p.Produce(context.Background(), request, eps))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	load, err := inflightload.InFlightLoadProducerFactory("load", json.NewDecoder(strings.NewReader(`{"prefixMatchInfoProducerName":"distkv"}`)), plugin.NewEppHandle(ctx, nil))
	require.NoError(t, err)
	require.NoError(t, load.(requestcontrol.DataProducer).Produce(ctx, request, eps))
	cache, err := prefixscorer.New(ctx, "prefix", "distkv")
	require.NoError(t, err)
	loadScorer, err := tokenload.TokenLoadScorerFactory("token-load", json.NewDecoder(strings.NewReader(`{"queueThresholdTokens":20,"inFlightLoadProducerName":"load"}`)), nil)
	require.NoError(t, err)
	cacheScores := cache.Score(ctx, request, eps)
	loadScores := loadScorer.(fwksched.Scorer).Score(ctx, request, eps)
	require.Greater(t, 3*cacheScores[eps[0]]+2*loadScores[eps[0]], 3*cacheScores[eps[1]]+2*loadScores[eps[1]])
	eps[0].Put(attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName("load").String(), &attrconcurrency.InFlightLoad{Tokens: 20})
	loadScores = loadScorer.(fwksched.Scorer).Score(ctx, request, eps)
	require.Less(t, 3*cacheScores[eps[0]]+2*loadScores[eps[0]], 3*cacheScores[eps[1]]+2*loadScores[eps[1]])
}

func TestInstanceRoutingRejectsObsoleteOverrides(t *testing.T) {
	for _, config := range []PluginConfig{{ControllerAddr: "unused:5556", GroupID: "group"}, {ControllerAddr: "unused:5556", ChunkSize: 64}} {
		_, err := New(context.Background(), "router", config)
		require.ErrorContains(t, err, "empty groupID and zero chunkSize")
	}
}

func TestRoutingBatchesFleetAndKeepsEarlierMatchesOnFailure(t *testing.T) {
	for _, failure := range []string{"none", "offline", "cancel"} {
		t.Run(failure, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			fc := &fakeController{vllmLookup: func(_ context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
				require.LessOrEqual(t, len(req.Candidates), routingBatchSize)
				if req.Candidates[0].InstanceId == "default/pod-32" {
					if failure == "offline" {
						return nil, errors.New("second batch unavailable")
					}
					if failure == "cancel" {
						cancel()
						return nil, context.Canceled
					}
				}
				response := &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}}
				for _, c := range req.Candidates {
					response.Matches = append(response.Matches, &controlv1.RoutingMatch{InstanceId: c.InstanceId, NodeId: c.NodeId, Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: c.DevicePrefixTokens, MatchedTokens: 6 - c.DevicePrefixTokens})
				}
				return response, nil
			}}
			p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pool: "p1", DevicePrefixProducerName: "gpu"})
			eps := make([]fwksched.Endpoint, 33)
			for i := range eps {
				eps[i] = endpoint("node", fmt.Sprintf("pod-%d", i))
			}
			eps[32].Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(1, 5, 2).WithDeviceCachedBlockCount(1))
			require.NoError(t, p.Produce(ctx, testRequest(), eps))
			for _, ep := range eps[:32] {
				require.Equal(t, 6, matchInfo(ep, p.dk.String()).MatchBlocks())
			}
			want := 6
			if failure != "none" {
				want = 2
			}
			require.Equal(t, want, matchInfo(eps[32], p.dk.String()).MatchBlocks())
			require.EqualValues(t, 2, atomic.LoadInt32(&fc.vllmCalls))
		})
	}
}

func TestCanceledRoutingObservationIsNotCached(t *testing.T) {
	started := make(chan struct{})
	var calls atomic.Int32
	fc := &fakeController{vllmLookup: func(ctx context.Context, _ *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		if calls.Add(1) == 1 {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}}, nil
	}}
	p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pool: "p1", LookupCacheTTL: "10s"})
	req := &controlv1.LookupRoutingRequest{Model: "m1", Candidates: []*controlv1.RoutingCandidate{{InstanceId: "pod", NodeId: "node"}}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := p.lookupRoutingCached(ctx, req); done <- err }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("routing query did not reach controller")
	}
	cancel()
	require.ErrorIs(t, <-done, context.Canceled)
	_, err := p.lookupRoutingCached(context.Background(), req)
	require.NoError(t, err)
	require.EqualValues(t, 2, calls.Load())
}

func TestRoutingCoverageCannotExceedPromptByHidingBehindReplay(t *testing.T) {
	fc := &fakeController{vllmLookup: func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}, Matches: []*controlv1.RoutingMatch{{InstanceId: "default/pod", NodeId: "node", Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: 4, MatchedTokens: 8, ReplayTokens: 4}}}, nil
	}}
	p := newVLLMProducer(t, fc, PluginConfig{Model: "m1", Pool: "p1", DevicePrefixProducerName: "gpu"})
	ep := endpoint("node", "pod")
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(2, 5, 2).WithDeviceCachedBlockCount(2))
	require.NoError(t, p.Produce(context.Background(), testRequest(), []fwksched.Endpoint{ep}))
	require.Equal(t, 4, matchInfo(ep, p.dk.String()).MatchBlocks())
}

func TestRoutingRejectsNonpositiveTimeoutAndNegativeCacheTTL(t *testing.T) {
	for _, config := range []PluginConfig{{LookupTimeout: "0s"}, {LookupTimeout: "-1s"}, {LookupCacheTTL: "-1s"}} {
		config.ControllerAddr = "unused:5556"
		_, err := New(context.Background(), "router", config)
		require.Error(t, err)
	}
}
