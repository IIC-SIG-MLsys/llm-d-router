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

	controlv1 "github.com/llm-d/llm-d-router/pkg/distkvproto"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/cachecost"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/test/utils"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/types"
)

type routingController struct {
	fakeController
	calls  int32
	lookup func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error)
}

func (f *routingController) LookupRouting(ctx context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
	atomic.AddInt32(&f.calls, 1)
	return f.lookup(ctx, req)
}

func newRoutingProducer(t *testing.T, f *routingController, ttl string) *Producer {
	t.Helper()
	// The embedded stub is registered through the same test listener.
	addr := startRoutingController(t, f)
	p, err := New(context.Background(), "sglang-test", PluginConfig{ControllerAddr: addr, RoutingMode: "sglang", Model: "m1", LookupCacheTTL: ttl})
	require.NoError(t, err)
	t.Cleanup(func() { _ = p.client.Close() })
	return p
}

func TestSGLangRoutingScoresInstancesAndSubtractsReplay(t *testing.T) {
	f := &routingController{lookup: func(_ context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		require.Equal(t, "m1", req.GetModel())
		require.Equal(t, testRequest().Body.TokenizedPrompt.PerPromptTokens[0], req.GetTokenIds())
		require.Len(t, req.GetCandidates(), 3)
		require.Equal(t, "default/pod-1", req.GetCandidates()[0].GetInstanceId())
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}, Matches: []*controlv1.RoutingMatch{
			{InstanceId: "default/pod-1", NodeId: "n1", Status: controlv1.Status_STATUS_OK, MatchedTokens: 8, ReplayTokens: 3},
			{InstanceId: "default/pod-2", NodeId: "n1", Status: controlv1.Status_STATUS_NOT_FOUND},
			{InstanceId: "default/pod-3", NodeId: "n2", Status: controlv1.Status_STATUS_ERROR, MatchedTokens: 10},
			{InstanceId: "default/pod-2", NodeId: "wrong-node", Status: controlv1.Status_STATUS_OK, MatchedTokens: 10},
		}}, nil
	}}
	p := newRoutingProducer(t, f, "0s")
	eps := []fwksched.Endpoint{endpoint("n1", "pod-1"), endpoint("n1", "pod-2"), endpoint("n2", "pod-3")}
	require.NoError(t, p.Produce(context.Background(), testRequest(), eps))
	require.Equal(t, 5, matchInfo(eps[0], p.dk.String()).MatchBlocks())
	require.Equal(t, 0, matchInfo(eps[1], p.dk.String()).MatchBlocks())
	require.Equal(t, 0, matchInfo(eps[2], p.dk.String()).MatchBlocks())
	require.Equal(t, 10, matchInfo(eps[0], p.dk.String()).TotalBlocks())
	require.Equal(t, 1, matchInfo(eps[0], p.dk.String()).BlockSizeTokens())
	require.Zero(t, atomic.LoadInt32(&f.fakeController.lookups))
}

func TestSGLangFailedQueryKeepsOnlyCompleteDeviceBlocksBeforeLastToken(t *testing.T) {
	f := &routingController{lookup: func(_ context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		require.EqualValues(t, 8, req.Candidates[0].DevicePrefixTokens)
		return nil, errors.New("controller unavailable")
	}}
	p := newRoutingProducer(t, f, "0s")
	p.devicePrefixProducer = "gpu"
	ep := endpoint("node", "pod")
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(5, 5, 2).WithDeviceCachedBlockCount(5))
	require.NoError(t, p.Produce(context.Background(), testRequest(), []fwksched.Endpoint{ep}))
	require.Equal(t, 8, matchInfo(ep, p.dk.String()).MatchBlocks())
	raw, _ := ep.Get(p.costKey.String())
	require.Equal(t, &cachecost.MatchInfo{TotalTokens: 10, DeviceTokens: 8}, raw)
}

func TestSGLangRoutingCacheIncludesTokensModelAndCandidateIdentity(t *testing.T) {
	f := &routingController{lookup: func(_ context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}}, nil
	}}
	p := newRoutingProducer(t, f, "10s")
	req := &controlv1.LookupRoutingRequest{Model: "m1", TokenIds: []uint32{1, 2}, Candidates: []*controlv1.RoutingCandidate{{InstanceId: "default/pod", NodeId: "node"}}}
	for i := 0; i < 2; i++ {
		_, err := p.lookupRoutingCached(context.Background(), req)
		require.NoError(t, err)
	}
	require.EqualValues(t, 1, atomic.LoadInt32(&f.calls))
	req.Candidates[0].InstanceId = "replacement"
	_, err := p.lookupRoutingCached(context.Background(), req)
	require.NoError(t, err)
	req.Candidates[0].NodeId = "other-node"
	_, err = p.lookupRoutingCached(context.Background(), req)
	require.NoError(t, err)
	req.Candidates[0].DevicePrefixTokens = 1
	_, err = p.lookupRoutingCached(context.Background(), req)
	require.NoError(t, err)
	req.TokenIds = []uint32{1, 3}
	_, err = p.lookupRoutingCached(context.Background(), req)
	require.NoError(t, err)
	req.Model = "m2"
	_, err = p.lookupRoutingCached(context.Background(), req)
	require.NoError(t, err)
	require.EqualValues(t, 6, atomic.LoadInt32(&f.calls))
}

func TestSGLangRoutingUnsupportedInputsAndFailuresClearScores(t *testing.T) {
	f := &routingController{lookup: func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		return nil, errors.New("offline")
	}}
	p := newRoutingProducer(t, f, "0s")
	for _, test := range []struct {
		name, result string
		edit         func(*fwkrh.TokenizedPrompt)
	}{
		{"estimate", "unsupported_tokens", func(tp *fwkrh.TokenizedPrompt) { tp.Estimated = true }},
		{"batch", "unsupported_batch", func(tp *fwkrh.TokenizedPrompt) { tp.PerPromptTokens = append(tp.PerPromptTokens, []uint32{1}) }},
		{"salt", "unsupported_salt", func(tp *fwkrh.TokenizedPrompt) { tp.CacheSalt = "tenant" }},
		{"multimodal", "unsupported_multimodal", func(tp *fwkrh.TokenizedPrompt) { tp.MultiModalFeatures = []fwkrh.MultiModalFeature{{Hash: "image"}} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := testRequest()
			test.edit(req.Body.TokenizedPrompt)
			ep := endpoint("n1", "pod")
			metric := routingRequests.WithLabelValues(p.typedName.Name, test.result)
			before := testutil.ToFloat64(metric)
			require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
			require.Zero(t, matchInfo(ep, p.dk.String()).MatchBlocks())
			require.Equal(t, before+1, testutil.ToFloat64(metric))
		})
	}
	require.Zero(t, atomic.LoadInt32(&f.calls))
	ep := endpoint("n1", "pod")
	require.NoError(t, p.Produce(context.Background(), testRequest(), []fwksched.Endpoint{ep}))
	require.Zero(t, matchInfo(ep, p.dk.String()).MatchBlocks())
	require.EqualValues(t, 1, atomic.LoadInt32(&f.calls))
}

func TestSGLangFactoryRequiresModelWithoutPools(t *testing.T) {
	handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
	_, err := PluginFactory("s", json.NewDecoder(strings.NewReader(`{"controllerAddr":"unused:5556","routingMode":"sglang"}`)), handle)
	require.ErrorContains(t, err, "model is required")
	p, err := PluginFactory("s", json.NewDecoder(strings.NewReader(`{"controllerAddr":"unused:5556","routingMode":"sglang","model":"m"}`)), handle)
	require.NoError(t, err)
	require.Zero(t, p.(*Producer).lookupTTL)
	require.NoError(t, p.(*Producer).client.Close())
}

func startRoutingController(t *testing.T, f *routingController) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := grpc.NewServer()
	controlv1.RegisterCacheServerControllerServer(server, f)
	go func() { _ = server.Serve(lis) }()
	t.Cleanup(server.Stop)
	return lis.Addr().String()
}

func TestSGLangRoutingCombinesConfirmedDevicePrefixAndExternalIncrement(t *testing.T) {
	f := &routingController{lookup: func(_ context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		require.EqualValues(t, 4, req.GetCandidates()[0].GetDevicePrefixTokens())
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}, Matches: []*controlv1.RoutingMatch{{InstanceId: "default/pod", NodeId: "n1", Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: 4, MatchedTokens: 6, ReplayTokens: 3}}}, nil
	}}
	p := newRoutingProducer(t, f, "0s")
	p.devicePrefixProducer = "gpu"
	ep := endpoint("n1", "pod")
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(2, 5, 2).WithCachedBlockCount(5).WithDeviceCachedBlockCount(2))
	require.Len(t, p.Consumes().Required, 2)
	require.NoError(t, p.Produce(context.Background(), testRequest(), []fwksched.Endpoint{ep}))
	require.Equal(t, 7, matchInfo(ep, p.dk.String()).MatchBlocks())
	f.lookup = func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		return nil, errors.New("offline")
	}
	require.NoError(t, p.Produce(context.Background(), testRequest(), []fwksched.Endpoint{ep}))
	require.Equal(t, 4, matchInfo(ep, p.dk.String()).MatchBlocks())
}

func TestSGLangRoutingRejectsLoRACacheIdentity(t *testing.T) {
	f := &routingController{lookup: func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		t.Fatal("LoRA request must not query base cache")
		return nil, nil
	}}
	p := newRoutingProducer(t, f, "0s")
	req := testRequest()
	req.Body.Payload = fwkrh.PayloadMap{"lora_path": "adapter"}
	ep := endpoint("n1", "pod")
	require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
	require.Zero(t, matchInfo(ep, p.dk.String()).MatchBlocks())
	require.Zero(t, atomic.LoadInt32(&f.calls))
}

func TestSGLangRoutingPromptLogprobsFallBackToLoad(t *testing.T) {
	for _, payload := range []fwkrh.PayloadMap{{"echo": true, "logprobs": float64(1)}, {"logprob_start_len": float64(0)}} {
		f := &routingController{lookup: func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
			t.Fatal("prompt logprobs must not query cache")
			return nil, nil
		}}
		p := newRoutingProducer(t, f, "0s")
		req := testRequest()
		req.Body.Payload = payload
		ep := endpoint("n1", "pod")
		require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
		require.Zero(t, matchInfo(ep, p.dk.String()).MatchBlocks())
	}
}

type gpuDataProducer struct{}

func (*gpuDataProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "gpu", Name: "gpu"}
}
func (*gpuDataProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu"): attrprefix.PrefixCacheMatchInfo{}}
}
func (*gpuDataProducer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{Required: map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: fwkrh.TokenizedPrompt{}}}
}
func (*gpuDataProducer) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

func TestConfiguredDeviceProducerRunsBeforeRouting(t *testing.T) {
	for _, mode := range []string{"", "vllm", "sglang"} {
		t.Run(mode, func(t *testing.T) {
			handle := plugin.NewEppHandle(utils.NewTestContext(t), nil)
			tk, err := tokenproducer.PluginFactory("token-producer", json.NewDecoder(strings.NewReader(`{"modelName":"m","sglang":{"url":"http://unused"}}`)), handle)
			require.NoError(t, err)
			p, err := New(context.Background(), "router", PluginConfig{ControllerAddr: "unused:5556", RoutingMode: mode, Model: "m", DevicePrefixProducerName: "gpu"})
			require.NoError(t, err)
			defer p.client.Close()
			gpu := &gpuDataProducer{}
			ordered, err := datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{p, gpu, tk})
			require.NoError(t, err)
			require.Equal(t, []string{tk.TypedName().String(), gpu.TypedName().String(), p.TypedName().String()}, ordered)
			handle.AddPlugin(tk.TypedName().Name, tk)
			handle.AddPlugin(p.TypedName().Name, p)
			err = datalayer.CreateMissingDataProducers(context.Background(), map[string]string{}, map[string]plugin.FactoryFunc{}, handle)
			require.Error(t, err)
			p.devicePrefixProducer = ""
			_, err = datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{p, tk})
			require.NoError(t, err)
		})
	}
}

func TestConfiguredInFlightLoadRunsAfterCombinedCache(t *testing.T) {
	ctx, cancel := context.WithCancel(utils.NewTestContext(t))
	defer cancel()
	handle := plugin.NewEppHandle(ctx, nil)
	tk, err := tokenproducer.PluginFactory("token-producer", json.NewDecoder(strings.NewReader(`{"modelName":"m","sglang":{"url":"http://unused"}}`)), handle)
	require.NoError(t, err)
	p, err := New(ctx, "router", PluginConfig{ControllerAddr: "unused:5556", RoutingMode: "vllm", Model: "m", DevicePrefixProducerName: "gpu"})
	require.NoError(t, err)
	defer p.client.Close()
	load, err := inflightload.InFlightLoadProducerFactory("load", json.NewDecoder(strings.NewReader(`{"prefixMatchInfoProducerName":"router"}`)), handle)
	require.NoError(t, err)
	gpu := &gpuDataProducer{}
	ordered, err := datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{load, p, gpu, tk})
	require.NoError(t, err)
	require.Equal(t, []string{tk.TypedName().String(), gpu.TypedName().String(), p.TypedName().String(), load.TypedName().String()}, ordered)
	handle.AddPlugin(tk.TypedName().Name, tk)
	handle.AddPlugin(load.TypedName().Name, load)
	require.Error(t, datalayer.CreateMissingDataProducers(ctx, map[string]string{}, map[string]plugin.FactoryFunc{}, handle))
}

func TestSGLangRoutingRequiresRegisteredModelIdentity(t *testing.T) {
	for _, name := range []string{"other-model", "m1:adapter"} {
		for _, payloadModel := range []bool{false, true} {
			f := &routingController{lookup: func(context.Context, *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
				t.Fatal("different model identity must not query cache")
				return nil, nil
			}}
			p := newRoutingProducer(t, f, "0s")
			req := testRequest()
			if payloadModel {
				req.Body.Payload = fwkrh.PayloadMap{"model": name}
			} else {
				req.TargetModel = name
			}
			ep := endpoint("n1", "pod")
			require.NoError(t, p.Produce(context.Background(), req, []fwksched.Endpoint{ep}))
			require.Zero(t, matchInfo(ep, p.dk.String()).MatchBlocks())
			require.Zero(t, atomic.LoadInt32(&f.calls))
		}
	}
}

func TestRoutingCostSeparatesNetworkBytesFromDeviceRestore(t *testing.T) {
	match := &controlv1.RoutingMatch{NodeId: "target", DevicePrefixTokens: 4, MatchedTokens: 12,
		Ranges: []*controlv1.RoutingCacheRange{
			{StartToken: 4, EndToken: 8, Medium: "memory", Bytes: 128},
			{StartToken: 8, EndToken: 12, Medium: "memory", Bytes: 128, SourceNodeId: "source", Transport: "hux"},
			{StartToken: 12, EndToken: 16, Medium: "memory", Bytes: 128, SourceNodeId: "source", Transport: "hux"},
		}}
	info, ok := routingCostInfo(17, match)
	require.True(t, ok)
	require.EqualValues(t, 384, info.BytesByMedium["memory"])
	require.EqualValues(t, 256, info.RemoteBytesByTransport["hux"])
	require.EqualValues(t, 1, info.RemoteSources)
	match.Ranges[1].Medium = "file"
	_, ok = routingCostInfo(17, match)
	require.False(t, ok)
	match.Ranges[1].Medium = "memory"
	match.Ranges[1].Transport = ""
	_, ok = routingCostInfo(17, match)
	require.False(t, ok)
}

func TestSGLangRoutingKeepsPodRanksIndependent(t *testing.T) {
	f := &routingController{lookup: func(_ context.Context, req *controlv1.LookupRoutingRequest) (*controlv1.LookupRoutingResponse, error) {
		require.Len(t, req.Candidates, 2)
		require.Equal(t, "default/pod-rank-0", req.Candidates[0].InstanceId)
		require.Equal(t, "default/pod-rank-1", req.Candidates[1].InstanceId)
		return &controlv1.LookupRoutingResponse{Header: &controlv1.ResponseHeader{Status: controlv1.Status_STATUS_OK}, Matches: []*controlv1.RoutingMatch{
			{InstanceId: "default/pod-rank-1", NodeId: "n1", Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: 2, MatchedTokens: 6},
			{InstanceId: "default/pod-rank-0", NodeId: "n1", Status: controlv1.Status_STATUS_OK, DevicePrefixTokens: 4, MatchedTokens: 0},
		}}, nil
	}}
	p := newRoutingProducer(t, f, "0s")
	p.devicePrefixProducer = "gpu"
	eps := make([]fwksched.Endpoint, 2)
	for i := range eps {
		eps[i] = fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "default", Name: fmt.Sprintf("pod-rank-%d", i)}, PodName: "pod", NodeName: "n1", Address: "host", Port: fmt.Sprint(8000 + i), RankIndex: i}, nil, nil)
		count := 2 - i
		eps[i].Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("gpu").String(), attrprefix.NewPrefixCacheMatchInfo(count, 5, 2).WithDeviceCachedBlockCount(count))
	}
	require.NoError(t, p.Produce(context.Background(), testRequest(), eps))
	require.Equal(t, 4, matchInfo(eps[0], p.dk.String()).MatchBlocks())
	require.Equal(t, 8, matchInfo(eps[1], p.dk.String()).MatchBlocks())
}
