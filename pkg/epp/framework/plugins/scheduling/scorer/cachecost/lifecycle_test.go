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

package cachecost

import (
	"context"
	"fmt"
	"sync"
	"testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrcache "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/cachecost"
	"github.com/stretchr/testify/require"
)

func costRequest(id string, endpoint fwksched.Endpoint, tokens int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: id,
		Body:      &fwkrh.InferenceRequestBody{TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, tokens)}}},
		SchedulingResult: &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		}},
	}
}

func TestPendingSSDRestoreChangesNextDecision(t *testing.T) {
	p := calibratedPlugin(t)
	busy := candidate(p, &attrcache.MatchInfo{TotalTokens: 10000, ExternalTokens: 9999, BytesByMedium: map[string]uint64{"file": 1000000000}}, 0)
	idle := candidate(p, &attrcache.MatchInfo{TotalTokens: 10000}, 0)
	request := costRequest("slow-restore", busy, 10000)
	p.PreRequest(context.Background(), request, request.SchedulingResult)
	require.InDelta(t, 10.051, p.queuedSeconds(busy), 1e-9)
	// A different prompt is already on the busy engine's GPU. Its current
	// request cost is zero, but the earlier restore still occupies the engine.
	busy.Put(p.matchKey.String(), &attrcache.MatchInfo{TotalTokens: 10000, DeviceTokens: 10000})
	scores := p.Score(context.Background(), nil, []fwksched.Endpoint{busy, idle})
	require.Greater(t, scores[idle], scores[busy])
	p.ResponseBody(context.Background(), request, &requestcontrol.Response{StartOfStream: true}, nil)
	scores = p.Score(context.Background(), nil, []fwksched.Endpoint{busy, idle})
	require.Greater(t, scores[busy], scores[idle])
}

func TestFirstTokenEndAndEvictionReleaseOnlyTheirReservation(t *testing.T) {
	for _, first := range []string{"first-token", "end", "ttl-or-cancel"} {
		t.Run(first, func(t *testing.T) {
			p := calibratedPlugin(t)
			ep := candidate(p, &attrcache.MatchInfo{TotalTokens: 1000}, 0)
			a, b := costRequest("a", ep, 1000), costRequest("b", ep, 1000)
			p.PreRequest(context.Background(), a, a.SchedulingResult)
			p.PreRequest(context.Background(), a, a.SchedulingResult)
			p.PreRequest(context.Background(), b, b.SchedulingResult)
			require.Equal(t, 2.0, p.queuedSeconds(ep))
			key := costKey("default", ep.GetMetadata().NamespacedName.String())
			value, err := p.state.Read(a.RequestID, key)
			require.NoError(t, err)
			copy := value.Clone().(plugin.EvictableStateData)
			switch first {
			case "first-token":
				p.ResponseBody(context.Background(), a, &requestcontrol.Response{StartOfStream: true}, nil)
			case "end":
				p.ResponseBody(context.Background(), a, &requestcontrol.Response{EndOfStream: true}, nil)
			default:
				p.state.Delete(a.RequestID) // Same Delete/OnEvicted path used by PluginState's janitor.
			}
			copy.OnEvicted(a.RequestID, key)
			p.ResponseBody(context.Background(), a, &requestcontrol.Response{StartOfStream: true, EndOfStream: true}, nil)
			require.Equal(t, 1.0, p.queuedSeconds(ep))
			p.state.Delete(b.RequestID)
			require.Zero(t, p.queuedSeconds(ep))
		})
	}
}

func TestEndpointRecreationDoesNotReceiveOldRelease(t *testing.T) {
	p := calibratedPlugin(t)
	ep := candidate(p, &attrcache.MatchInfo{TotalTokens: 1000}, 0)
	old := fwkdl.NewEndpoint(ep.GetMetadata(), fwkdl.NewMetrics())
	require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: old}))
	a := costRequest("old", ep, 1000)
	p.PreRequest(context.Background(), a, a.SchedulingResult)
	require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: old}))
	late := costRequest("late-pre-request", ep, 1000)
	p.PreRequest(context.Background(), late, late.SchedulingResult)
	require.Zero(t, p.queuedSeconds(ep))
	newEndpoint := fwkdl.NewEndpoint(ep.GetMetadata(), fwkdl.NewMetrics())
	require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: newEndpoint}))
	b := costRequest("new", ep, 1000)
	p.PreRequest(context.Background(), b, b.SchedulingResult)
	// A delayed deletion and response from the removed instance are harmless.
	require.NoError(t, p.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: old}))
	p.ResponseBody(context.Background(), a, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Equal(t, 1.0, p.queuedSeconds(ep))
	p.ResponseBody(context.Background(), b, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Zero(t, p.queuedSeconds(ep))
}

func TestConcurrentResponsesAndReaping(t *testing.T) {
	p := calibratedPlugin(t)
	ep := candidate(p, &attrcache.MatchInfo{TotalTokens: 1000}, 0)
	var wg sync.WaitGroup
	for i := range 100 {
		request := costRequest(fmt.Sprintf("r%d", i), ep, 1000)
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.PreRequest(context.Background(), request, request.SchedulingResult)
			var release sync.WaitGroup
			release.Add(3)
			go func() { defer release.Done(); p.state.Delete(request.RequestID) }()
			go func() {
				defer release.Done()
				p.ResponseBody(context.Background(), request, &requestcontrol.Response{StartOfStream: true}, nil)
			}()
			go func() {
				defer release.Done()
				p.ResponseBody(context.Background(), request, &requestcontrol.Response{EndOfStream: true}, nil)
			}()
			p.Score(context.Background(), request, []fwksched.Endpoint{ep})
			release.Wait()
		}()
	}
	wg.Wait()
	require.InDelta(t, 0, p.queuedSeconds(ep), 1e-9)
}

func TestUnsupportedCacheIdentityTracksFullEstimatedPrefill(t *testing.T) {
	p := calibratedPlugin(t)
	ep := candidate(p, &attrcache.MatchInfo{}, 0)
	request := costRequest("unsupported", ep, 2500)
	request.Body.TokenizedPrompt.Estimated = true
	scores := p.Score(context.Background(), request, []fwksched.Endpoint{ep})
	require.InDelta(t, 1.0/3.5, scores[ep], 1e-9)
	p.PreRequest(context.Background(), request, request.SchedulingResult)
	require.Equal(t, 2.5, p.queuedSeconds(ep))
	p.ResponseBody(context.Background(), request, &requestcontrol.Response{EndOfStream: true}, nil)
	require.Zero(t, p.queuedSeconds(ep))
}

func TestQueuedReplayUsesItsCalibratedRate(t *testing.T) {
	p := calibratedPlugin(t)
	p.config.ReplayTokensPerSecond = 500
	ep := candidate(p, &attrcache.MatchInfo{TotalTokens: 10000, DeviceTokens: 5000,
		ExternalTokens: 4000, ReplayTokens: 1000,
		BytesByMedium: map[string]uint64{"memory": 1000000000, "file": 100000000}}, 0)
	request := costRequest("replay", ep, 10000)
	p.PreRequest(context.Background(), request, request.SchedulingResult)
	// One second cold prefill, two replay, two actual I/O, plus setup.
	require.InDelta(t, 5.05, p.queuedSeconds(ep), 1e-9)
	p.state.Delete(request.RequestID)
	require.Zero(t, p.queuedSeconds(ep))
}

type sourceRegistrar struct{ registration fwkdl.PendingRegistration }

func (r *sourceRegistrar) Register(registration fwkdl.PendingRegistration) error {
	r.registration = registration
	return nil
}

func TestEndpointNotificationsAreRegisteredAutomatically(t *testing.T) {
	p := calibratedPlugin(t)
	r := &sourceRegistrar{}
	require.NoError(t, p.RegisterDependencies(r))
	require.Equal(t, p.TypedName(), r.registration.Owner)
	require.Equal(t, "endpoint-notification-source", r.registration.SourceType)
	require.Same(t, p, r.registration.Extractor)
	require.NotNil(t, r.registration.DefaultSource)
}
