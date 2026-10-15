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
	"math"
	"sync"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrcache "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/cachecost"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

var (
	_ requestcontrol.PreRequest            = (*Plugin)(nil)
	_ requestcontrol.ResponseBodyProcessor = (*Plugin)(nil)
	_ fwkdl.Registrant                     = (*Plugin)(nil)
	_ fwkdl.EndpointExtractor              = (*Plugin)(nil)
)

type endpointLoad struct {
	endpoint fwkdl.Endpoint
	mu       sync.Mutex
	seconds  float64
}

func (l *endpointLoad) add(seconds float64) {
	l.mu.Lock()
	l.seconds = max(0, l.seconds+seconds)
	l.mu.Unlock()
}

func (l *endpointLoad) get() float64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.seconds
}

type costReservation struct {
	load    *endpointLoad
	seconds float64
	once    sync.Once
}

// Clones refer to the same reservation; eviction must release it only once.
type costEntry struct{ reservation *costReservation }

func (e *costEntry) Clone() plugin.StateData { return &costEntry{reservation: e.reservation} }
func (e *costEntry) OnEvicted(_ string, _ plugin.StateKey) {
	e.reservation.once.Do(func() { e.reservation.load.add(-e.reservation.seconds) })
}

func (p *Plugin) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner: p.TypedName(), SourceType: sourcenotifications.EndpointNotificationSourceType, Extractor: p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

func (p *Plugin) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	id := event.Endpoint.GetMetadata().NamespacedName.String()
	p.mu.Lock()
	defer p.mu.Unlock()
	load := p.loads[id]
	switch event.Type {
	case fwkdl.EventDelete:
		if load != nil && (load.endpoint == nil || load.endpoint == event.Endpoint) {
			delete(p.loads, id)
		}
	case fwkdl.EventAddOrUpdate:
		if load == nil {
			load = &endpointLoad{}
			p.loads[id] = load
		}
		load.endpoint = event.Endpoint
	}
	return nil
}

func (p *Plugin) queuedSeconds(endpoint fwksched.Endpoint) float64 {
	if endpoint.GetMetadata() == nil {
		return 0
	}
	p.mu.Lock()
	load := p.loads[endpoint.GetMetadata().NamespacedName.String()]
	p.mu.Unlock()
	if load == nil {
		return 0
	}
	return load.get()
}

func (p *Plugin) estimateSeconds(request *fwksched.InferenceRequest, match *attrcache.MatchInfo) float64 {
	if match != nil && match.TotalTokens > 0 {
		return p.requestSeconds(match)
	}
	// Unsupported cache identities still contribute their full estimated prefill.
	if request != nil && request.Body != nil && request.Body.TokenizedPrompt != nil {
		return float64(request.Body.TokenizedPrompt.TokenCount()) / p.config.PrefillTokensPerSecond
	}
	return 0
}

func costKey(profile, endpoint string) plugin.StateKey {
	return plugin.StateKey(fmt.Sprintf("cost/%s/%s", profile, endpoint))
}

func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) {
	if request == nil || request.RequestID == "" || result == nil {
		return
	}
	// Serialize hook writes so repeated PreRequest calls cannot overwrite
	// an owned reservation without releasing it.
	p.mu.Lock()
	defer p.mu.Unlock()
	for profile, selected := range result.ProfileResults {
		if selected == nil || len(selected.TargetEndpoints) == 0 {
			continue
		}
		endpoint := selected.TargetEndpoints[0]
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		id := endpoint.GetMetadata().NamespacedName.String()
		key := costKey(profile, id)
		if _, err := p.state.Read(request.RequestID, key); err == nil {
			continue
		}
		value, _ := endpoint.Get(p.matchKey.String())
		match, _ := value.(*attrcache.MatchInfo)
		seconds := p.estimateSeconds(request, match)
		if math.IsInf(seconds, 0) || math.IsNaN(seconds) {
			continue
		}
		load := p.loads[id]
		if load == nil {
			// A late request hook must not recreate a deleted endpoint's queue.
			continue
		}
		load.add(seconds)
		p.state.Write(request.RequestID, key, &costEntry{reservation: &costReservation{load: load, seconds: seconds}})
	}
}

func (p *Plugin) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *requestcontrol.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || request.RequestID == "" || response == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if response.StartOfStream || response.EndOfStream {
		p.state.Delete(request.RequestID)
	} else {
		p.state.Touch(request.RequestID)
	}
}
