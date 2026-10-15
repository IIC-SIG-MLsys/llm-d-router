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
	"encoding/json"
	"fmt"
	"math"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrcache "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/cachecost"
)

const PluginType = "distkv-cache-cost-scorer"

// Rates are deployment calibrations, not hardware peak bandwidths.
type Config struct {
	PrefillTokensPerSecond float64            `json:"prefillTokensPerSecond"`
	ReplayTokensPerSecond  float64            `json:"replayTokensPerSecond,omitempty"`
	MemoryBytesPerSecond   float64            `json:"memoryBytesPerSecond"`
	FileBytesPerSecond     float64            `json:"fileBytesPerSecond"`
	RestoreOverheadMs      float64            `json:"restoreOverheadMs,omitempty"`
	RemoteBytesPerSecond   map[string]float64 `json:"remoteBytesPerSecond,omitempty"`
	RemoteOverheadMs       float64            `json:"remoteOverheadMs,omitempty"`
	MatchInfoProducerName  string             `json:"matchInfoProducerName,omitempty"`
}

type Plugin struct {
	typedName plugin.TypedName
	config    Config
	matchKey  plugin.DataKey
	state     *plugin.PluginState
	mu        sync.Mutex
	loads     map[string]*endpointLoad
}

var _ fwksched.Scorer = (*Plugin)(nil)

func Factory(name string, decoder *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	var config Config
	if decoder != nil {
		if err := decoder.Decode(&config); err != nil {
			return nil, fmt.Errorf("decode cache cost parameters: %w", err)
		}
	}
	if config.ReplayTokensPerSecond == 0 {
		config.ReplayTokensPerSecond = config.PrefillTokensPerSecond
	}
	for name, rate := range map[string]float64{
		"prefillTokensPerSecond": config.PrefillTokensPerSecond,
		"replayTokensPerSecond":  config.ReplayTokensPerSecond,
		"memoryBytesPerSecond":   config.MemoryBytesPerSecond,
		"fileBytesPerSecond":     config.FileBytesPerSecond,
	} {
		if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
			return nil, fmt.Errorf("%s must be a finite positive rate", name)
		}
	}
	if config.RestoreOverheadMs < 0 || math.IsNaN(config.RestoreOverheadMs) || math.IsInf(config.RestoreOverheadMs, 0) {
		return nil, fmt.Errorf("restoreOverheadMs must be finite and nonnegative")
	}
	for transport, rate := range config.RemoteBytesPerSecond {
		if transport == "" || rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
			return nil, fmt.Errorf("remoteBytesPerSecond[%s] must be a finite positive rate", transport)
		}
	}
	if config.RemoteOverheadMs < 0 || math.IsNaN(config.RemoteOverheadMs) || math.IsInf(config.RemoteOverheadMs, 0) {
		return nil, fmt.Errorf("remoteOverheadMs must be finite and nonnegative")
	}
	if handle == nil || handle.Context() == nil {
		return nil, fmt.Errorf("cache cost scorer requires a plugin context")
	}
	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name}, config: config,
		matchKey: attrcache.MatchInfoDataKey.WithNonEmptyProducerName(config.MatchInfoProducerName),
		state:    plugin.NewPluginState(handle.Context()), loads: make(map[string]*endpointLoad),
	}, nil
}

func (p *Plugin) TypedName() plugin.TypedName       { return p.typedName }
func (p *Plugin) Category() fwksched.ScorerCategory { return fwksched.Balance }

func (p *Plugin) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{Required: map[plugin.DataKey]any{
		p.matchKey: attrcache.MatchInfo{},
	}}
}

func (p *Plugin) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = 0
		value, _ := endpoint.Get(p.matchKey.String())
		match, ok := value.(*attrcache.MatchInfo)
		if !ok || match == nil {
			continue
		}
		seconds := p.queuedSeconds(endpoint) + p.estimateSeconds(request, match)
		scores[endpoint] = 1 / (1 + seconds)
	}
	return scores
}

func (p *Plugin) requestSeconds(match *attrcache.MatchInfo) float64 {
	covered := uint64(match.DeviceTokens) + uint64(match.ExternalTokens)
	if covered > uint64(match.TotalTokens) || match.ReplayTokens > match.ExternalTokens {
		return math.Inf(1)
	}
	seconds := float64(uint64(match.TotalTokens)-covered) / p.config.PrefillTokensPerSecond
	seconds += float64(match.ReplayTokens) / p.config.ReplayTokensPerSecond
	if match.ExternalTokens == 0 {
		return seconds
	}
	seconds += p.config.RestoreOverheadMs / 1000
	var bytes uint64
	for medium, count := range match.BytesByMedium {
		bytes += count
		switch medium {
		case "memory":
			seconds += float64(count) / p.config.MemoryBytesPerSecond
		case "file":
			seconds += float64(count) / p.config.FileBytesPerSecond
		default:
			return math.Inf(1)
		}
	}
	if bytes == 0 {
		return math.Inf(1)
	}
	for transport, count := range match.RemoteBytesByTransport {
		rate := p.config.RemoteBytesPerSecond[transport]
		if rate <= 0 {
			return math.Inf(1)
		}
		seconds += float64(count) / rate
	}
	seconds += float64(match.RemoteSources) * p.config.RemoteOverheadMs / 1000
	return seconds
}
