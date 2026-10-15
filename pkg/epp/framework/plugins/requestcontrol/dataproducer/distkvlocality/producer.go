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

// Package distkvlocality combines device prefixes with verified external cache observations.
package distkvlocality

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/cachecost"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// PluginType is the registered type name of the distkv-locality-producer.
const PluginType = "distkv-locality-producer"

const (
	defaultLookupTimeout  = 500 * time.Millisecond
	defaultLookupCacheMax = 4096
)

// PluginConfig configures the distkv-locality-producer.
type PluginConfig struct {
	// ControllerAddr is the DistKV CacheServerController gRPC address (required).
	ControllerAddr           string `json:"controllerAddr"`
	RoutingMode              string `json:"routingMode,omitempty"`
	DevicePrefixProducerName string `json:"devicePrefixProducerName,omitempty"`
	// Model/Pool is the legacy single-pool form; equivalent to
	// Pools=[{Model, Pool}]. Kept for backward compatibility.
	Model string `json:"model"`
	// Pool is the DistKV pool name (legacy single-pool form).
	Pool string `json:"pool"`
	// Pools limits vLLM observations to these registered engine pools.
	Pools []PoolConfig `json:"pools,omitempty"`
	// GroupID is unsupported by instance routing and must be empty.
	GroupID string `json:"groupID,omitempty"`
	// ChunkSize must be zero; the engine layout supplies the chunk size.
	ChunkSize int `json:"chunkSize,omitempty"`
	// RoutingInfoTTL is retained for configuration compatibility; engine leases govern layouts.
	RoutingInfoTTL string `json:"routingInfoTTL,omitempty"`
	// LookupTimeout bounds each DistKV RPC (Go duration string; default 500ms).
	LookupTimeout string `json:"lookupTimeout,omitempty"`
	// LookupCacheTTL bounds cached routing observations (Go duration string; default 0).
	LookupCacheTTL string `json:"lookupCacheTTL,omitempty"`
}

// PoolConfig identifies one allowed vLLM pool.
type PoolConfig struct {
	// Model is retained metadata; the request must match the registered HTTP model.
	Model string `json:"model"`
	// Pool is the DistKV pool name (required).
	Pool string `json:"pool"`
}

var _ requestcontrol.DataProducer = &Producer{}

// Producer publishes combined cache observations for every candidate endpoint.
type Producer struct {
	typedName plugin.TypedName
	dk        plugin.DataKey
	costKey   plugin.DataKey

	client        *Client
	pools         []PoolConfig
	lookupTimeout time.Duration

	mu                   sync.Mutex
	lookupTTL            time.Duration
	lookupMax            int
	routingMode          string
	routingModel         string
	devicePrefixProducer string
	routingLookups       map[string]cachedRoutingLookup
}

// PluginFactory parses the raw plugin configuration and returns a configured
// Producer.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	var params PluginConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", PluginType, err)
		}
	}
	if params.ControllerAddr == "" {
		return nil, errors.New("controllerAddr is required")
	}
	pools := params.Pools
	if params.RoutingMode == "sglang" {
		if params.Model == "" {
			return nil, errors.New("model is required for SGLang routing")
		}
		return New(handle.Context(), name, params)
	}
	if len(pools) == 0 {
		// Legacy single-pool form: model+pool.
		if params.Model == "" || params.Pool == "" {
			return nil, errors.New("either model+pool or pools is required")
		}
		pools = []PoolConfig{{Model: params.Model, Pool: params.Pool}}
	}
	for _, pc := range pools {
		if pc.Pool == "" {
			return nil, errors.New("each pool entry requires a pool name")
		}
	}
	params.Pools = pools
	return New(handle.Context(), name, params)
}

// New constructs a name-bound cache observation producer.
func New(ctx context.Context, name string, config PluginConfig) (*Producer, error) {
	if config.RoutingMode != "" && config.RoutingMode != "vllm" && config.RoutingMode != "sglang" {
		return nil, fmt.Errorf("invalid routingMode %q", config.RoutingMode)
	}
	registerRoutingMetrics()
	client, err := newClient(config.ControllerAddr)
	if err != nil {
		return nil, err
	}
	pools := config.Pools
	if len(pools) == 0 && config.Pool != "" {
		pools = []PoolConfig{{Model: config.Model, Pool: config.Pool}}
	}
	if config.GroupID != "" || config.ChunkSize != 0 {
		_ = client.Close()
		return nil, errors.New("instance routing requires empty groupID and zero chunkSize")
	}
	timeout := defaultLookupTimeout
	if config.LookupTimeout != "" {
		parsed, err := time.ParseDuration(config.LookupTimeout)
		if err != nil {
			_ = client.Close()
			return nil, fmt.Errorf("invalid lookupTimeout %q: %w", config.LookupTimeout, err)
		}
		if parsed <= 0 {
			_ = client.Close()
			return nil, errors.New("lookupTimeout must be positive")
		}
		timeout = parsed
	}
	_ = ctx // producer lifetime is process-scoped; no background goroutines yet
	lookupTTL := time.Duration(0)
	if config.LookupCacheTTL != "" {
		parsed, err := time.ParseDuration(config.LookupCacheTTL)
		if err != nil {
			_ = client.Close()
			return nil, fmt.Errorf("invalid lookupCacheTTL %q: %w", config.LookupCacheTTL, err)
		}
		if parsed < 0 {
			_ = client.Close()
			return nil, errors.New("lookupCacheTTL must be nonnegative")
		}
		lookupTTL = parsed
	}
	return &Producer{
		typedName:            plugin.TypedName{Type: PluginType, Name: name},
		dk:                   attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		costKey:              cachecost.MatchInfoDataKey.WithNonEmptyProducerName(name),
		client:               client,
		pools:                pools,
		lookupTimeout:        timeout,
		lookupTTL:            lookupTTL,
		lookupMax:            defaultLookupCacheMax,
		routingMode:          config.RoutingMode,
		routingModel:         config.Model,
		devicePrefixProducer: config.DevicePrefixProducerName,
		routingLookups:       make(map[string]cachedRoutingLookup),
	}, nil
}

// TypedName returns the plugin's registered type and name.
func (p *Producer) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces declares the PrefixCacheMatchInfoDataKey published per endpoint,
// name-bound to this producer instance.
func (p *Producer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: attrprefix.PrefixCacheMatchInfo{}, p.costKey: cachecost.MatchInfo{}}
}

// Consumes declares the TokenizedPrompt dependency from token-producer so
// the data-layer DAG orders tokenization before this producer runs.
func (p *Producer) Consumes() plugin.DataDependencies {
	dependencies := plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			tokenproducer.TokenizedPromptDataKey: fwksched.TokenizedPrompt{},
		},
	}
	if p.devicePrefixProducer != "" {
		dependencies.Required[attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(p.devicePrefixProducer)] = attrprefix.PrefixCacheMatchInfo{}
	}
	return dependencies
}

// ProduceTimeout bounds the data-layer execution of Produce.
func (p *Producer) ProduceTimeout() time.Duration {
	return p.lookupTimeout * 2
}

// Produce combines each endpoint's device prefix with its verified external increment.
func (p *Producer) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	return p.produceRouting(ctx, request, endpoints)
}

func (p *Producer) writeMatchInfo(endpoints []fwksched.Endpoint, match, total, blockSize int) {
	for _, ep := range endpoints {
		ep.Put(p.dk.String(), attrprefix.NewPrefixCacheMatchInfo(match, total, blockSize))
		ep.Put(p.costKey.String(), &cachecost.MatchInfo{TotalTokens: uint32(total * blockSize)})
	}
}
