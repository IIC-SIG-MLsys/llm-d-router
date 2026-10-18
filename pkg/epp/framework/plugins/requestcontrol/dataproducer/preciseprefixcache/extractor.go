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

package preciseprefixcache

import (
	"context"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

var _ fwkdl.EndpointExtractor = &Producer{}

// Extract processes endpoint lifecycle events emitted by the
// endpoint-notification-source: add/update installs a per-pod ZMQ KV-events
// subscriber, delete tears one down. No-op unless per-pod discovery is
// enabled.
func (p *Producer) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	if !p.kvEventsConfig.DiscoverPods || p.kvEventsConfig.PodDiscoveryConfig == nil {
		return nil
	}
	meta := event.Endpoint.GetMetadata()
	if meta == nil || meta.NamespacedName.Name == "" {
		return nil
	}

	logger := log.FromContext(ctx).WithName(p.typedName.String())
	endpointKey := meta.NamespacedName.String()

	switch event.Type {
	case fwkdl.EventAddOrUpdate:
		if err := p.ensureSubscriber(ctx, meta); err != nil {
			return err
		}
		logger.V(logging.DEBUG).Info("Adding subscriber", "endpoint", endpointKey)
	case fwkdl.EventDelete:
		p.sources.remove(meta)
		logger.V(logging.DEBUG).Info("Removed KV-events subscriber", "endpoint", endpointKey)
	}
	return nil
}

// ensureSubscriber idempotently installs a KV-events subscriber for the given
// endpoint, dialing SocketPort + RankIndex to match standard inference-engine port offsetting
// (one ZMQ PUB socket per DP rank on the same pod IP).
func (p *Producer) ensureSubscriber(ctx context.Context, meta *fwkdl.EndpointMetadata) error {
	if meta == nil || meta.Address == "" {
		return nil
	}
	if p.sources == nil {
		return nil
	}
	port := p.kvEventsConfig.PodDiscoveryConfig.SocketPort + meta.GetRankIndex()
	if err := p.sources.ensure(meta, port); err != nil {
		return fmt.Errorf("ensure subscriber for %s: %w", meta.NamespacedName.String(), err)
	}
	return nil
}
