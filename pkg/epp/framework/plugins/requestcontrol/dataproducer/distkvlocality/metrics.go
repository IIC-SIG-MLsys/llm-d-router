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
	"sync"

	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

var routingRequests = prometheus.NewCounterVec(prometheus.CounterOpts{
	Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
	Name:      "distkv_routing_requests_total",
	Help:      "DistKV routing lookups and unsupported requests by result.",
}, []string{"plugin_name", "result"})

var registerRoutingOnce sync.Once

func registerRoutingMetrics() {
	registerRoutingOnce.Do(func() { metrics.Registry.MustRegister(routingRequests) })
}
