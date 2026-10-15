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
	"testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrcache "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/cachecost"
	testutils "github.com/llm-d/llm-d-router/test/utils"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
)

func calibratedPlugin(t *testing.T) *Plugin {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	p, err := Factory("cost", plugin.StrictDecoder([]byte(`{
		"prefillTokensPerSecond":1000,"memoryBytesPerSecond":1000000000,
		"fileBytesPerSecond":100000000,"restoreOverheadMs":50
	}`)), testutils.NewTestHandle(ctx))
	require.NoError(t, err)
	return p.(*Plugin)
}

func candidate(p *Plugin, match *attrcache.MatchInfo, queued float64) fwksched.Endpoint {
	id := fmt.Sprintf("candidate-%d", len(p.loads))
	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "default", Name: id}}, fwkdl.NewMetrics(), nil)
	if match != nil {
		ep.Put(p.matchKey.String(), match)
	}
	p.loads[ep.GetMetadata().NamespacedName.String()] = &endpointLoad{seconds: queued}
	return ep
}

func TestRestoreCostAndLoadChangeEndpointSelection(t *testing.T) {
	p := calibratedPlugin(t)
	memory := &attrcache.MatchInfo{TotalTokens: 10000, ExternalTokens: 9000, ReplayTokens: 1000, BytesByMedium: map[string]uint64{"memory": 900000000}}
	file := &attrcache.MatchInfo{TotalTokens: 10000, ExternalTokens: 9000, ReplayTokens: 1000, BytesByMedium: map[string]uint64{"file": 900000000}}
	device := &attrcache.MatchInfo{TotalTokens: 10000, DeviceTokens: 10000}
	cold := &attrcache.MatchInfo{TotalTokens: 10000}
	for _, tc := range []struct {
		name                   string
		fast, slow             *attrcache.MatchInfo
		fastQueued, slowQueued float64
	}{
		{"device avoids restoration", device, memory, 0, 0},
		{"same prefix in memory beats file", memory, file, 0, 0},
		{"idle recompute beats slow SSD", cold, file, 0, 0},
		{"load outweighs device hit", cold, device, 0, 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fast := candidate(p, tc.fast, tc.fastQueued)
			slow := candidate(p, tc.slow, tc.slowQueued)
			scores := p.Score(context.Background(), nil, []fwksched.Endpoint{fast, slow})
			require.Greater(t, scores[fast], scores[slow])
		})
	}
}

func TestReplayAndActualTransferredBytes(t *testing.T) {
	p := calibratedPlugin(t)
	match := &attrcache.MatchInfo{TotalTokens: 10000, DeviceTokens: 5000, ExternalTokens: 4000, ReplayTokens: 1000,
		BytesByMedium: map[string]uint64{"memory": 1000000000, "file": 100000000}}
	// One second of remaining prefill, one of replay, two of I/O and setup.
	require.InDelta(t, 4.05, p.requestSeconds(match), 1e-9)
	copy := match.Clone().(*attrcache.MatchInfo)
	copy.BytesByMedium["memory"] = 2000000000
	require.InDelta(t, 5.05, p.requestSeconds(copy), 1e-9)
	require.Equal(t, uint64(1000000000), match.BytesByMedium["memory"])
}

func TestMissingOrUnsupportedObservationDoesNotWin(t *testing.T) {
	p := calibratedPlugin(t)
	cold := candidate(p, &attrcache.MatchInfo{TotalTokens: 1000}, 0)
	for _, match := range []*attrcache.MatchInfo{
		nil,
		{TotalTokens: 1000, DeviceTokens: 2000},
		{TotalTokens: 1000, ExternalTokens: 1000},
		{TotalTokens: 1000, ExternalTokens: 1000, BytesByMedium: map[string]uint64{"remote": 10}},
	} {
		other := candidate(p, match, 0)
		scores := p.Score(context.Background(), nil, []fwksched.Endpoint{cold, other})
		require.Greater(t, scores[cold], scores[other])
	}
}

func TestRatesRequireDeploymentCalibration(t *testing.T) {
	for _, input := range []string{
		`{}`,
		`{"prefillTokensPerSecond":1000,"memoryBytesPerSecond":-1,"fileBytesPerSecond":1}`,
		`{"prefillTokensPerSecond":1000,"memoryBytesPerSecond":1,"fileBytesPerSecond":1,"restoreOverheadMs":-1}`,
	} {
		_, err := Factory("cost", plugin.StrictDecoder([]byte(input)), nil)
		require.Error(t, err)
	}
	p := calibratedPlugin(t)
	require.Contains(t, p.Consumes().Required, p.matchKey)
	require.Len(t, p.Consumes().Required, 1)
}

func TestRemoteTransferCostCanFavorLocalHitOrRecompute(t *testing.T) {
	p := calibratedPlugin(t)
	p.config.RemoteBytesPerSecond = map[string]float64{"hux": 1000000000}
	p.config.RemoteOverheadMs = 20
	local := &attrcache.MatchInfo{TotalTokens: 10000, ExternalTokens: 9000, BytesByMedium: map[string]uint64{"memory": 1000000000}}
	remote := local.Clone().(*attrcache.MatchInfo)
	remote.RemoteBytesByTransport = map[string]uint64{"hux": 1000000000}
	remote.RemoteSources = 1
	require.InDelta(t, p.requestSeconds(local)+1.02, p.requestSeconds(remote), 1e-9)
	cold := &attrcache.MatchInfo{TotalTokens: 10000}
	require.Less(t, p.requestSeconds(remote), p.requestSeconds(cold))
	p.config.RemoteBytesPerSecond["hux"] = 100000000
	require.Greater(t, p.requestSeconds(remote), p.requestSeconds(cold))
	delete(p.config.RemoteBytesPerSecond, "hux")
	require.True(t, math.IsInf(p.requestSeconds(remote), 1))
	clone := remote.Clone().(*attrcache.MatchInfo)
	clone.RemoteBytesByTransport["hux"] = 1
	require.EqualValues(t, 1000000000, remote.RemoteBytesByTransport["hux"])
}
