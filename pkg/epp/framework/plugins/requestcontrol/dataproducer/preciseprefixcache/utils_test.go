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
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/stretchr/testify/assert"
)

func TestMatchedBlockCount(t *testing.T) {
	const (
		podA = "10.0.0.1:8000"
		podB = "10.0.0.2:8000"
	)
	keys := []kvblock.BlockHash{1, 2, 3, 4}

	// gpu/cpu tiers must count identically — the unweighted count ignores tier.
	gpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu"} }
	cpu := func(pod string) kvblock.PodEntry { return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "cpu"} }

	tests := []struct {
		name      string
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry
		podID     string
		want      int
	}{
		{
			name: "all blocks held on RAM/cpu tier count fully (unweighted)",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {cpu(podA)}, 2: {cpu(podA)}, 3: {cpu(podA)}, 4: {cpu(podA)},
			},
			podID: podA,
			want:  4,
		},
		{
			name: "single RAM block counts as one, not zero",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {cpu(podA)},
			},
			podID: podA,
			want:  1,
		},
		{
			name: "stops at first missing block",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA)}, 2: {gpu(podA)}, 4: {gpu(podA)}, // block 3 missing
			},
			podID: podA,
			want:  2,
		},
		{
			name: "pod absent from first block yields zero",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podB)}, 2: {gpu(podA)},
			},
			podID: podA,
			want:  0,
		},
		{
			name: "counts are per-pod independent",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {gpu(podA), cpu(podB)}, 2: {gpu(podA)}, 3: {cpu(podB)},
			},
			podID: podA,
			want:  2,
		},
		{
			name:      "empty index yields zero",
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{},
			podID:     podA,
			want:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, matchedBlockCount(keys, tt.keyToPods, tt.podID))
		})
	}
}

func TestDeviceMatchedBlockCountExcludesHostAndSpeculativeEntries(t *testing.T) {
	keys := []kvblock.BlockHash{1, 2, 3}
	for _, tier := range []string{"cpu_pinned", "disk", "unknown"} {
		entries := map[kvblock.BlockHash][]kvblock.PodEntry{
			1: {{PodIdentifier: "pod", DeviceTier: "gpu"}},
			2: {{PodIdentifier: "pod", DeviceTier: tier}},
			3: {{PodIdentifier: "pod", DeviceTier: "gpu"}},
		}
		if got := deviceMatchedBlockCount(keys, entries, "pod"); got != 1 {
			t.Fatalf("tier %s device count=%d, want 1", tier, got)
		}
	}
	entries := map[kvblock.BlockHash][]kvblock.PodEntry{1: {{PodIdentifier: "pod", DeviceTier: "gpu", Speculative: true}}}
	if got := deviceMatchedBlockCount(keys, entries, "pod"); got != 0 {
		t.Fatalf("speculative device count=%d", got)
	}
}

func TestGroupedDeviceEvidenceRequiresRegisteredSemantics(t *testing.T) {
	keys := []kvblock.BlockHash{1, 2}
	entries := map[kvblock.BlockHash][]kvblock.PodEntry{
		1: {{PodIdentifier: "pod", DeviceTier: "gpu", HasGroup: true, GroupIdx: 0}, {PodIdentifier: "pod", DeviceTier: "gpu", HasGroup: true, GroupIdx: 1}},
		2: {{PodIdentifier: "pod", DeviceTier: "gpu", HasGroup: true, GroupIdx: 0}, {PodIdentifier: "pod", DeviceTier: "gpu", HasGroup: true, GroupIdx: 1, Speculative: true}, {PodIdentifier: "other", DeviceTier: "gpu", HasGroup: true, GroupIdx: 1}},
	}
	assert.Zero(t, deviceMatchedBlockCount(keys, entries, "pod"))
	groups, grouped := deviceCacheGroups(keys, entries, "pod", 2)
	assert.True(t, grouped)
	assert.Equal(t, []attrprefix.DeviceCacheGroup{
		{GroupIdx: 0, Ranges: []attrprefix.DeviceTokenRange{{Start: 0, End: 4}}},
		{GroupIdx: 1, Ranges: []attrprefix.DeviceTokenRange{{Start: 0, End: 2}}},
	}, groups)
	info := attrprefix.NewPrefixCacheMatchInfo(0, 2, 2).WithDeviceCacheGroups(groups, true)
	clone := info.Clone().(*attrprefix.PrefixCacheMatchInfo)
	clone.DeviceCacheGroups()[0].Ranges[0].End = 2
	assert.EqualValues(t, 4, info.DeviceCacheGroups()[0].Ranges[0].End)
	next, grouped := deviceCacheGroups([]kvblock.BlockHash{3, 4}, entries, "pod", 2)
	assert.False(t, grouped)
	assert.Empty(t, next)
}
