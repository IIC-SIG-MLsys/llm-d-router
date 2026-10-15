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
	"fmt"
	"math"
	"slices"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// extractEndpointSet builds the "address:port" identifier set used to filter
// kvblock.Index lookups to candidate endpoints. Endpoints without metadata
// are skipped.
func extractEndpointSet(endpoints []scheduling.Endpoint) sets.Set[string] {
	endpointSet := sets.New[string]()
	for _, ep := range endpoints {
		if m := ep.GetMetadata(); m != nil {
			endpointSet.Insert(fmt.Sprintf("%s:%s", m.Address, m.Port))
		}
	}
	return endpointSet
}

// matchedBlockCount returns the number of contiguous cached prefix blocks held
// by podID, counting from the first block until the first block the pod does
// not hold. This is the unweighted counterpart of the device-tier-weighted
// kvblock scorer: every cached block counts as one regardless of device tier,
// so a pod present at keys[0..n-1] yields n.
func matchedBlockCount(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) int {
	count := 0
	for _, key := range keys {
		if !slices.ContainsFunc(keyToPods[key], func(e kvblock.PodEntry) bool { return e.PodIdentifier == podID }) {
			break
		}
		count++
	}
	return count
}

func deviceMatchedBlockCount(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string) int {
	for _, key := range keys {
		if slices.ContainsFunc(keyToPods[key], func(e kvblock.PodEntry) bool { return e.PodIdentifier == podID && e.HasGroup }) {
			return 0
		}
	}
	count := 0
	for _, key := range keys {
		if !slices.ContainsFunc(keyToPods[key], func(e kvblock.PodEntry) bool {
			return e.PodIdentifier == podID && e.DeviceTier == "gpu" && !e.Speculative
		}) {
			break
		}
		count++
	}
	return count
}

// Group observations are evidence, not a declaration of the required group set.
func deviceCacheGroups(keys []kvblock.BlockHash, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, podID string, blockSize int) ([]attrprefix.DeviceCacheGroup, bool) {
	groups := make(map[uint32][]attrprefix.DeviceTokenRange)
	grouped := false
	seen := make(map[uint32]bool)
	for i, key := range keys {
		clear(seen)
		for _, entry := range keyToPods[key] {
			if entry.PodIdentifier != podID || !entry.HasGroup {
				continue
			}
			grouped = true
			if entry.GroupIdx < 0 || entry.GroupIdx > math.MaxInt32 {
				continue
			}
			g := uint32(entry.GroupIdx)
			if _, exists := groups[g]; !exists {
				groups[g] = nil
			}
			if entry.DeviceTier != "gpu" || entry.Speculative || seen[g] {
				continue
			}
			seen[g] = true
			start, end := uint32(i*blockSize), uint32((i+1)*blockSize)
			ranges := groups[g]
			if len(ranges) > 0 && ranges[len(ranges)-1].End == start {
				ranges[len(ranges)-1].End = end
			} else {
				ranges = append(ranges, attrprefix.DeviceTokenRange{Start: start, End: end})
			}
			groups[g] = ranges
		}
	}
	result := make([]attrprefix.DeviceCacheGroup, 0, len(groups))
	for g, ranges := range groups {
		result = append(result, attrprefix.DeviceCacheGroup{GroupIdx: g, Ranges: ranges})
	}
	slices.SortFunc(result, func(a, b attrprefix.DeviceCacheGroup) int { return int(a.GroupIdx) - int(b.GroupIdx) })
	return result, grouped
}
