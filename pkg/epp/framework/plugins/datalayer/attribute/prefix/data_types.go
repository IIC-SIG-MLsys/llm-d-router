/*
Copyright 2025 The Kubernetes Authors.

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

package prefix

import (
	"slices"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	approxprefixconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/constants"
)

var PrefixCacheMatchInfoDataKey = plugin.NewDataKey("PrefixCacheMatchInfoDataKey", approxprefixconstants.ApproxPrefixCachePluginType)

type DeviceTokenRange struct{ Start, End uint32 }

type DeviceCacheGroup struct {
	GroupIdx uint32
	Ranges   []DeviceTokenRange
}

type PrefixCacheMatchInfo struct {
	// matched prefix length in blocks. For the precise prefix cache this is the
	// device-tier-weighted longest-prefix score (e.g. RAM-tier blocks count as
	// less than 1.0), suitable for relative endpoint ranking.
	matchBlocks int
	// total length in blocks
	totalBlocks int
	// block length in tokens
	blockSizeTokens int
	// unweighted count of contiguous cached prefix blocks on the endpoint.
	// Unlike matchBlocks this is the literal number of cached blocks regardless
	// of device tier, so consumers that convert blocks to a token count (e.g.
	// the prefix-based PD decider) get an accurate cached-token figure rather
	// than a tier-attenuated one. Defaults to matchBlocks when not set.
	cachedBlockCount       int
	deviceCachedBlockCount int
	groupedDeviceCache     bool
	deviceCacheGroups      []DeviceCacheGroup
}

func NewPrefixCacheMatchInfo(matchBlocks int, totalBlocks int, blockSizeTokens int) *PrefixCacheMatchInfo {
	return &PrefixCacheMatchInfo{
		matchBlocks:      matchBlocks,
		totalBlocks:      totalBlocks,
		blockSizeTokens:  blockSizeTokens,
		cachedBlockCount: matchBlocks,
	}
}

// WithCachedBlockCount sets the unweighted contiguous cached-block count and
// returns the receiver for chaining.
func (p *PrefixCacheMatchInfo) WithCachedBlockCount(cachedBlockCount int) *PrefixCacheMatchInfo {
	p.cachedBlockCount = cachedBlockCount
	return p
}

func (p *PrefixCacheMatchInfo) WithDeviceCachedBlockCount(count int) *PrefixCacheMatchInfo {
	p.deviceCachedBlockCount = count
	return p
}

func (p *PrefixCacheMatchInfo) DeviceCachedBlockCount() int { return p.deviceCachedBlockCount }

func (p *PrefixCacheMatchInfo) WithDeviceCacheGroups(groups []DeviceCacheGroup, grouped bool) *PrefixCacheMatchInfo {
	p.deviceCacheGroups, p.groupedDeviceCache = groups, grouped
	return p
}

func (p *PrefixCacheMatchInfo) DeviceCacheGroups() []DeviceCacheGroup { return p.deviceCacheGroups }
func (p *PrefixCacheMatchInfo) GroupedDeviceCache() bool              { return p.groupedDeviceCache }

func (p *PrefixCacheMatchInfo) MatchBlocks() int {
	return p.matchBlocks
}

func (p *PrefixCacheMatchInfo) TotalBlocks() int {
	return p.totalBlocks
}

func (p *PrefixCacheMatchInfo) BlockSizeTokens() int {
	return p.blockSizeTokens
}

// CachedBlockCount returns the unweighted count of contiguous cached prefix
// blocks on the endpoint.
func (p *PrefixCacheMatchInfo) CachedBlockCount() int {
	return p.cachedBlockCount
}

func (p *PrefixCacheMatchInfo) Clone() fwkdl.Cloneable {
	groups := slices.Clone(p.deviceCacheGroups)
	for i := range groups {
		groups[i].Ranges = slices.Clone(groups[i].Ranges)
	}
	return &PrefixCacheMatchInfo{
		matchBlocks:            p.matchBlocks,
		totalBlocks:            p.totalBlocks,
		blockSizeTokens:        p.blockSizeTokens,
		cachedBlockCount:       p.cachedBlockCount,
		deviceCachedBlockCount: p.deviceCachedBlockCount,
		groupedDeviceCache:     p.groupedDeviceCache,
		deviceCacheGroups:      groups,
	}
}
