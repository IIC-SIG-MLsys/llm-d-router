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
	"maps"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var MatchInfoDataKey = plugin.NewDataKey("DistKVCacheMatchInfoDataKey", "distkv-locality-producer")

// Bytes describe actual reads, including shards and any device-prefix overlap.
type MatchInfo struct {
	TotalTokens            uint32
	DeviceTokens           uint32
	ExternalTokens         uint32
	ReplayTokens           uint32
	BytesByMedium          map[string]uint64
	RemoteBytesByTransport map[string]uint64
	RemoteSources          uint32
}

func (m *MatchInfo) Clone() fwkdl.Cloneable {
	if m == nil {
		return nil
	}
	copy := *m
	copy.BytesByMedium = maps.Clone(m.BytesByMedium)
	copy.RemoteBytesByTransport = maps.Clone(m.RemoteBytesByTransport)
	return &copy
}
