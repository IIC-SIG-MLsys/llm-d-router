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
	"encoding/binary"
	"errors"
	zmq4 "github.com/go-zeromq/zmq4"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

type sourceAdapter struct {
	parse func(*kvevents.RawMessage) (kvevents.EventBatch, error)
}

func (a sourceAdapter) ShardingKey(*kvevents.RawMessage) string { return "untrusted-topic-address" }
func (a sourceAdapter) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	batch, err := a.parse(msg)
	return "untrusted-topic-address", "model", batch, err
}

type sourceTokens struct{}

func (sourceTokens) BlockSize() int { return 1 }
func (sourceTokens) TokensToKVBlockKeys(_ kvblock.BlockHash, tokens []uint32, _ string, _ []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
	keys := make([]kvblock.BlockHash, len(tokens))
	for i, v := range tokens {
		keys[i] = kvblock.BlockHash(v)
	}
	return keys, nil
}
func stored(value uint32) kvevents.EventBatch {
	return kvevents.EventBatch{Events: []kvevents.GenericEvent{&kvevents.BlockStoredEvent{Tokens: []uint32{value}, BlockHashes: []uint64{uint64(value)}, BlockSize: 1}}}
}
func sourceMeta(name, uid, address string) *fwkdl.EndpointMetadata {
	return &fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "ns", Name: name}, PodUID: types.UID(uid), Address: address, Port: "8000"}
}
func newSourcePool(t *testing.T, index *fakeKVBlockIndex, adapter kvevents.EngineAdapter) *sourceEvents {
	t.Helper()
	s := newSourceEvents(discardCtx(t), index, adapter, "kv@")
	cfg := kvevents.DefaultConfig()
	cfg.Concurrency = 2
	s.pool = kvevents.NewPool(cfg, index, sourceTokens{}, s)
	s.pool.Start(s.ctx)
	t.Cleanup(func() { s.Shutdown(context.Background()) })
	return s
}

// Tests install sources without sockets so queue ordering is deterministic.
func installSource(s *sourceEvents, meta *fwkdl.EndpointMetadata) *eventSource {
	s.mu.Lock()
	defer s.mu.Unlock()
	source := &eventSource{metadata: meta.Clone(), address: meta.Address + ":" + meta.Port, cancel: func() {}, sequences: make(map[string]uint64)}
	s.sources[meta.NamespacedName.String()] = source
	s.resetLocked(source)
	return source
}
func awaitSource(t *testing.T, s *sourceEvents, meta *fwkdl.EndpointMetadata) sourceVersion {
	t.Helper()
	var version sourceVersion
	require.Eventually(t, func() bool { version = s.snapshot(meta); return s.current(version) }, time.Second, time.Millisecond)
	return version
}
func taskCount(s *sourceEvents) int {
	count := 0
	s.tasks.Range(func(_, _ any) bool { count++; return true })
	return count
}

func TestSourcesRetirementOrdersQueuedAndInflightEvents(t *testing.T) {
	var mu sync.Mutex
	state := map[string][]uint32{}
	index := &fakeKVBlockIndex{
		clearFn: func(_ context.Context, pod string) error {
			mu.Lock()
			defer mu.Unlock()
			delete(state, pod)
			return nil
		},
		addFn: func(_ context.Context, _, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			mu.Lock()
			defer mu.Unlock()
			for _, key := range keys {
				state[entries[0].PodIdentifier] = append(state[entries[0].PodIdentifier], uint32(key))
			}
			return nil
		},
	}
	entered, unblock := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(unblock) })
	s := newSourcePool(t, index, sourceAdapter{parse: func(msg *kvevents.RawMessage) (kvevents.EventBatch, error) {
		if msg.Payload[0] == 1 {
			close(entered)
			<-unblock
		}
		return stored(uint32(msg.Payload[0])), nil
	}})
	oldMeta := sourceMeta("old", "uid-old", "127.0.0.1")
	old := installSource(s, oldMeta)
	awaitSource(t, s, oldMeta)
	otherMeta := sourceMeta("other", "uid-other", "127.0.0.2")
	other := installSource(s, otherMeta)
	awaitSource(t, s, otherMeta)
	s.enqueue(old, &kvevents.RawMessage{Topic: "model", Sequence: 1, Payload: []byte{1}})
	<-entered
	const backlog = 8192
	for i := uint64(2); i < backlog+2; i++ {
		s.enqueue(old, &kvevents.RawMessage{Topic: "model", Sequence: i, Payload: []byte{2}})
	}
	require.Equal(t, backlog, taskCount(s))
	s.remove(oldMeta)
	replacement := sourceMeta("new", "uid-new", "127.0.0.1")
	current := installSource(s, replacement)
	s.enqueue(current, &kvevents.RawMessage{Topic: "wrong-address", Sequence: 1, Payload: []byte{3}})
	s.enqueue(old, &kvevents.RawMessage{Topic: "model", Sequence: backlog + 2, Payload: []byte{4}}) // delayed old receiver
	s.enqueue(other, &kvevents.RawMessage{Topic: "model", Sequence: 1, Payload: []byte{5}})
	once.Do(func() { close(unblock) })
	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(state[current.address]) == 1 && state[current.address][0] == 3 && len(state[other.address]) == 1 && state[other.address][0] == 5
	}, 5*time.Second, time.Millisecond)
	mu.Lock()
	gotCurrent := append([]uint32(nil), state[current.address]...)
	gotOther := append([]uint32(nil), state[other.address]...)
	mu.Unlock()
	require.Equal(t, []uint32{3}, gotCurrent)
	require.Equal(t, []uint32{5}, gotOther)
	awaitSource(t, s, replacement)
	require.Eventually(t, func() bool { return taskCount(s) == 0 }, time.Second, time.Millisecond)
}

func TestSourcesReplacementUIDAndAddressOwnership(t *testing.T) {
	p := newExtractorProducer(true)
	defer p.sources.Shutdown(context.Background())
	s := p.sources
	old := sourceMeta("pod", "old", "127.0.0.1")
	require.NoError(t, s.ensure(old, 1))
	v := awaitSource(t, s, old)
	require.NoError(t, s.ensure(old.Clone(), 1))
	require.True(t, s.current(v))
	next := old.Clone()
	next.PodUID = "new"
	require.NoError(t, s.ensure(next, 1))
	require.False(t, s.current(v))
	s.remove(old)
	awaitSource(t, s, next) // late old delete cannot remove the replacement
	other := sourceMeta("different-name", "third", "127.0.0.1")
	require.NoError(t, s.ensure(other, 1))
	awaitSource(t, s, other)
	require.Nil(t, s.snapshot(next).source)
	ids, _ := s.GetActiveSubscribers()
	require.Equal(t, []string{"ns/different-name"}, ids)
}

func TestSourcesSequencesArePerTopicAndRollbackResets(t *testing.T) {
	var clears, parsed atomic.Int32
	s := newSourcePool(t, &fakeKVBlockIndex{clearFn: func(context.Context, string) error { clears.Add(1); return nil }}, sourceAdapter{parse: func(*kvevents.RawMessage) (kvevents.EventBatch, error) {
		parsed.Add(1)
		return kvevents.EventBatch{}, nil
	}})
	meta := sourceMeta("pod", "uid", "127.0.0.1")
	source := installSource(s, meta)
	initial := awaitSource(t, s, meta)
	for _, row := range []struct {
		topic string
		seq   uint64
	}{{"a", 100}, {"b", 1}, {"a", 100}, {"a", 101}} {
		s.enqueue(source, &kvevents.RawMessage{Topic: row.topic, Sequence: row.seq})
	}
	require.Eventually(t, func() bool { return parsed.Load() == 3 }, time.Second, time.Millisecond)
	require.EqualValues(t, 1, clears.Load())
	require.True(t, s.current(initial))
	s.enqueue(source, &kvevents.RawMessage{Topic: "a", Sequence: 1})
	require.False(t, s.current(initial))
	require.Eventually(t, func() bool { return parsed.Load() == 4 }, time.Second, time.Millisecond)
	awaitSource(t, s, meta)
	s.enqueue(source, &kvevents.RawMessage{Topic: "b", Sequence: 0})
	require.Eventually(t, func() bool { return parsed.Load() == 5 }, time.Second, time.Millisecond)
	require.EqualValues(t, 2, clears.Load())
}

func TestSourcesDecodeFailureAndShutdownReclaimSidecars(t *testing.T) {
	for _, cancelWorker := range []bool{false, true} {
		t.Run(map[bool]string{false: "decode-error", true: "worker-canceled"}[cancelWorker], func(t *testing.T) {
			ctx, cancel := context.WithCancel(discardCtx(t))
			defer cancel()
			s := newSourceEvents(ctx, &fakeKVBlockIndex{}, sourceAdapter{parse: func(*kvevents.RawMessage) (kvevents.EventBatch, error) {
				return kvevents.EventBatch{}, errors.New("decode")
			}}, "kv@")
			cfg := kvevents.DefaultConfig()
			cfg.Concurrency = 1
			s.pool = kvevents.NewPool(cfg, s.index, sourceTokens{}, s)
			meta := sourceMeta("pod", "uid", "127.0.0.1")
			source := installSource(s, meta)
			for i := uint64(1); i < 20; i++ {
				s.enqueue(source, &kvevents.RawMessage{Topic: "m", Sequence: i})
			}
			if cancelWorker {
				cancel()
			}
			s.pool.Start(ctx)
			if !cancelWorker {
				require.Eventually(t, func() bool { return taskCount(s) == 0 }, time.Second, time.Millisecond)
			}
			s.Shutdown(context.Background())
			require.Nil(t, s.pool)
			require.Zero(t, taskCount(s))
			s.enqueue(source, &kvevents.RawMessage{Sequence: 21})
			require.Zero(t, taskCount(s))
		})
	}
}

func TestSourcesCopiedMessageCannotBorrowOwnership(t *testing.T) {
	s := newSourceEvents(context.Background(), &fakeKVBlockIndex{}, sourceAdapter{parse: func(*kvevents.RawMessage) (kvevents.EventBatch, error) { return kvevents.EventBatch{}, nil }}, "kv@")
	original := &kvevents.RawMessage{Topic: "m"}
	s.tasks.Store(original, sourceTask{source: &eventSource{address: "pod:8000"}})
	copied := *original
	require.Empty(t, s.ShardingKey(&copied))
	_, _, _, err := s.ParseMessage(&copied)
	require.ErrorContains(t, err, "no subscriber source")
	require.Equal(t, 1, taskCount(s))
	_, _, _, err = s.ParseMessage(original)
	require.NoError(t, err)
	require.Zero(t, taskCount(s))
}

func TestSourcesClearFailureRetriesOnLookup(t *testing.T) {
	var attempts atomic.Int32
	s := newSourcePool(t, &fakeKVBlockIndex{clearFn: func(context.Context, string) error {
		if attempts.Add(1) == 1 {
			return errors.New("transient clear")
		}
		return nil
	}}, sourceAdapter{})
	meta := sourceMeta("pod", "uid", "127.0.0.1")
	installSource(s, meta)
	require.Eventually(t, func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		return !s.sources[meta.NamespacedName.String()].resetPending
	}, time.Second, time.Millisecond)
	require.Nil(t, s.snapshot(meta).source)
	awaitSource(t, s, meta)
	require.EqualValues(t, 2, attempts.Load())
}

func TestProducerDiscardsLookupAcrossSourceEpochChange(t *testing.T) {
	p := newProducerWithIndexer(discardCtx(t), &fakeKVCacheIndexer{index: &fakeKVBlockIndex{}}, &fakeKVBlockScorer{})
	meta := sourceMeta("pod", "uid", "127.0.0.1")
	s := newSourcePool(t, &fakeKVBlockIndex{}, sourceAdapter{})
	p.sources = s
	old := installSource(s, meta)
	awaitSource(t, s, meta)
	idx := p.kvCacheIndexer.(*fakeKVCacheIndexer).index
	idx.lookup = func(context.Context, []kvblock.BlockHash, sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
		s.mu.Lock()
		s.resetLocked(old)
		s.mu.Unlock()
		return map[kvblock.BlockHash][]kvblock.PodEntry{1: {{PodIdentifier: old.address, DeviceTier: "gpu"}}}, nil
	}
	p.kvBlockScorer = &fakeKVBlockScorer{score: func(context.Context, []kvblock.BlockHash, map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error) {
		return map[string]float64{old.address: 1}, nil
	}}
	ep := scheduling.NewEndpoint(meta, nil, nil)
	req := &scheduling.InferenceRequest{RequestID: "request"}
	run := func() {
		require.NoError(t, p.produceFromBlockKeys(context.Background(), trace.SpanFromContext(context.Background()), req, []scheduling.Endpoint{ep}, [][]kvblock.BlockHash{{1}}))
	}
	run()
	data, present := ep.Get(p.dk.String())
	require.True(t, present)
	info, ok := data.(*attrprefix.PrefixCacheMatchInfo)
	require.True(t, ok)
	require.Zero(t, info.DeviceCachedBlockCount())
	awaitSource(t, s, meta)
	idx.lookup = func(context.Context, []kvblock.BlockHash, sets.Set[string]) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
		return map[kvblock.BlockHash][]kvblock.PodEntry{1: {{PodIdentifier: old.address, DeviceTier: "gpu"}}}, nil
	}
	run()
	data, _ = ep.Get(p.dk.String())
	info = data.(*attrprefix.PrefixCacheMatchInfo)
	require.Equal(t, 1, info.DeviceCachedBlockCount())
}

func TestSourcesShutdownUnblocksReceive(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	require.NoError(t, listener.Close())
	pub := zmq4.NewPub(context.Background())
	require.NoError(t, pub.Listen("tcp://127.0.0.1:"+strconv.Itoa(port)))
	defer pub.Close()
	p := newExtractorProducer(true)
	require.NoError(t, p.sources.ensure(sourceMeta("pod", "uid", "127.0.0.1"), port))
	awaitSource(t, p.sources, sourceMeta("pod", "uid", "127.0.0.1"))
	seq := make([]byte, 8)
	binary.BigEndian.PutUint64(seq, 1)
	require.Eventually(t, func() bool {
		if err := pub.Send(zmq4.NewMsgFrom([]byte("kv@127.0.0.1:8000@model"), seq, []byte{1})); err != nil {
			return false
		}
		p.sources.mu.Lock()
		defer p.sources.mu.Unlock()
		return len(p.sources.sources["ns/pod"].sequences) > 0
	}, time.Second, 10*time.Millisecond)
	done := make(chan struct{})
	go func() { p.sources.Shutdown(context.Background()); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("subscriber receiver did not stop")
	}
	require.Zero(t, taskCount(p.sources))
}

func TestPreRequestDoesNotSeedRetiredSource(t *testing.T) {
	var adds atomic.Int32
	index := &fakeKVBlockIndex{addFn: func(context.Context, []kvblock.BlockHash, []kvblock.BlockHash, []kvblock.PodEntry) error {
		adds.Add(1)
		return nil
	}}
	p := newProducerForPreRequest(discardCtx(t), true, index)
	s := newSourcePool(t, index, sourceAdapter{})
	p.sources = s
	meta := sourceMeta("pod", "old", "127.0.0.1")
	installSource(s, meta)
	version := awaitSource(t, s, meta)
	req := &scheduling.InferenceRequest{RequestID: "old-request"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: [][]kvblock.BlockHash{{1}}, sources: map[string]sourceVersion{meta.NamespacedName.String(): version}})
	s.remove(meta)
	replacement := meta.Clone()
	replacement.PodUID = "new"
	installSource(s, replacement)
	fresh := awaitSource(t, s, replacement)
	p.PreRequest(context.Background(), req, primaryOnly("default", scheduling.NewEndpoint(meta, nil, nil)))
	require.Zero(t, adds.Load())
	req.RequestID = "new-request"
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: [][]kvblock.BlockHash{{1}}, sources: map[string]sourceVersion{replacement.NamespacedName.String(): fresh}})
	p.PreRequest(context.Background(), req, primaryOnly("default", scheduling.NewEndpoint(replacement, nil, nil)))
	require.EqualValues(t, 1, adds.Load())
}

func TestNewRejectsUnownedGlobalEventsDuringDiscovery(t *testing.T) {
	config := kvevents.DefaultConfig()
	config.ZMQEndpoint = "tcp://127.0.0.1:5557"
	_, err := New(context.Background(), "test", PluginConfig{KVEventsConfig: config})
	require.ErrorContains(t, err, "mutually exclusive")
}

func TestSourcesContainerRestartWithoutSequenceRollback(t *testing.T) {
	p := newExtractorProducer(true)
	defer p.sources.Shutdown(context.Background())
	s := p.sources
	meta := sourceMeta("pod", "uid", "127.0.0.1")
	meta.PodRuntimeID = "container-a"
	require.NoError(t, s.ensure(meta, 1))
	version := awaitSource(t, s, meta)
	replacement := meta.Clone()
	replacement.PodRuntimeID = "container-b"
	require.NoError(t, s.ensure(replacement, 1))
	require.False(t, s.current(version))
	require.Nil(t, s.snapshot(meta).source)
	s.remove(meta)
	awaitSource(t, s, replacement)
}

func TestSourcesIPv6MatchesIndexIdentifier(t *testing.T) {
	p := newExtractorProducer(true)
	defer p.sources.Shutdown(context.Background())
	meta := sourceMeta("pod", "uid", "::1")
	require.NoError(t, p.sources.ensure(meta, 5557))
	source := awaitSource(t, p.sources, meta).source
	require.Equal(t, "::1:8000", source.address)
	require.Equal(t, "tcp://[::1]:5557", source.socket)
}
