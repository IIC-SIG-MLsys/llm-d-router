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
	"fmt"
	"net"
	"strconv"
	"sync"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

type subscriberManager interface {
	Shutdown(context.Context)
	GetActiveSubscribers() ([]string, []string)
}

type eventSource struct {
	metadata            *fwkdl.EndpointMetadata
	address, socket     string
	cancel              context.CancelFunc
	epoch               uint64
	ready, resetPending bool
	sequences           map[string]uint64
}

type sourceVersion struct {
	source *eventSource
	epoch  uint64
}
type sourceTask struct {
	source *eventSource
	reset  bool
	epoch  uint64
}

// sourceEvents adds source ownership to v0.9's raw messages without changing the
// engine wire format or the shared worker count. The pool preserves pointers.
type sourceEvents struct {
	ctx       context.Context
	index     kvblock.Index
	adapter   kvevents.EngineAdapter
	topic     string
	pool      *kvevents.Pool
	mu        sync.Mutex
	sources   map[string]*eventSource
	tasks     sync.Map // *kvevents.RawMessage -> sourceTask, owned until ParseMessage.
	closed    bool
	stop      sync.Once
	receivers sync.WaitGroup
}

func newSourceEvents(ctx context.Context, index kvblock.Index, adapter kvevents.EngineAdapter, topic string) *sourceEvents {
	return &sourceEvents{ctx: ctx, index: index, adapter: adapter, topic: topic, sources: make(map[string]*eventSource)}
}

func (s *sourceEvents) ShardingKey(msg *kvevents.RawMessage) string {
	if task, ok := s.tasks.Load(msg); ok {
		return task.(sourceTask).source.address
	}
	return ""
}

func (s *sourceEvents) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	value, ok := s.tasks.LoadAndDelete(msg)
	if !ok {
		return "", "", kvevents.EventBatch{}, fmt.Errorf("KV event has no subscriber source")
	}
	task := value.(sourceTask)
	if task.reset {
		err := s.index.Clear(s.ctx, task.source.address)
		s.mu.Lock()
		if task.source.epoch == task.epoch {
			task.source.resetPending = false
			task.source.ready = err == nil
		}
		s.mu.Unlock()
		return task.source.address, "", kvevents.EventBatch{}, err
	}
	_, model, batch, err := s.adapter.ParseMessage(msg)
	return task.source.address, model, batch, err
}

func (s *sourceEvents) queueLocked(source *eventSource, msg *kvevents.RawMessage, reset bool) {
	s.tasks.Store(msg, sourceTask{source: source, reset: reset, epoch: source.epoch})
	s.pool.AddTask(msg)
}

func (s *sourceEvents) resetLocked(source *eventSource) {
	source.epoch++
	source.sequences = make(map[string]uint64)
	source.ready, source.resetPending = false, true
	s.queueLocked(source, &kvevents.RawMessage{}, true)
}

func (s *sourceEvents) ensure(meta *fwkdl.EndpointMetadata, port int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.ctx.Err() != nil {
		return fmt.Errorf("KV event subscribers are stopped")
	}
	key := meta.NamespacedName.String()
	address := fmt.Sprintf("%s:%s", meta.Address, meta.Port)
	socket := "tcp://" + net.JoinHostPort(meta.Address, strconv.Itoa(port))
	if old := s.sources[key]; old != nil {
		if old.metadata.PodUID == meta.PodUID && old.metadata.PodRuntimeID == meta.PodRuntimeID && old.address == address && old.socket == socket {
			return nil
		}
		old.cancel()
		s.resetLocked(old)
		delete(s.sources, key)
	}
	// An IP:port can be reused under a different Pod name before its delete arrives.
	for id, old := range s.sources {
		if old.address == address {
			old.cancel()
			s.resetLocked(old)
			delete(s.sources, id)
		}
	}
	ctx, cancel := context.WithCancel(s.ctx)
	source := &eventSource{metadata: meta.Clone(), address: address, socket: socket,
		cancel: cancel, sequences: make(map[string]uint64)}
	s.sources[key] = source
	s.resetLocked(source)
	s.receivers.Add(1)
	go s.subscribe(ctx, source)
	return nil
}

func (s *sourceEvents) remove(meta *fwkdl.EndpointMetadata) {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := meta.NamespacedName.String()
	source := s.sources[key]
	if source == nil || meta.PodUID != source.metadata.PodUID || meta.PodRuntimeID != source.metadata.PodRuntimeID {
		return
	}
	delete(s.sources, key)
	source.cancel()
	s.resetLocked(source)
}

func (s *sourceEvents) enqueue(source *eventSource, msg *kvevents.RawMessage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.sources[source.metadata.NamespacedName.String()] != source {
		return
	}
	if seq, seen := source.sequences[msg.Topic]; seen {
		if msg.Sequence == seq {
			return
		}
		if msg.Sequence < seq {
			s.resetLocked(source)
		}
	}
	if !source.ready && !source.resetPending {
		s.resetLocked(source)
	}
	source.sequences[msg.Topic] = msg.Sequence
	s.queueLocked(source, msg, false)
}

func (s *sourceEvents) snapshot(meta *fwkdl.EndpointMetadata) sourceVersion {
	s.mu.Lock()
	defer s.mu.Unlock()
	source := s.sources[meta.NamespacedName.String()]
	if s.closed || source == nil || source.metadata.PodUID != meta.PodUID || source.metadata.PodRuntimeID != meta.PodRuntimeID || source.address != fmt.Sprintf("%s:%s", meta.Address, meta.Port) {
		return sourceVersion{}
	}
	if !source.ready {
		if !source.resetPending {
			s.resetLocked(source)
		}
		return sourceVersion{}
	}
	return sourceVersion{source: source, epoch: source.epoch}
}

func (s *sourceEvents) current(version sourceVersion) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.currentLocked(version)
}

func (s *sourceEvents) currentLocked(version sourceVersion) bool {
	source := version.source
	return !s.closed && source != nil && source.ready && source.epoch == version.epoch && s.sources[source.metadata.NamespacedName.String()] == source
}

func (s *sourceEvents) GetActiveSubscribers() ([]string, []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids, endpoints := []string{}, []string{}
	for id, source := range s.sources {
		ids = append(ids, id)
		endpoints = append(endpoints, source.socket)
	}
	return ids, endpoints
}

func (s *sourceEvents) Shutdown(ctx context.Context) {
	s.stop.Do(func() {
		s.mu.Lock()
		s.closed = true
		sources := s.sources
		s.sources = make(map[string]*eventSource)
		for _, source := range sources {
			source.cancel()
		}
		s.mu.Unlock()
		s.receivers.Wait()
		s.pool.Shutdown(ctx)
		s.tasks.Clear()
		s.mu.Lock()
		// Canceled workers can leave raw messages in the stopped pool queues.
		s.pool = nil
		s.mu.Unlock()
	})
}

func (s *sourceEvents) subscribe(ctx context.Context, source *eventSource) {
	defer s.receivers.Done()
	for ctx.Err() == nil {
		s.receive(ctx, source)
		select {
		case <-ctx.Done():
			return
		case <-time.After(5 * time.Second):
		}
	}
}

func (s *sourceEvents) receive(ctx context.Context, source *eventSource) {
	sub := zmq4.NewSub(ctx, zmq4.WithDialerMaxRetries(0))
	defer sub.Close()
	if err := sub.Dial(source.socket); err != nil {
		if ctx.Err() == nil {
			log.FromContext(ctx).Error(err, "Failed to connect KV event subscriber", "endpoint", source.socket)
		}
		return
	}
	if err := sub.SetOption(zmq4.OptionSubscribe, s.topic); err != nil {
		log.FromContext(ctx).Error(err, "Failed to subscribe to KV events", "endpoint", source.socket)
		return
	}
	for ctx.Err() == nil {
		msg, err := sub.Recv()
		if err != nil {
			if ctx.Err() == nil {
				log.FromContext(ctx).Error(err, "KV event subscriber disconnected", "endpoint", source.socket)
			}
			return
		}
		if len(msg.Frames) != 3 || len(msg.Frames[1]) < 8 {
			log.FromContext(ctx).Error(nil, "Invalid KV event frame", "endpoint", source.socket)
			continue
		}
		s.enqueue(source, &kvevents.RawMessage{Topic: string(msg.Frames[0]), Sequence: binary.BigEndian.Uint64(msg.Frames[1]), Payload: msg.Frames[2]})
	}
}
