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
	"fmt"

	controlv1 "github.com/llm-d/llm-d-router/pkg/distkvproto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client queries read-only instance cache observations.
type Client struct {
	conn *grpc.ClientConn
	stub controlv1.CacheServerControllerClient
}

func newClient(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial distkv controller %s: %w", addr, err)
	}
	return &Client{conn: conn, stub: controlv1.NewCacheServerControllerClient(conn)}, nil
}

func (c *Client) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}
