# distkvproto

Generated gRPC stubs for the DistKV control-plane protocol
(`proto/distkv/v1/control.proto` in the DistKV repository).

This package is a local vendor of the generated Go code so the router can
speak to a DistKV `CacheServerController` without importing the DistKV Go
module (whose `go` directive is newer than this module's).  Do not hand-edit:
regenerate from DistKV (`cd control && go generate` / `make gen-go`) and copy
`control/v1/control.pb.go` + `control/v1/control_grpc.pb.go` here.

Only the RPCs used by the routing producer are exercised today:

- `CacheServerController.GetRoutingInfo`
- `CacheServerController.GlobalLookup`
- `CacheServerController.LookupRouting`
