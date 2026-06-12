# server-lifecycle

## ADDED Requirements

### Requirement: Shutdown-responsive batch pipeline

The model worker SHALL check the shutdown event after dequeuing a batch and before
starting a forward pass, so shutdown latency is bounded by at most one in-flight
forward pass rather than the queue backlog.

#### Scenario: SIGTERM with queued batches

- GIVEN a running engine with multiple batches queued behind a slow forward pass
- WHEN `astop()` is invoked (as by uvicorn lifespan shutdown on SIGTERM)
- THEN no new forward pass starts after the shutdown event is set and `astop()` returns promptly once the in-flight pass completes

### Requirement: Bounded request wait (opt-in)

The batch handler SHALL support an optional request timeout, configured via
`INFINITY_REQUEST_TIMEOUT_S` (default 0, disabled), bounding how long a caller waits
for embedding results. With the default, behavior SHALL be unchanged.

#### Scenario: Pipeline death with timeout configured

- GIVEN `INFINITY_REQUEST_TIMEOUT_S=1` and a pipeline whose result future never completes
- WHEN a client awaits an embed call
- THEN the call fails with a timeout error mapped to a 5xx response within approximately 1 second instead of hanging indefinitely

#### Scenario: Default disabled

- GIVEN `INFINITY_REQUEST_TIMEOUT_S` unset
- WHEN a client awaits an embed call
- THEN no timeout is applied (current behavior)

### Requirement: Readiness-gated health endpoint

The `/health` endpoint SHALL return HTTP 503 until all engines report running, and
HTTP 200 with the existing payload afterwards, so supervisors (llama-swap polls its
checkEndpoint for HTTP 200) get an explicit readiness contract.

#### Scenario: Probe before engines are running

- GIVEN the app is serving requests while `engine_array.is_running()` is false
- WHEN GET `/health` is called
- THEN the response is 503 with body `{"status": "loading"}`

#### Scenario: Probe after startup

- GIVEN engines are started and running
- WHEN GET `/health` is called
- THEN the response is 200 with the unix-timestamp payload

#### Scenario: Probe after shutdown begins

- GIVEN `astop()` has set engines to not running
- WHEN GET `/health` is called
- THEN the response is 503
