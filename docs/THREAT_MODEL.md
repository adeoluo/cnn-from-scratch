# Threat Model - CNN From Scratch

## Assets
- The HTTP service's availability and the integrity of its responses.

## Trust boundaries
- The only external input is the JSON body and path of a request. The model, data, and training
  are in-process and trusted.

## Entry points and mitigations

### `POST /classify`
- **Malformed JSON** is rejected with 400 before routing.
- **Validation:** the image must be a list of exactly `size` rows, each with exactly `size`
  numeric pixels; non-numeric values and booleans are rejected. Invalid input returns 400 with a
  message, never a stack trace.
- **Bounded work:** a classification is a single fixed-size forward pass, so a request cannot
  force unbounded computation.

### `GET` routes
- `/health`, `/ready`, `/evaluate`, and `/metrics` take no user input and return fixed-shape
  payloads. `/metrics` exposes only aggregate counters.

## Non-threats (by construction)
- **No secrets:** the service holds no credentials and makes no outbound calls.
- **No persistence:** state is the in-memory trained network; nothing is written on the request path.
- **No code execution:** inputs are numbers; no user-supplied string is evaluated.

## Supply chain
- Zero runtime dependencies. Development tooling is pinned via `uv.lock`. CodeQL and dependency
  scanning run in CI.

## Residual risk
- A flood of requests could exhaust CPU; a production deployment would add a rate limiter and run
  behind a reverse proxy.
