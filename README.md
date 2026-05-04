# image-gen-svc

A small, generic HTTP server that wraps [diffusers](https://github.com/huggingface/diffusers)
pipelines (SDXL, Chroma, AuraFlow, Z-Image) behind a content-neutral REST API.
Bring your own checkpoints; the service streams `image/webp` bytes back,
surfaces generation progress over Server-Sent Events, and lazily downloads
model files (single-file URLs or Hugging Face snapshots) from a built-in
registry.

The service has **no opinion** about who's calling it, what the prompts mean,
or where the output should be stored. It takes a prompt and parameters, returns
image bytes. Path layouts, file naming, content policy — those belong to the
caller.

License: Apache-2.0 (service code). Model checkpoint licenses are independent
and travel with the model files.

## Quickstart

```bash
docker compose -f compose.example.yml up
```

First request triggers a model download. With request defaults (`safe=true`,
no `model_id`), that's the **z-image-turbo** snapshot (~20 GB,
`Tongyi-MAI/Z-Image-Turbo`). The `/events/{job_id}` SSE channel surfaces a
`model_loading` event so callers can show a progress UI; subsequent requests
hit the cached file.

To pre-populate the volume in one shot instead of paying first-request
latency, run the bundled pull script as a one-shot container against the
same `image-gen-models` volume:

```bash
docker run --rm \
  -v image-gen-models:/models \
  ghcr.io/muxu-io/image-gen-svc:v0.1.0 \
  python3.11 scripts/pull_models.py
```

The script handles both shapes the registry uses: `url` entries are streamed
directly; `repo_id` entries are pulled via `huggingface_hub.snapshot_download`.

## API

### `POST /render` — txt2img

```http
POST /render HTTP/1.1
Content-Type: application/json

{
  "prompt": "a contemplative scholar at dusk",
  "width": 1024, "height": 1024,
  "model_id": "z-image-turbo"
}
```

Only `prompt`, `width`, and `height` are strictly required. `negative_prompt`,
`model_id`, `steps`, `guidance`, `seed`, `safe`, and `job_id` all have
sensible defaults — supply them only when you want to override.

`model_id` is optional. It accepts either a model id (`z-image-turbo`,
`chroma-1-hd`, ...) or an alias (`photorealistic`, `anime`, ...). When
omitted, the registry's `default_alias` is used.

`steps` and `guidance` are optional. When omitted, the service uses the
resolved model's `default_steps` / `default_guidance` from the registry,
falling back to architecture-tier defaults (`z_image`: 4 / 3.5; `sdxl`:
25 / 7.0; `chroma`: 28 / 4.0; `auraflow`: 25 / 5.0). The effective values
for every model are visible on `GET /models` so callers can preview them
without a probe render.

`safe` defaults to `true`. Under `safe=true`, alias resolution skips entries
marked `safe: false` in the registry; explicit `model_id` lookups bypass the
filter (the caller has taken responsibility). Set `safe=false` to opt into
the unrestricted pool.

Returns `200 OK` with body = raw `image/webp` bytes and metadata in headers:

| Header | Meaning |
|---|---|
| `X-Job-Id` | Job identifier (echoes the supplied `job_id` if any) |
| `X-Model-Used` | The model the request resolved to |
| `X-Seed` | Seed used (useful when the caller didn't supply one) |
| `X-Generation-Time-Ms` | Wall-clock generation time |

### `POST /render` — img2img

Same endpoint, `Content-Type: multipart/form-data`. Two parts:

- `payload` — JSON, same fields as above
- `reference_image` — file (`image/webp` or `image/png`)

The reference is used as IP-Adapter / img2img conditioning input by pipelines
that support it (chroma, auraflow). Pipelines that don't ignore it.

### `GET /events/{job_id}` — SSE progress

Subscribe before POSTing (use a client-supplied `job_id`) to avoid the race
where progress events fire before the subscriber connects.

Event types:

| Event | Data shape |
|---|---|
| `job_queued` | `{}` |
| `model_loading` | `{model, url}` for url-based fetches; `{model, repo_id}` for snapshots |
| `job_started` | `{model}` |
| `step_progress` | `{step, total_steps}` |
| `job_completed` | `{model_used, seed, generation_time_s}` |
| `job_failed` | `{error, message}` |

### `GET /models` — list registry contents

Each entry exposes its registry metadata plus three flags / hints:

- `safe: bool` — whether this model is **not** specifically NSFW-tuned. The
  request-time `safe` filter on `/render` uses this. `false` does not mean
  "renders NSFW only" — the designation gates filter behavior, not generation
  behavior.
- `loaded: bool` — whether the checkpoint is on disk now (vs. needs download).
- `default_steps: int` / `default_guidance: float` — the values `/render`
  will use when the caller omits `steps` / `guidance`. Resolved from the
  entry's declared defaults, falling back to architecture-tier defaults.

The response also includes a top-level `default_alias` and `default_models`
table; the latter is the tiebreaker the resolver consults when an alias has
multiple matching entries (e.g. `photorealistic` matches both `z-image-turbo`
and `chroma-1-hd`).

### `GET /version`

Returns service version + diffusers/torch/transformers versions so callers can
detect drift.

### `GET /health`

Liveness probe. `{ok: true, ...}`.

### `GET /docs`

FastAPI auto-generated OpenAPI UI.

### Errors

All error responses are JSON:

```json
{ "error": "invalid_request", "message": "...", "job_id": "..." | null }
```

| Status | `error` value | When |
|---|---|---|
| 400 | `invalid_request` | Schema validation, unknown model_id, multipart issues |
| 400 | `no_safe_model_for_alias` | Alias matched only unsafe entries under `safe=true`; envelope includes the offending `alias` |
| 401 | `unauthorized` | `IMAGE_GEN_API_KEY` set, `Authorization` missing/wrong |
| 500 | `generation_failed` | Pipeline crash, OOM, unexpected runtime failure |
| 503 | `model_loading` | First-request fetch in progress (single-file url or HF snapshot); retry |

There is no `nsfw_blocked` status. The service has no content filter; the
`safe` axis is a routing knob over the bundled model pool, not a generation-
time policy.

## Authentication

Off by default. Set the `IMAGE_GEN_API_KEY` env var to require
`Authorization: Bearer <token>` on `/render`. The `/health`, `/models`,
`/version`, and `/docs` endpoints remain public regardless. Token comparison
uses `hmac.compare_digest`, so the contents don't leak through timing.

The service does not terminate TLS itself. Deploy it behind a TLS-terminating
reverse proxy (nginx, Caddy, Traefik, an ingress controller, a cloud LB) if
it's reachable from anything other than loopback — otherwise the bearer
token travels in the clear.

## Concurrency

Single-GPU server. Pipeline cache is LRU=1: switching between architectures
evicts the previous via `aclose()`. Renders are serialized — do not expect a
fleet from a single container.

## Model registry

Models are declared in `src/image_gen_svc/default_registry.yml`. Each entry has:

- `path` — where the checkpoint lives in-container (default volume: `/models`)
- `url` *or* `repo_id` — where to fetch it on first request (`url` for
  single-file `.safetensors`; `repo_id` for Hugging Face snapshot directories)
- `architecture` — which pipeline adapter to use (`sdxl`, `chroma`, `auraflow`,
  `z_image`)
- `aliases` — convenience names callers can pass instead of the model id
- `safe` — whether the model is in the `safe=true` pool (defaults to `false`
  if omitted)
- optional `default_steps`, `default_guidance`, `vram_gb`, `seed_stability`,
  `license`, `sha256`

The bundled registry ships:

| id | architecture | aliases | safe |
|---|---|---|---|
| `z-image-turbo` | z_image | photorealistic, stylized_realism | true |
| `animagine-xl-4.0` | sdxl | illustration, anime, fantasy | true |
| `chroma-1-hd` | chroma | photorealistic, stylized_realism | false |
| `realvis-xl-v5` | sdxl | photorealistic_lowvram | true |
| `pony-v7-base` | auraflow | illustration, anime, fantasy | false |

To add a model, build a derivative image:

```dockerfile
FROM ghcr.io/muxu-io/image-gen-svc:v0.1.0
COPY my-registry.yml /app/src/image_gen_svc/default_registry.yml
# Optional: pre-bake the checkpoint to skip first-request download
COPY my-extra-model.safetensors /models/
```

The service does not honor a runtime overlay — derivative images are the
extension point.

## Versioning

Semver. Pre-`1.0.0`, breaking changes are allowed in `0.MINOR` bumps. After
`1.0.0`, breaking changes require major bumps. The `/version` endpoint returns
the running build's exact versions.

**Pin exact tags** (`v0.1.0`) in production — never `latest` or `main`. Bumps
should be deliberate and visible in caller's git history.

## Development

```bash
poetry install
poetry run pytest
poetry run ruff check src tests
poetry run black src tests
```

The package is GPU-free at the Python level — `torch`, `diffusers`, etc. are
installed by the Dockerfile only. Hermetic tests use a mock pipeline so they
run on any machine.

### Integration tests

Two opt-in suites that spin up the built container inside a pytest fixture,
both excluded from the default `pytest` run. Build the image first:

```bash
docker build -t image-gen-svc:integration .
```

**Mock mode** — full HTTP stack against the mock pipeline; no GPU, no model
downloads:

```bash
poetry run pytest -m integration
```

**GPU mode** — real `--gpus all` container, real pipelines. The render test
parametrizes over every entry in the registry, so each model is its own test
case:

```bash
poetry run pytest -m gpu
```

Models lazy-download on first request and cache into a named docker volume.
Pre-populate to keep cold runs from being dominated by network:

```bash
docker run --rm \
  -v image-gen-svc-integration-models:/models \
  image-gen-svc:integration \
  python3 scripts/pull_models.py
```

Environment overrides:

| Var | Default | Effect |
|---|---|---|
| `IMAGE_GEN_SVC_IMAGE` | `image-gen-svc:integration` | image tag both fixtures use |
| `IMAGE_GEN_SVC_INTEGRATION_MODELS_VOLUME` | `image-gen-svc-integration-models` | volume name or host path mounted at `/models` (GPU only) |
| `IMAGE_GEN_SVC_INTEGRATION_RENDER_TIMEOUT_S` | `1800` | per-model render budget including download retries |
| `IMAGE_GEN_SVC_INTEGRATION_RENDER_MODEL_IDS` | (all) | comma-separated subset of registry ids to test |
| `HF_TOKEN` | (unset) | forwarded into the container; required for gated repos (e.g. chroma → FLUX.1-schnell) |

Without `HF_TOKEN`, gated-model render tests skip cleanly with a clear
reason rather than fail. FLUX.1-schnell is Apache 2.0 and free — accept the
license at huggingface.co/black-forest-labs/FLUX.1-schnell, then generate a
Read token in HF account settings.
