# Game ID Service

Local FastAPI service for identifying the most likely game mentioned in a news article body.

## Contract

### `GET /health`

Returns model and runtime status.

### `POST /identify/game`

Request:

```json
{
  "request_id": "debug-1",
  "article_id": 123,
  "body_text": "...",
  "task": "Identify the game mentioned in the article body and return only the English title or unknown."
}
```

Response:

```json
{
  "request_id": "debug-1",
  "article_id": 123,
  "status": "ok",
  "identified_game_name": "Black Myth: Wukong",
  "confidence": 0.92,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "external_game_id": null,
  "slug": "black-myth-wukong",
  "error": null
}
```

If the game cannot be identified confidently, `identified_game_name` must be `unknown`.

`task` is optional. If omitted, the service uses its default identification task.

If token auth is enabled, send one of these headers:

- `X-News-Game-Id-Token`
- `X-Game-Id-Token`
- `X-Translation-Token` as legacy fallback

## Local setup

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and adjust values.
4. Start both windows with `scripts\run_windows_game_id.ps1`, or start them separately with `scripts\start_service.ps1` and `scripts\start_tunnel.ps1`.
5. If you only want the tunnel, run `scripts\start_tunnel.ps1`.

## Tunnel

The service listens only on `127.0.0.1:8018`. The VPS reaches it through an SSH reverse tunnel on `127.0.0.1:19192` by default.

If the VPS client runs in Docker and needs to access the tunnel through the host, bind the tunnel to `0.0.0.0` and make sure `sshd` allows remote ports on non-loopback interfaces.

If you manage a miner, prefer `MINER_LAUNCH_PATH=C:\\cofex\\detect\\game-id-service\\scripts\\start_miner_hidden.ps1`. That wrapper starts the miner hidden and avoids the old translator wrapper window spam.

Before starting the miner, the service checks the neighboring `translation` service at `TRANSLATION_HEALTH_URL`. If `translation` is still processing or still has its model loaded, the miner start is skipped and `translation` can bring it back later.

The game-identification model itself now runs in a short-lived worker process per request, so the main API process does not keep the GPU model resident between calls.
As a result, `/health` reports `model_loaded=false` for the API process; the real model lives only inside the per-request worker and exits after the response is produced.
