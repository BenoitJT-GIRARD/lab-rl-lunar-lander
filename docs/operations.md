# Running the service

Start it, ask it something, and know what to look at when it stops answering.
`docs/architecture.md` argues why the inference sits behind a service; this page is the
operating manual.

## Starting it

```powershell
uv sync --all-extras
uv run uvicorn rl_lander.api:app                        # http://127.0.0.1:8000/docs
```

Nothing else is required. The checkpoint at `models/ppo/best.zip` is loaded once, at boot, in
the application's `lifespan`.

| Variable | Default | What it changes |
|---|---|---|
| `RL_LANDER_MODEL_PATH` | `models/ppo/best.zip` | the checkpoint to serve |
| `RL_LANDER_ALGO` | `ppo` | the class that loads it; `ppo` or `dqn` |
| `RL_LANDER_API_URL` | `http://127.0.0.1:8000` | where the cockpit looks for the service |

`.env.example` carries the same list. There is no secret: no key, no database, no third-party
call. A `.env` is read when one is present, and nothing breaks when none is.

## What answers

<!-- source: docs/images/MANIFEST.json -->
![The OpenAPI page of the running service, with the meta and agent tags expanded and the schema of the observation the play route accepts](images/api-docs.png)

| Route | Method | Answers |
|---|---|---|
| `/health` | GET | the process is serving |
| `/ready` | GET | a checkpoint is loaded, and which |
| `/info` | GET | algorithm, environment, version, checkpoint path |
| `/play` | POST | the deterministic action for one observation |
| `/run` | POST | a whole episode: actions, rewards, final state |
| `/reset` | POST | a starting observation for a given seed |

The page above is generated from the Pydantic models, so what a caller reads is what the
service enforces. Why the first two are separate routes rather than one is in
`docs/architecture.md`.

`POST /run` is the slow one: it simulates up to a thousand environment steps before it answers.
Everything else is a few milliseconds.

## When something goes wrong

| Symptom | Likely cause | What to do |
|---|---|---|
| `/ready` answers 503 | no checkpoint at the path the service resolved | check `RL_LANDER_MODEL_PATH`; `GET /info` says what it tried |
| A `/play` call returns 422 | the observation is not eight finite numbers inside the box | read the message: it names the index and the bounds |
| Loading fails on tensor shapes | the checkpoint was saved by the other algorithm | set `RL_LANDER_ALGO` to match |
| The cockpit says the replay diverged | the page and the service are not on the same environment version | reinstall from the lock file on both sides |
| The dashboard says an export is missing columns | it was written by an older exporter | rerun `scripts/evaluate_and_export.py` |

## What is not here

**No container, no queue, no worker.** `docs/architecture.md` gives the reasoning and the size
of the load that would change it.

**No authentication.** Five read-only routes over a policy committed in the open. A key would
protect nothing and would be one more thing to leak.

**Nothing watches it.** No metrics endpoint, no traces, no alerting. The service is started by
hand, for a demonstration or for the cockpit, and it is stopped the same way.
