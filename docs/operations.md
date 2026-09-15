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
| `RL_LANDER_ROOT` | the directory holding `pyproject.toml` | the root every path hangs off |
| `PYTHONHASHSEED` | unset | string hashing, for a result that must cross machines |

`.env.example` carries the same list, with what each one is for. There is no secret: no key,
no database, no third-party call. **Nothing loads a `.env` on its own.** Pass it to the
command that needs it, `uv run --env-file .env uvicorn rl_lander.api:app`, so that what the
process reads is visible in what was typed.

## In a container

```powershell
docker compose up --build    # the API on :8000, the dashboard on :8501
```

One image builds both services. It carries the package, the checkpoint and the published
exports, and neither the tests nor the documents nor `var/`; `.dockerignore` is where that
list lives. Three things in it are worth knowing before the first `up`:

| | |
|---|---|
| **The CPU build of PyTorch** | `tool.uv.sources` routes torch to PyTorch's CPU index off Windows. Without it the Linux resolution pulls a CUDA runtime, and the image goes from 2.0 GB to nearly 5. |
| **`SDL_VIDEODRIVER=dummy`** | pygame renders the replay frames through SDL, which aborts when it finds no display. The frames are pixels in memory and leave over HTTP; no container of this project ever needs a screen. |
| **The probe is `/health`, not `/ready`** | a container restarted for not being ready yet never becomes ready. The distinction is the next section. |

The cockpit is not among the services. It replays each episode locally to build its
animation, which means Box2D, pygame and a GIF encoder inside the page: worth it on a
workstation, and 700 MB of image for a page that already has a local `uv run` command.

The first run of the shipped policy inside the container scores 280.34 on seed 2024, against
280.49 on the machine that produced `reports/evaluation_episodes.csv`. The policy is the same
file; the difference is what two builds of a tensor library do with the same weights, and it
is the reason the reproducibility claim of this repository is made about one platform and not
about all of them.

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
