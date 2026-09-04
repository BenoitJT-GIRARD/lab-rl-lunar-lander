# Why a service, and where the boundaries are

One page. Read it after the README and before the code.

## The shape

```
                 models/ppo/best.zip          data/*.csv, data/*.json
                          │                             │
                          ▼                             ▼
   ┌──────────────┐   ┌────────────────┐        ┌──────────────────┐
   │  cockpit     │──▶│  FastAPI       │        │   dashboard      │
   │  (Streamlit) │   │  rl_lander.api │        │   (Streamlit)    │
   └──────────────┘   └────────────────┘        └──────────────────┘
          │                    │                          │
          └────────────────────┴──────────────────────────┘
                               │
                    src/rl_lander/  — one implementation
```

Three surfaces, one library. The cockpit and the dashboard hold no reinforcement learning:
one asks the API for a trajectory, the other reads the published exports.

## Why the inference is behind a service

**Loading is expensive and shared.** Unpacking a Stable-Baselines3 checkpoint and building
the torch graph takes a noticeable moment. The service pays it once, at boot, in its
`lifespan`. A frontend that loaded the model itself would pay it per session, and two
frontends would pay it twice.

**One implementation of "play an episode".** `LunarLanderAgent.play_episode` is the only
place an action is chosen and a landing is judged. The API, the video recorder and the
notebook all go through it. When `landed` was being read from the score instead of the
environment's terminal reward, there was exactly one line to fix, and fixing it corrected
every surface at once.

**The frontends become replaceable.** A JSON contract is the only thing the cockpit knows
about the agent. Swapping Streamlit for anything else touches no reinforcement-learning code.

**It is testable without a browser.** `TestClient` exercises the whole request path —
validation, status codes, the missing-model case — in a few seconds and with no UI at all.

## Why the cockpit rebuilds the pictures locally

`POST /run` returns the trajectory: actions, rewards, the final state. It does **not** return
frames. A thousand frames of 600×400 RGB is roughly a gigabyte on the wire to draw an
animation the client can rebuild from the environment in a second.

That is only legitimate if the local replay is the same episode. It is checked, not assumed:
`rl_lander.replay.replay_matches` compares the replayed reward sequence against the one the
service returned, and the page says so when they differ. Without that check, an animation of
one episode could sit beside the metrics of another and nothing would notice.

## Why the dashboard reads files rather than the API

The dashboard describes a *published evaluation*, not a live one. Its inputs are the artefacts
under `data/` that `scripts/evaluate_and_export.py` writes, and those are versioned. A
dashboard that re-ran the evaluation on every page load would show a different number each
time, none of them the one the README quotes.

The schema of those exports lives in `rl_lander/artifacts.py`, outside the Streamlit module,
and is checked before the first panel is drawn. The earlier version indexed columns as it
rendered, so an export written by an older exporter failed halfway down the page — after the
reader had already seen three panels drawn from a fourth of the data.

## Liveness and readiness are two endpoints

`/health` answers "the process is up". `/ready` answers "a policy is loaded and I can
predict". An orchestrator acts differently on each: the first failing means restart, the
second failing means stop sending traffic. A single endpoint returning 200 with
`model_loaded: false` conflates them, and a load balancer reading the status code keeps
routing requests to a service that answers 503 to every one of them.

## What is deliberately not here

**A model registry.** Nine training runs and one shipped policy. `scripts/publish_run.py`
promotes exactly one run to `models/ppo/best.zip`, and the run's manifest travels with it.
MLflow would be scaffolding around a decision that is one command.

**A queue, a worker, a container orchestration.** The inference is a few milliseconds on the
CPU and the service is single-process. Anything more would be architecture for a load that
does not exist.

**A database.** The published artefacts are CSV and JSON, small enough to read in git and to
diff. That is a feature of the size, not a principle — it stops being true at the first run
that produces a million rows.
