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

**It is testable without a browser.** `TestClient` exercises the whole request path, from
validation to status codes to the missing-model case, in a few seconds and with no UI at all.

## Why the cockpit rebuilds the pictures locally

`POST /run` returns the trajectory: actions, rewards, the final state. It does **not** return
frames. A thousand frames of 600×400 RGB is roughly a gigabyte on the wire to draw an
animation the client can rebuild from the environment in a second.

It is legitimate only while the two are the same episode, so that is checked rather than
assumed. The comparison lives in `rl_lander.replay`, whose header says what it compares and
what a mismatch means, and the cockpit turns red on the answer.

## Why the dashboard reads files rather than the API

The dashboard describes a *published evaluation*, not a live one. Its inputs are the artefacts
under `data/` that `scripts/evaluate_and_export.py` writes, and those are versioned. A
dashboard that re-ran the evaluation on every page load would show a different number each
time, none of them the one the README quotes.

The schema of those exports lives in `rl_lander/artifacts.py`, outside the Streamlit module,
and is checked before the first panel is drawn. The earlier version indexed columns as it
rendered, and `tests/unit/test_artifacts_schema.py` pins each of the ways that went wrong.

## Liveness and readiness are two endpoints

`/health` answers "the process is up". `/ready` answers "a policy is loaded and I can
predict". An orchestrator acts differently on each: the first failing means restart, the second
failing means stop sending traffic. Merging them is the mistake the docstring of `ready` in
`rl_lander/api.py` describes, with what a load balancer then does about it.

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
