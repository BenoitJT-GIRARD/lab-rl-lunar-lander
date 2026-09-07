# Lunar Lander

A PPO agent that lands Gymnasium's `LunarLander-v3`, served behind an API, with the number
it is judged on measured across five independent trainings rather than one.

![The trained policy landing, seed 6](docs/images/landing.gif)

**Project status** — finished, and archived in a runnable state. The CI is frozen to manual
trigger so that nothing here decays into a red badge on a project nobody maintains. Every
figure below is read from a file in `data/`, written by a script in `scripts/`.

## The problem

Landing the lander is the easy half. Any competent PPO configuration clears Gymnasium's
solved threshold of 200 in about a million steps, and this one does.

The hard half is the number you then publish. A reinforcement-learning score depends on the
training seed, on the evaluation episodes drawn to measure it, and on which checkpoint of
the run gets saved — and the usual practice is to report one run, on one grid, with a
standard deviation that describes the episodes rather than the method. Retrain the same
configuration and you get a different number. The published one was never about the
configuration at all.

So the question this repository is built around is **what a reported score is a claim
about, and how much of the spread around it comes from where.**

## What it does

A FastAPI service loads the shipped policy and plays episodes on request: `/play` for one
action, `/run` for a full trajectory with its rewards and its seed.

![The API surface at /docs](docs/images/api-docs.png)

`/health` and `/ready` are separate on purpose. One means restart the process, the other
means stop sending it traffic; a single endpoint returning 200 with `model_loaded: false`
conflates them, and a load balancer reading the status code keeps routing requests to a
service that answers 503 to all of them.

A Streamlit cockpit replays an episode the API ran.

![The cockpit, replaying an episode the API ran](docs/images/cockpit.png)

It asks the API for a trajectory and rebuilds the frames locally — a thousand 600×400
frames over HTTP would be a bill, not a design. That is only legitimate if the replay is
the same episode, so the page compares its replayed rewards against the API's and says so
when they differ.

A second Streamlit surface reads the published evaluation.

![The shipped policy's 100 evaluation episodes, behind the sidebar filters](docs/images/dashboard.png)

Alongside them, `src/rl_lander/exercises/` holds the three guided exercises the agent is
built on top of: a random policy, tabular Q-learning, and DQN.

## The result

Five PPO runs, identical hyper-parameters, seeds 42 to 46, one million steps each. Every
run scored the same way: 100 episodes on a fixed evaluation grid, through the same code the
exporter uses.

| Training seed | Mean reward | sd across episodes | Landed | Above 200 | Worst episode |
|---|---|---|---|---|---|
| 42 | 264.05 | 19.13 | 100% | 100% | 217.51 |
| 43 | 242.03 | 27.41 | 97% | 97% | 111.52 |
| 44 | 237.43 | 36.09 | 98% | 96% | **−33.38** |
| **45** *(shipped)* | 261.80 | 29.09 | 97% | 97% | 122.78 |
| 46 | 265.39 | 19.20 | 100% | 100% | 224.65 |

**254.14 ± 13.32 between runs**, worst 237.43, five out of five above the solved threshold
of 200. That is the transportable number: retrain this configuration and you land somewhere
in that range.

The shipped policy is **seed 45**, chosen because its mean is nearest the median of the
five, not because it is the best. `scripts/publish_run.py --from-study` makes that choice,
so it is a rule rather than a judgement made after seeing the scores.

Its own evaluation, on six independent seed grids of 100 episodes:

| Grid seed | Mean | sd | Worst episode | Landed |
|---|---|---|---|---|
| **2024** *(exported)* | 261.80 | 29.09 | 122.78 | 97% |
| 7 | 266.52 | 20.27 | 224.37 | 100% |
| 99 | 269.57 | 20.38 | 226.51 | 100% |
| 123 | 269.08 | 20.98 | 223.12 | 100% |
| 555 | 264.31 | 22.22 | 154.11 | 99% |
| 31337 | 262.34 | 23.64 | 135.41 | 99% |

**265.60 ± 3.33 across grids.** The mean is stable to about a point; the worst episode
moves from 123 to 227 and the landing rate from 97% to 100%. The exported grid is the
*least* flattering of the six, which is not a virtue — it is what happens when the grid is
chosen by code, in advance, instead of after the numbers are in.

Read the two spreads together and they answer the question at the top. **13.32 between
trainings, 3.33 between evaluation grids of the same policy.** The variance that matters is
the one the usual report leaves out.

### Against DQN, at an equal budget

| | Mean | sd | Landed | Worst episode | Mean episode length |
|---|---|---|---|---|---|
| PPO, mean of 5 runs | 254.14 | between runs 13.32 | 98.4% | — | — |
| PPO, shipped run | 261.80 | 29.09 | 97% | 122.78 | 322.6 |
| DQN, 1 run | **271.13** | 47.51 | 96% | 34.41 | 212.2 |

One million steps each, not the 200,000 DQN's own defaults suggest, because a comparison at
unequal budgets measures the budget. The DQN scores higher on the mean and is worse
everywhere else that matters: its episode spread is 47.5 against 19–36 for the PPO runs,
its worst episode is 34.4 against 122.8, and it is **one seed against five**. What can be
said is that DQN is competitive on this task at this budget, and that nothing here supports
ranking the two.

### What the hyper-parameters are worth

One parameter changed at a time, same budget, same evaluation protocol.

| Trial | Measured | Δ vs baseline | Larger than the seed spread? |
|---|---|---|---|
| baseline (5 seeds) | **254.14 ± 13.32** | — | — |
| `learning_rate=1e-3` | 267.73 | +13.6 | just barely |
| `n_steps=2048` | 210.06 | −44.1 | yes |
| `gamma=0.99` | 173.13 | −81.0 | yes |

The last column is what makes the table readable. Each trial is a **single** seed, and the
baseline's own spread between seeds is 13.32. A difference smaller than that is one draw
from the same distribution, not evidence about the parameter. So `n_steps=2048` and
`gamma=0.99` are real effects — `gamma=0.99` does not clear the threshold at all, at 173.13
with a 73% landing rate — while `lr=1e-3` clears the spread by three tenths of a point and
establishes nothing. Reading a winner out of that row would repeat, one level up, the error
this study exists to correct.

## Why these numbers can be believed

They are not the figures this repository published first. It used to report `262.2 ± 18.2` —
one run, on its most favourable evaluation grid — as if the ± described the method. Seven
things were wrong, all seven in what was *published* rather than in what was computed. Each
is given with the number before and the number after, because a correction nobody can see
is half a correction.

**The file called `best` held the last model.** `EvalCallback` wrote its best checkpoint to
`models/best_model.zip`. Then `model.save(output)` wrote the state training *ended* on,
under the name `ppo_lunarlander_best.zip` — the only tracked file, and the source of every
published figure. In reinforcement learning those are not the same policy: performance
oscillates late in training, which is why `EvalCallback` exists at all. Both algorithms
pointed at that one directory, so a DQN run overwrote a PPO one, and `--output` defaulted to
the PPO path whatever `--algo` said, so it did not even take a mistake.

The DQN run puts a price on it. Both checkpoints were kept, so both can be scored on the
same grid:

| Run | `best.zip` | `final.zip` |
|---|---|---|
| PPO, seed 45 | 261.80 | 268.14 |
| DQN, seed 42 | **271.13** | **−610.34** |

The DQN policy at the end of training crashes every episode — 0% landing, worst episode
−2495. Its best checkpoint lands 96%. On that run the repository would have published a
policy that never lands, and its own dashboard would have shown it. For PPO the same bug
costs about six points in the other direction, which is exactly why it survived: on the
algorithm that was shipped, it was almost free.

There is now one directory per run, `models/<algo>/seed-<n>/`, and
`scripts/publish_run.py` is the only path from a run to the shipped policy — it refuses a
run whose best *is* its final, one no evaluation ever improved on.

**`landed` meant "scored more than 200".** `evaluate.py` set `landed = total_reward >= 200`.
200 is the threshold at which Gymnasium considers the *environment solved on average* — a
property of the task over many episodes, saying nothing about any single one. The
environment answers the question directly: on termination it assigns exactly **+100** for
coming to rest and **−100** for crashing or leaving the frame. That is what is read now,
with the score kept as a separate column. On the shipped policy the two agree — 97% and 97%
— which is exactly why the confusion was invisible. Across the five trainings they do not:
**seed 44 lands 98% of its episodes and clears 200 on 96%.**

**Two official means, in the same file.** `evaluation_summary.json` carried `261.394` at its
root and `262.218` under `metrics`, with standard deviations 44% apart — two evaluation
loops with different reset semantics, both published, neither designated as the result. Not
a calculation error: an absent decision. There is now one collection, and every row carries
the seed that replays it.

**One evaluation grid, published as the performance.** The old export ran a single grid,
seed 2024, and did not say so. Re-running the policy shipped at the time on six grids showed
its published standard deviation, 18.2, was the lowest of the six against a median of 23.3,
and that its 100% landing rate was a property of that grid — the worst episode elsewhere was
63.0 against the 222.5 on record. The mean survived that check and still does; what changed
is what is published beside it.

**One training run, published as the method.** Everything above concerns the variance of
evaluation conditions with the model held fixed. The variance that matters in reinforcement
learning — between two trainings identical but for the seed — was not measured at all. It is
the first table above: 237.43 to 265.39, a 28-point range against the single ±18.2 the
repository offered as its only uncertainty.

**A hyper-parameter study that did not exist.** The notebook carried a table — `~280`,
`< 200`, `~270`, `~240` — and a conclusion drawn from it. None of those numbers existed
anywhere else in the repository, and the evidence it cited, TensorBoard logs, is ignored by
git. Measured, not one row was right, and the ranking is inverted: the table put the
baseline first and `lr=1e-3` last, where the measurement puts `lr=1e-3` first and the
baseline second.

**The DQN baseline was announced and never delivered.** `models/` was described as holding
"best PPO / baseline DQN". It held only the PPO, and the notebook discussed a DQN baseline
as though it existed. It is trained and scored above.

### What the tests assert

Not that the code runs — that the published claims are true:

- the exported CSV and the published summary are **one** collection: same episode count,
  same mean, same standard deviation, same landing rate;
- every exported row carries the seed that replays it, and those seeds are the grid the
  summary names;
- the manifest points at a policy that exists in the repository, by a relative path;
- the solved-threshold claim holds on **every** evaluation grid, not only the exported one;
- `scripts/evaluate_and_export.py` exits non-zero when a grid falls short, so a claim cannot
  outlive the run behind it.

Warnings are errors (`filterwarnings = ["error"]`), with three exemptions that each name the
message they silence and come from a dependency. That paid for itself on the first run:
Stable-Baselines3 had been saying all along that PPO with an MLP policy belongs on the CPU,
and the old configuration ignored `UserWarning` wholesale. Measured here: **50,000 steps in
14.9 s on the CPU against 20.3 s on an RTX 4060 Ti**. DQN is the other way round — 20,000
steps in 41.8 s on the GPU against 64.0 s — because it replays batches of 128 through two
256-unit layers, which is enough work to pay for the transfers. Each algorithm now names its
device, with the measurement in the comment beside it.

One defect no linter or warning sees: these hyper-parameter classes use `slots=True`, so
`PPOHyperParameters.total_timesteps` is the slot descriptor and not the default value. The
CLI used it as a fallback when `--timesteps` was absent, and the descriptor travelled
through the constructor without complaint before failing inside Stable-Baselines3's loop,
several minutes into a run, on a comparison between an int and a descriptor. It is pinned by
a test now.

## Running it

```powershell
uv sync --all-extras
```

Go straight to the surfaces, on what is committed:

```powershell
uv run uvicorn rl_lander.api:app                        # http://127.0.0.1:8000/docs
uv run streamlit run src/rl_lander/gui.py               # one episode, animated
uv run streamlit run src/rl_lander/dashboard.py         # the evaluation run, filtered
uv run python -m rl_lander.record_video                 # videos/landing.mp4
```

Or reproduce the study — nine trainings, resumable, about three hours on one machine:
roughly 14 minutes per PPO run on an idle CPU, 40 for the DQN.

```powershell
uv run python scripts/train_study.py              # 5 PPO seeds, 3 trials, 1 DQN at equal budget
uv run python scripts/aggregate_study.py          # the three artefacts under data/
uv run python scripts/publish_run.py --from-study # promotes the median run
uv run python scripts/evaluate_and_export.py      # re-scores it, and states a verdict
```

The reasoning behind one service and thin frontends is in
[`docs/architecture.md`](docs/architecture.md). The full walk-through, exercises included, is
[`notebooks/lunar_lander.ipynb`](notebooks/lunar_lander.ipynb) — committed without outputs,
and reading every figure from `data/` when it runs.

## Structure

```
├── data/                         # everything published
│   ├── evaluation_episodes.csv   #   the canonical collection, one row per episode
│   ├── evaluation_summary.json   #   its metrics, and the spread across evaluation grids
│   ├── evaluation_manifest.json  #   model, seeds, package versions, git revision
│   ├── seed_study.json           #   the five trainings, and the DQN at equal budget
│   ├── hyperparameter_trials.csv #   one row per single-parameter trial
│   ├── learning_curve_band.csv   #   median and interquartile range across the five
│   └── training_curves.csv       #   the shipped run's own curve
├── docs/architecture.md          # why a service, and where the boundaries are
├── models/ppo/best.zip           # the shipped policy; per-run directories are untracked
├── notebooks/lunar_lander.ipynb  # the work in order
├── scripts/                      # train_study, aggregate_study, publish_run, evaluate_and_export
├── src/rl_lander/
│   ├── agent.py                  # the only place an action is chosen and a landing judged
│   ├── api.py                    # FastAPI: /play /run /reset /info /health /ready
│   ├── artifacts.py              # the schema of the published exports
│   ├── replay.py                 # re-run a trajectory, and check it is the same episode
│   ├── dashboard.py, gui.py      # Streamlit surfaces (optional `ui` extra)
│   ├── record_video.py           # the clip, and metadata that matches its frames
│   ├── exercises/                # the three guided exercises
│   ├── training/                 # environments, hyper-parameters, callbacks, evaluation
│   └── utils/                    # paths, seeding
└── tests/                        # the published artefacts are among the assertions
```

## What this does not prove

**Five seeds is few.** Enough to show the training seed matters and to publish a spread; not
enough to put a confidence interval on that spread. Twenty would be the right number, and
the conclusion — publish the method's dispersion, not one run's score — does not change past
five.

**One seed per hyper-parameter trial.** The trials answer "is that table real", not "which
value is best".

**One hundred evaluation episodes.** That is the horizon this repository fixed, and the
across-grid table shows what it costs: the mean is stable, the tails are not. For a landing
autopilot the number that would actually matter is the worst case, and 100 episodes says
very little about it — seed 44's worst episode is a crash at −33.

**The task is nearly saturated, and that is the real limit.** The shipped policy clears the
threshold on every grid and lands 97 to 100% of episodes. There is little headroom left in
which one configuration can distinguish itself from another, which is the honest reason the
`lr=1e-3` trial cannot be called an improvement: not that the learning rate does not matter,
but that at this budget on this environment its effect is the size of the noise between
seeds. A harder variant — `enable_wind=True`, or a stochastic initial state — would make the
same comparison informative, and that is the experiment this repository is one step away
from.

## Licence and data

MIT, for the code.

There is no dataset. The environment is `LunarLander-v3` from
[Gymnasium](https://gymnasium.farama.org/) (MIT), simulated by Box2D (zlib); every episode
in `data/` was generated locally by running it, and every row carries the seed that
reproduces it. Nothing here is derived from a third-party corpus, so there is nothing to
attribute and nothing whose redistribution needs checking.
