# The evaluation protocol

What each published number is a claim about, how it was measured, and what the `±` beside it
covers. `metrics.yaml` defines every label a table uses; this page says how the tables were
obtained.

Three things move a reinforcement-learning score, and a report that fixes two of them and
varies the third silently is a report about nothing. Everything below exists to say which of
the three a given `±` describes.

## What is fixed, and what varies

| Held fixed | Varied, and measured |
|---|---|
| The environment: `LunarLander-v3`, no wind, default gravity | The **training seed**: 42 to 46, five independent runs |
| The budget: 1 000 000 steps per run | The **evaluation grid**: six seeds, 100 episodes each |
| The hyper-parameters, except in the trials | One **hyper-parameter at a time**, in three trials |
| The scoring code: one `evaluate_seeds`, used by every table | The **algorithm**, once: DQN at the same budget |

Every episode is replayable: each row of `reports/evaluation_episodes.csv` carries the seed
that produced it, and `reports/evaluation_manifest.json` carries the policy, the package
versions and the git revision the scores were taken at.

## The three dispersions, and which is which

Every `±` in this repository is a **standard deviation**, computed over a population the
sentence beside it names. A column that disperses one thing on one row and another on the next
teaches a reader to ignore it. Three populations appear here, and they are never mixed:

| Written | What it disperses | Where it is read |
|---|---|---|
| `sd across episodes` | the 100 episodes of one grid, one policy | a row of the per-seed table |
| `between runs` | the five training seeds, each already averaged | the headline score |
| `across grids` | the six evaluation grids, one policy | the stability of the shipped policy |

**The headline number is the second one.** It is the only one that answers « retrain this
configuration and what do you get », which is the question a reader has. The first describes
episodes and says nothing about the method; the earlier version of this repository published
it as if it did.

## How a run becomes the shipped policy

1. `scripts/train_study.py` trains the five seeds, the three trials and the DQN. Each run
   writes its checkpoints and its curve to its own directory under `var/runs/`.
2. Each run keeps **two** checkpoints under different names, and the header of
   `scripts/compare_checkpoints.py` says why the distinction is expensive. On the DQN run
   they differ by 880 points.
3. `scripts/aggregate_study.py` summarises the study into `reports/seed_study.json`,
   `reports/hyperparameter_trials.csv` and `reports/learning_curve_band.csv`.
4. `scripts/publish_run.py --from-study` promotes one run by a rule written before the
   scores were seen, and its docstring states the rule and the tie-break. It also refuses a
   run that never improved on its own starting point.
5. `scripts/evaluate_and_export.py` re-scores the promoted policy on the six grids and
   fails the command when one of them falls short. A published claim cannot outlive the run
   behind it.

## What `landed` means

The environment's terminal reward, and not the score. `LunarLander-v3` pays one hundred for
coming to rest and takes one hundred for a crash or a departure from the frame, and that sign
is what `landed` reads. The score is kept as a separate column.

The 200 Gymnasium calls solved describes the task over many episodes and settles nothing about
one of them; the docstring of `rl_lander.api` makes the same point where it refuses an
impossible observation. Reading `landed` from it looked right on the
shipped policy, where the landing rate and the threshold rate are both 0.97, and was wrong on
seed 44, whose landing rate is 0.98 and whose threshold rate is 0.96.

## Reading the hyper-parameter trials

Each trial changes one parameter, trains one seed at the same budget, and is scored by the
same code. One seed is not enough to rank a parameter, and the table says so in its last
column: a difference smaller than the baseline's own spread between seeds is one draw from
the same distribution.

That column is the whole point of the table. Without it, the reader ranks four numbers and
repeats, one level up, the error this repository exists to correct.

## What this protocol does not give

**A confidence interval on the dispersion.** Five seeds are enough to show that the training
seed matters and to publish a spread; they are not enough to bound that spread. Twenty would
be the right number, and the conclusion does not change past five.

**A worst-case guarantee.** One hundred episodes per grid pins the mean and says very little
about the tail, which for a landing autopilot is the number that would actually matter. The
across-grid table is where that shows: the mean moves by a point, the worst episode moves
from 123 to 227.

**A ranking between PPO and DQN.** One DQN run against five PPO runs answers « is DQN
competitive here », not « which is better ». `docs/architecture.md` says why the comparison
is run at an equal budget rather than at each algorithm's own defaults.
