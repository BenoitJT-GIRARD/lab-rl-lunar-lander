# AstroDynamics Eagle-1 — RL autopilot

Reinforcement-learning autopilot for the Eagle-1 lunar lander, based on
[Gymnasium](https://gymnasium.farama.org/) `LunarLander-v3` and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

The repository delivers a full project to practice training RL agents:

- a clean implementation of the three foundational exercises (CartPole
  random policy, FrozenLake tabular Q-learning, CartPole DQN — manual
  PyTorch + Stable-Baselines3),
- the Eagle-1 mission (LunarLander baseline, hyper-parameter
  optimisation, 100-episode evaluation `> 200` mean reward),
- a FastAPI inference service (`/play`, `/run`, `/reset`, `/info`,
  `/health`),
- a Streamlit GUI that animates an episode driven by the API,
- an interactive Streamlit performance dashboard with multiple filters,
- a 20-30 s `.mp4` of a successful landing.

A full walk-through is provided in `notebooks/eagle1_mission.ipynb`.

## Repository layout

```
astrodynamics/
├── data/                    # CSV exports consumed by the dashboard
├── logs/                    # TensorBoard runs (gitignored)
├── models/                  # Saved checkpoints (best PPO / baseline DQN)
├── notebooks/
│   └── eagle1_mission.ipynb # End-to-end walk-through
├── scripts/
│   └── evaluate_and_export.py
├── src/astrodynamics/
│   ├── agent.py             # SB3 inference wrapper
│   ├── api.py               # FastAPI service
│   ├── dashboard.py         # Streamlit dashboard
│   ├── gui.py               # Streamlit cockpit
│   ├── record_video.py      # 20-30 s mp4 generator
│   ├── exercises/           # Exercises 1-3
│   ├── training/            # Mission-specific training pipeline
│   └── utils/               # Shared helpers (paths, seeding)
├── tests/                   # Pytest suite
├── videos/                  # Generated landing videos
├── pyproject.toml           # uv-managed deps + ruff/bandit/pytest
├── .pre-commit-config.yaml  # ruff, bandit, nbstripout
└── README.md
```

## Quick start

```powershell
# 1. Install Python 3.12 and project dependencies in one go
uv sync --all-extras

# 2. Train PPO on LunarLander-v3 (≈ 12-18 min on RTX 4060 Ti)
uv run python -m astrodynamics.training.train_lunarlander `
    --algo ppo --timesteps 1000000 --n-envs 16 `
    --output models/ppo_lunarlander_best.zip

# 3. Run the official 100-episode evaluation and write CSV exports
uv run python scripts/evaluate_and_export.py

# 4. Record a landing video (20-30 s, libx264)
uv run python -m astrodynamics.record_video --output videos/eagle1_landing.mp4

# 5. Serve the trained agent through the FastAPI service
uv run uvicorn astrodynamics.api:app --reload

# 6. Launch the GUI / dashboard (point the GUI at the API URL)
uv run streamlit run src/astrodynamics/gui.py
uv run streamlit run src/astrodynamics/dashboard.py
uv run tensorboard --logdir logs/tensorboard
```

## Reproducibility

* `uv.lock` pins every dependency including the CUDA 12.4 build of
  PyTorch.
* `astrodynamics.utils.set_global_seed` seeds Python, NumPy and PyTorch
  (CPU + CUDA) for both training and evaluation.
* The training callbacks persist the rolling reward statistics to
  `data/training_curves.csv` and full TensorBoard logs to
  `logs/tensorboard/`.

## Quality gates

| Tool | Configuration | Command |
| ---- | ------------- | ------- |
| Ruff | `pyproject.toml` (`E,W,F,I,B,C4,UP,N,SIM,RUF`) | `uv run ruff check src tests` |
| Bandit | `pyproject.toml` (`skips=B101,B311`) | `uv run bandit -c pyproject.toml -r src` |
| Pytest | `pyproject.toml` | `uv run pytest tests` |
| Pre-commit | `.pre-commit-config.yaml` | `uv run pre-commit run --all-files` |

The pre-commit pipeline includes `ruff --fix`, `ruff-format`, `bandit`
and `nbstripout` (notebooks are committed without execution outputs).

## License

MIT.
