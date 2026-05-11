# AstroDynamics Eagle-1 — RL autopilot

Reinforcement-learning autopilot for the Eagle-1 lunar lander, based on
[Gymnasium](https://gymnasium.farama.org/) `LunarLander-v3` and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

The repository is organised in line with the project brief
"Entrainez votre agent RL" and ships:

- a clean implementation of the three foundational exercises (CartPole random
  policy, FrozenLake Q-learning, CartPole DQN — manual + SB3),
- the Eagle-1 mission (LunarLander baseline, hyper-parameter optimisation,
  evaluation),
- a FastAPI inference service,
- a Streamlit GUI that animates an episode driven by the API,
- a Streamlit performance dashboard.

A full walk-through is provided in `notebooks/eagle1_mission.ipynb`.

## Quick start

```powershell
# Install Python and the dependencies in one go
uv sync --all-extras

# Run the training pipeline (LunarLander-v3, PPO)
uv run python -m astrodynamics.training.train_lunarlander --timesteps 1000000

# Serve the trained agent
uv run uvicorn astrodynamics.api:app --reload

# Launch the GUI / dashboard
uv run streamlit run src/astrodynamics/gui.py
uv run streamlit run src/astrodynamics/dashboard.py
```

See `docs/` for the design notes.
