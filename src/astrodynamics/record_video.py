"""Record a 20-30 s video of the trained autopilot landing the Eagle-1.

A typical LunarLander-v3 episode lasts ~7 s at the renderer's native
50 FPS; a single rollout therefore does not fill the 20-30 s envelope
required by the brief.  The recorder concatenates several successful
landings (with a short fade-to-black between each) until the cumulative
duration falls within the target window.  This produces a continuously
informative clip rather than a static title screen padded with empty
frames.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from astrodynamics.agent import LunarLanderAgent
from astrodynamics.utils import DEFAULT_MODEL_PATH, VIDEOS_DIR, ensure_dirs


def _ensure_block_size(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Pad frame width to a multiple of 16 to please H.264 encoders."""
    if not frames:
        return frames
    height, width, channels = frames[0].shape
    padded_width = ((width + 15) // 16) * 16
    if padded_width == width:
        return frames
    out = []
    for frame in frames:
        canvas = np.zeros((height, padded_width, channels), dtype=frame.dtype)
        canvas[:, :width, :] = frame
        out.append(canvas)
    return out


def _fade_separator(reference: np.ndarray, length: int = 8) -> list[np.ndarray]:
    """A short fade-to-black used between concatenated episodes."""
    blank = np.zeros_like(reference)
    return [blank] * length


def record_landing(
    model_path: Path = DEFAULT_MODEL_PATH,
    output: Path = VIDEOS_DIR / "eagle1_landing.mp4",
    max_attempts: int = 16,
    target_min_seconds: float = 22.0,
    target_max_seconds: float = 30.0,
    fps: int = 50,
    seed_start: int = 0,
    min_landings: int = 2,
) -> dict:
    """Record a video of one or several successful Eagle-1 landings."""
    ensure_dirs()
    output.parent.mkdir(parents=True, exist_ok=True)
    agent = LunarLanderAgent(model_path)

    min_frames = int(target_min_seconds * fps)
    max_frames = int(target_max_seconds * fps)

    accepted: list[tuple[int, float, list[np.ndarray]]] = []
    best_single: tuple[float, list[np.ndarray], dict] | None = None
    for attempt in range(max_attempts):
        seed = seed_start + attempt
        result, frames = agent.play_episode(seed=seed, render_mode="rgb_array")
        if result.landed:
            accepted.append((seed, result.total_reward, frames))
        meta = {"seed": seed, **asdict(result)}
        if best_single is None or result.total_reward > best_single[0]:
            best_single = (result.total_reward, frames, meta)

        merged = []
        merged_length = 0
        for _, _reward, ep_frames in sorted(accepted, key=lambda item: -item[1])[
            : min_landings * 3
        ]:
            if merged_length > 0:
                merged.extend(_fade_separator(ep_frames[0]))
            merged.extend(ep_frames)
            merged_length = len(merged)
            if merged_length >= min_frames:
                break
        if min_frames <= len(merged) <= max_frames and len(accepted) >= min_landings:
            sequence = merged
            chosen_seeds = [seed for seed, _, _ in accepted[:min_landings]]
            mean_reward = float(np.mean([reward for _, reward, _ in accepted[:min_landings]]))
            sequence = _ensure_block_size(sequence)
            imageio.mimsave(str(output), sequence, fps=fps, codec="libx264", quality=8)
            return {
                "output": str(output),
                "n_landings": len(chosen_seeds),
                "seeds": chosen_seeds,
                "mean_reward": mean_reward,
                "frames": len(sequence),
                "seconds": len(sequence) / fps,
            }

    # Fallback: render the best single episode, padded only if necessary.
    assert best_single is not None
    reward, frames, meta = best_single
    if len(frames) > max_frames:
        frames = frames[-max_frames:]
    elif len(frames) < min_frames:
        pad = [frames[0]] * (min_frames - len(frames))
        frames = pad + frames
    frames = _ensure_block_size(frames)
    imageio.mimsave(str(output), frames, fps=fps, codec="libx264", quality=8)
    return {
        "output": str(output),
        "n_landings": 1,
        "seeds": [meta["seed"]],
        "mean_reward": float(reward),
        "frames": len(frames),
        "seconds": len(frames) / fps,
        "fallback": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Eagle-1 landing video(s)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=VIDEOS_DIR / "eagle1_landing.mp4")
    parser.add_argument("--max-attempts", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--min-landings", type=int, default=2)
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI helper
    args = _parse_args()
    info = record_landing(
        model_path=args.model,
        output=args.output,
        max_attempts=args.max_attempts,
        seed_start=args.seed_start,
        min_landings=args.min_landings,
    )
    print(info)


if __name__ == "__main__":  # pragma: no cover
    main()
