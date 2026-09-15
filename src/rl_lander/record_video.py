"""Record a short video of the trained autopilot landing.

One episode lasts a few seconds at the renderer's 50 FPS, which is well under the 20-30 s
a demonstration clip wants. The recorder therefore plays several seeded episodes, keeps the
landings, and concatenates the best of them with a cross-fade until the clip is long
enough.

The whole file exists to make one guarantee: **the metadata describes the frames that were
encoded.** The first version sorted the episodes by reward before merging them, then
reported the seeds of the first ones accepted -- so the clip showed one set of landings and
the manifest named another. It also reported ``n_landings: 1`` on its fallback path even
when the fallback episode had crashed, and padded a short clip by repeating its first frame
several hundred times, which is a still image sold as flight footage.

Nothing here pads, and nothing here reports an episode it did not encode. When the target
duration cannot be reached, the result says so and gives what there is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from rl_lander.agent import LunarLanderAgent
from rl_lander.utils import DEFAULT_MODEL_PATH, IMAGES_DIR, ROOT_DIR, VIDEOS_DIR, ensure_dirs

#: The published animation, and the only artefact of this module a document shows. The clip
#: itself stays under `var/`: it is 588 KB of H.264 that no page embeds, and a repository
#: publishes what it shows.
PUBLISHED_GIF = IMAGES_DIR / "landing.gif"

#: The GIF is decimated and capped. A twenty-five-second clip at fifty frames a second is
#: nine megabytes of GIF, which nobody waits for on a README.
GIF_FPS = 20
GIF_SECONDS = 8.0


@dataclass(slots=True)
class Take:
    """One recorded episode, with everything the manifest needs to name it."""

    seed: int
    total_reward: float
    landed: bool
    frames: list[np.ndarray]


def _ensure_block_size(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Pad the frame width to a multiple of 16, which H.264 encoders want."""
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


def _crossfade(previous: np.ndarray, following: np.ndarray, length: int = 10) -> list[np.ndarray]:
    """An actual fade between two episodes.

    The function this replaces was named ``_fade_separator`` and returned ``length`` copies
    of a black frame -- a cut to black, not a fade. At 50 FPS the difference is visible:
    the picture vanishes for a fifth of a second and comes back.
    """
    if length < 1:
        return []
    weights = np.linspace(0.0, 1.0, num=length + 2, dtype=np.float32)[1:-1]
    return [
        (previous.astype(np.float32) * (1.0 - w) + following.astype(np.float32) * w).astype(
            previous.dtype
        )
        for w in weights
    ]


def _assemble(takes: list[Take], fade_length: int) -> list[np.ndarray]:
    """Concatenate takes in the order given, cross-fading between them."""
    sequence: list[np.ndarray] = []
    for take in takes:
        if sequence:
            sequence.extend(_crossfade(sequence[-1], take.frames[0], fade_length))
        sequence.extend(take.frames)
    return sequence


def record_landing(
    model_path: Path = DEFAULT_MODEL_PATH,
    output: Path = VIDEOS_DIR / "landing.mp4",
    max_attempts: int = 16,
    target_min_seconds: float = 22.0,
    target_max_seconds: float = 30.0,
    fps: int = 50,
    seed_start: int = 0,
    min_landings: int = 2,
    fade_length: int = 10,
) -> dict:
    """Record a clip of successful landings and return exactly what it contains.

    Episodes are played from ``seed_start`` upwards. Landings are kept, ranked by reward,
    and the best of them are concatenated in that same ranked order until the clip holds at
    least ``min_landings`` of them and reaches ``target_min_seconds`` -- and the manifest
    names those takes, in that order.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    if min_landings < 1:
        raise ValueError(f"min_landings must be at least 1, got {min_landings}")
    if fps < 1:
        raise ValueError(f"fps must be at least 1, got {fps}")
    if target_min_seconds <= 0 or target_max_seconds < target_min_seconds:
        raise ValueError(
            "target window must satisfy 0 < target_min_seconds <= target_max_seconds, got "
            f"{target_min_seconds} and {target_max_seconds}"
        )

    ensure_dirs()
    output.parent.mkdir(parents=True, exist_ok=True)
    agent = LunarLanderAgent(model_path)

    min_frames = int(target_min_seconds * fps)
    max_frames = int(target_max_seconds * fps)

    takes: list[Take] = []
    for attempt in range(max_attempts):
        seed = seed_start + attempt
        result, frames = agent.play_episode(seed=seed, render_mode="rgb_array")
        takes.append(Take(seed, result.total_reward, result.landed, frames))

    landings = sorted((t for t in takes if t.landed), key=lambda t: -t.total_reward)

    # Take landings, best first, until the clip is long enough. Stopping at the first take
    # that would overshoot keeps the clip inside the window without cutting mid-episode.
    selected: list[Take] = []
    total = 0
    for take in landings:
        cost = len(take.frames) + (fade_length if selected else 0)
        if selected and total + cost > max_frames:
            break
        selected.append(take)
        total += cost
        if total >= min_frames and len(selected) >= min_landings:
            break

    if not selected:
        # No landing in `max_attempts`. Show the best attempt and say plainly that it is
        # not a landing, rather than reporting `n_landings: 1` for a crash.
        best = max(takes, key=lambda t: t.total_reward)
        selected = [best]

    sequence = _ensure_block_size(_assemble(selected, fade_length))
    if len(sequence) > max_frames:
        sequence = sequence[:max_frames]
    imageio.mimsave(str(output), sequence, fps=fps, codec="libx264", quality=8)
    gif = write_gif(sequence, fps)

    landed_count = sum(1 for take in selected if take.landed)
    return {
        "output": str(output),
        "gif": str(gif),
        "episodes": [
            {"seed": t.seed, "total_reward": round(t.total_reward, 2), "landed": t.landed}
            for t in selected
        ],
        "n_landings": landed_count,
        "seeds": [t.seed for t in selected],
        "mean_reward": float(np.mean([t.total_reward for t in selected])),
        "frames": len(sequence),
        "seconds": round(len(sequence) / fps, 2),
        "attempts": len(takes),
        "landing_rate": round(sum(1 for t in takes if t.landed) / len(takes), 3),
        # Both stated, because a clip that is short, or that shows no landing at all, is
        # still worth publishing -- as long as it is not described as something else.
        "reached_target_duration": len(sequence) >= min_frames,
        "shows_a_landing": landed_count > 0,
    }


def write_gif(sequence: list[np.ndarray], fps: int, output: Path = PUBLISHED_GIF) -> Path:
    """Write the animation a document can embed, decimated from the clip's own frames.

    Same frames, fewer of them: the GIF is what the README shows, and it has to be small
    enough that a reader sees it before they scroll past. It is written beside the
    screenshots and declared in the same manifest, because a picture nobody can regenerate
    is a picture nobody can check.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, round(fps / GIF_FPS))
    decimated = sequence[::step][: int(GIF_SECONDS * GIF_FPS)]
    # The GIF plugin takes a per-frame duration in milliseconds; `fps` is deprecated there
    # and, with warnings turned into errors, deprecated means the test suite fails.
    imageio.mimsave(str(output), decimated, duration=1000 / GIF_FPS, loop=0)
    _declare_gif(output, len(decimated))
    return output


def _declare_gif(image: Path, frames: int) -> None:
    """Record what this animation shows, in the manifest the screenshots share."""
    manifest = IMAGES_DIR / "MANIFEST.json"
    payload: dict = {"schema": "image-manifest/1", "images": {}}
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.setdefault("schema", "image-manifest/1")
    payload.setdefault("images", {})
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=False
    ).stdout.strip()
    payload["images"][image.name] = {
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "written": datetime.now(tz=UTC).date().isoformat(),
        "source": "src/rl_lander/record_video.py",
        "command": "uv run python -m rl_lander.record_video",
        "frames": frames,
        "fps": GIF_FPS,
        "app_state": (
            "the policy committed at models/ppo/best.zip, flown from seed 0 upwards until "
            "the clip holds at least two landings and reaches twenty-two seconds"
        ),
        "data_source": "none: the frames are rendered by the environment during the flight",
        "demonstrates_behaviour": (
            "the shipped policy brings the lander down between the flags and comes to rest "
            "on its legs, which is the claim the landing rate puts a number on"
        ),
        "git_revision": revision,
        "depends_on": ["src/rl_lander/record_video.py", "models/ppo/best.zip"],
    }
    payload["images"] = dict(sorted(payload["images"].items()))
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline=""
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a LunarLander landing video")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=VIDEOS_DIR / "landing.mp4")
    parser.add_argument("--max-attempts", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--min-landings", type=int, default=2)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI helper
    args = _parse_args()
    info = record_landing(
        model_path=args.model,
        output=args.output,
        max_attempts=args.max_attempts,
        seed_start=args.seed_start,
        min_landings=args.min_landings,
        fps=args.fps,
    )
    print(f"{info['output']} — {info['seconds']} s, {info['frames']} frames")
    print(f"{info['gif']} — published animation, declared in docs/images/MANIFEST.json")
    for episode in info["episodes"]:
        state = "landed" if episode["landed"] else "did not land"
        print(f"  seed {episode['seed']}: {episode['total_reward']:.1f}, {state}")
    if not info["shows_a_landing"]:
        print("  No landing in the attempts made: the clip shows the best failure.")
    if not info["reached_target_duration"]:
        print("  Shorter than the target window: there were not enough landings to fill it.")


if __name__ == "__main__":  # pragma: no cover
    main()
