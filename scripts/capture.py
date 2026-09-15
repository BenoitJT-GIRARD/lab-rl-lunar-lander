"""Take this repository's screenshots, and write down what each one shows.

A screenshot is the only image of a repository that cannot be regenerated from data. What
makes it checkable is the sentence next to it: which build was running, what had already
happened to the application, where the displayed data came from. This script takes the picture
and writes that sentence into ``docs/images/MANIFEST.json`` in the same gesture, because a
manifest filled in afterwards is filled in from memory.

The repository fills in :data:`CAPTURES` and :func:`prepare`, and nothing else. Each entry says
what the image must *prove*: « the API answers a prediction » names a surface, « a request with
a missing feature is refused with the field named » names a behaviour; :func:`prepare` holds the
commands that put the product into the state being photographed.

    uv run python scripts/capture.py                 # every capture
    uv run python scripts/capture.py --only api-docs
    uv run python scripts/capture.py --check         # take nothing, report what is stale

Two engines. Chrome headless is enough for a page that renders server-side or in one pass —
Swagger, Airflow, a static report — as long as ``--virtual-time-budget`` is given, without
which the picture is taken before the JavaScript has drawn anything. It is **not** enough for
a page whose content arrives over a websocket: a Streamlit page loads an empty skeleton and
fills it afterwards, and the virtual clock advances timers without waiting for that round
trip, so the capture comes out black however long the budget. Those pages go through
``playwright``, which waits for a selector that only exists once the content is there.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from rl_lander.utils.paths import IMAGES_DIR, ROOT_DIR, SRC_DIR

#: Logical size of every capture, and the density it is rendered at. One size for the whole
#: portfolio: a reader comparing two repositories compares two images of the same shape.
VIEWPORT = (1280, 1000)
DEVICE_SCALE_FACTOR = 2

MANIFEST = IMAGES_DIR / "MANIFEST.json"
SOURCE = "scripts/capture.py"


@dataclass(frozen=True)
class Capture:
    """One image, and everything a reader needs to believe it.

    ``app_state`` is described precisely enough to be reproduced: « after 300 scoring
    requests, two of them refused », and never « with data ». ``data_source`` names where what
    is displayed comes from; a capture never redistributes someone else's work, so a
    third-party source is refused outright.
    """

    name: str
    route: str
    app_state: str
    demonstrates_behaviour: str
    data_source: str
    engine: str = "chrome"
    #: A selector that exists only once the content has arrived. Required by the playwright
    #: engine, ignored by Chrome.
    ready_selector: str | None = None
    #: What to do before the picture, in order. Each step is ``(action, target, value)``:
    #: ``("click", role, accessible name)``, ``("open", css selector, "")``,
    #: ``("fill", css selector, text)``,
    #: ``("scroll", css selector, "")``. A capture of an answer needs the three: click the
    #: control, type a real question, bring the response into the frame.
    steps: tuple[tuple[str, str, str], ...] = ()
    #: The twin image, when the product has a nominal and a degraded behaviour. A refusal
    #: shown alone reads as a failure; shown next to the nominal answer it reads as a design.
    paired_with: str | None = None
    depends_on: tuple[str, ...] = ()
    #: A surface of this repository that is NOT the one ``SERVE_COMMAND`` starts — an
    #: orchestrator's web interface, a database console, a second service of the same
    #: compose file. ``served_by`` is the command a reader runs to bring it up, and the
    #: capture is skipped, loudly, when nothing answers there: a picture is worth taking
    #: only of something that is running.
    base_url: str | None = None
    served_by: str | None = None

    @property
    def target(self) -> str:
        return f"{self.base_url or BASE_URL}{self.route}"

    @property
    def is_second_surface(self) -> bool:
        return self.base_url is not None

    @property
    def path(self) -> Path:
        return IMAGES_DIR / f"{self.name}.png"


#: Where the service listens once started, and the command that starts it. Both are quoted in
#: the README's « Running it » section: a capture taken against a service started some other
#: way proves something about that other way.
BASE_URL = "http://127.0.0.1:8000"
SERVE_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-m",
    "uvicorn",
    "rl_lander.api:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
)
#: The route that answers once the product is ready. ``/ready`` and not ``/health``: the
#: second says the process is up, the first says a policy is loaded, and photographing the
#: documentation page of a service with no policy would show five routes that all answer 503.
HEALTH_ROUTE: str | None = "/ready"

#: The two Streamlit pages, each on its own port. They are surfaces of this repository that
#: `SERVE_COMMAND` does not start, so `prepare` starts them and `served_by` says how a reader
#: would.
COCKPIT_URL = "http://127.0.0.1:8601"
DASHBOARD_URL = "http://127.0.0.1:8602"

STATE = (
    "the API started from the policy committed at models/ppo/best.zip, and the two Streamlit "
    "pages started against it on their own ports"
)
DATA = (
    "the published evaluation under reports/, and episodes the environment generates during "
    "the capture; nothing here belongs to anyone else"
)

#: The pages started by `prepare`, stopped when this process ends.
_PAGES: list[subprocess.Popen] = []


def _start_page(module: str, port: int) -> None:
    """Start one Streamlit page, headless, and leave it running for the captures."""
    _PAGES.append(
        subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", str(SRC_DIR / module),
                "--server.port", str(port), "--server.headless", "true",
            ],
            cwd=ROOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    )


def _stop_pages() -> None:
    for page in _PAGES:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(page.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        else:
            page.terminate()


def prepare() -> None:
    """Start the two Streamlit pages the API does not start.

    They are not the product `SERVE_COMMAND` serves; they are two more surfaces of the same
    repository, and a reader starts them with the two commands the README quotes. Starting
    them here means the three pictures are taken against one state, in one run, and never
    against whatever happened to be open.
    """
    _start_page("gui.py", 8601)
    _start_page("dashboard.py", 8602)
    atexit.register(_stop_pages)
    for url in (COCKPIT_URL, DASHBOARD_URL):
        wait_until_healthy(f"{url}/_stcore/health", timeout=120)


#: The repository's captures. Nothing else in this file changes from one repository to
#: the next.
CAPTURES: tuple[Capture, ...] = (
    Capture(
        name="api-docs",
        route="/docs",
        # Swagger draws itself from the schema after the page loads, so the engine has to
        # wait for an element, and never for a clock.
        engine="playwright",
        ready_selector="#operations-tag-agent",
        app_state=STATE,
        demonstrates_behaviour=(
            "the five routes, the eight-number observation the agent routes accept and the "
            "trajectory the run route returns are generated from the Pydantic models, so the "
            "page a caller reads cannot drift from the contract the service enforces"
        ),
        data_source="none: the page is generated from the Pydantic models",
        depends_on=("src/rl_lander/api.py",),
    ),
    Capture(
        name="cockpit",
        base_url=COCKPIT_URL,
        served_by="uv run streamlit run src/rl_lander/gui.py",
        route="/",
        engine="playwright",
        app_state=STATE + ", then one episode flown at seed 42 through the API",
        demonstrates_behaviour=(
            "the animation is rebuilt locally from the trajectory the service returned, and "
            "the page states which of the two computed the metrics beside it; a replay that "
            "diverged from the service's episode turns the panel red, where a page could have shown two "
            "different flights side by side"
        ),
        data_source=DATA,
        ready_selector='div[data-testid="stImage"] img',
        steps=(("click", "button", "Run episode"),),
        depends_on=("src/rl_lander/gui.py", "src/rl_lander/replay.py"),
    ),
    Capture(
        name="dashboard",
        base_url=DASHBOARD_URL,
        served_by="uv run streamlit run src/rl_lander/dashboard.py",
        route="/",
        engine="playwright",
        app_state=STATE,
        demonstrates_behaviour=(
            "the hundred episodes of the exported grid, read from a tracked file and not "
            "recomputed, with every panel of the page drawing from the same filtered frame"
        ),
        data_source=DATA,
        ready_selector='div[data-testid="stPlotlyChart"]',
        paired_with="cockpit",
        depends_on=("src/rl_lander/dashboard.py", "src/rl_lander/artifacts.py"),
    ),
)


# --- Starting the product, and knowing when it is up ------------------------


def _answers(url: str) -> bool:
    """Whether something is already serving that URL, right now."""
    try:
        with urllib.request.urlopen(url, timeout=1) as answer:
            return answer.status < 500
    except (urllib.error.URLError, OSError):
        return False


def wait_until_healthy(url: str, *, timeout: float = 90.0) -> None:
    """Poll until the service answers. Never sleep a fixed number of seconds.

    A fixed sleep is either too short on a cold start, and the capture photographs a
    connection error, or wasted on every run afterwards.
    """
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as answer:
                if answer.status < 500:
                    return
        except (urllib.error.URLError, OSError) as exc:  # not up yet
            last = exc
        time.sleep(0.25)
    raise TimeoutError(f"{url} never answered in {timeout:.0f}s ({last})")


class Serving:
    """Start the product, wait for it, capture, stop it — even when a capture raises."""

    def __init__(self, command: tuple[str, ...], health: str | None):
        self.command = command
        self.health = health
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> Self:
        if self.health is None:
            return self
        if not self.command:
            wait_until_healthy(self.health, timeout=5)
            return self
        # Something already answering on that port gets photographed in place of the
        # product: a server left over from an earlier run serves an older build, and its
        # picture is indistinguishable from a fresh one.
        if _answers(self.health):
            raise RuntimeError(
                f"{self.health} already answers: stop what is listening before capturing, "
                "or the picture will be of that and not of this build"
            )
        self.process = subprocess.Popen(
            list(self.command), cwd=ROOT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        wait_until_healthy(self.health)
        return self

    def __exit__(self, *_exception) -> None:
        """Stop the whole tree. `uv run uvicorn` is two processes, and killing the first
        leaves the second holding the port for the next run."""
        if self.process is None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()


# --- The two engines --------------------------------------------------------


def _chrome_binary() -> str:
    """The browser on this machine, named by the environment and never guessed.

    A path hard-coded here would be one machine's installation shipped inside a published
    repository; ``CHROME_PATH`` keeps that constraint where it belongs.
    """
    explicit = os.environ.get("CHROME_PATH")
    if explicit:
        return explicit
    for name in ("chrome", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "no Chrome on PATH: set CHROME_PATH to the browser's executable, or run the "
        "capture with --engine playwright"
    )


def by_chrome(capture: Capture) -> None:
    """One pass, headless. `--virtual-time-budget` is what makes a JavaScript page render."""
    width, height = VIEWPORT
    subprocess.run(
        [
            _chrome_binary(),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--virtual-time-budget=8000",
            f"--window-size={width},{height}",
            f"--force-device-scale-factor={DEVICE_SCALE_FACTOR}",
            f"--screenshot={capture.path}",
            capture.target,
        ],
        check=True,
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def by_playwright(capture: Capture) -> None:
    """For a page whose content arrives over a websocket, and that Chrome photographs black.

    ``channel="chrome"`` reuses the system browser: no download, and the picture is taken by
    the same engine a reader would open the page with.
    """
    from playwright.sync_api import sync_playwright  # installed in the capture environment

    width, height = VIEWPORT
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        page = browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=DEVICE_SCALE_FACTOR,
        )
        page.goto(capture.target, wait_until="networkidle", timeout=90_000)
        for action, target, value in capture.steps:
            if action == "click":
                # `.first`: a Swagger operation carries a title button and an arrow button
                # under the same accessible name, and the first in document order is the one
                # a reader sees and clicks.
                page.get_by_role(target, name=value).first.click()
            elif action == "open":
                # A control named by a CSS selector rather than by an accessible name. A
                # Swagger operation reached through a deep link is not always expanded by the
                # time the page settles, and its « Try it out » button is not in the DOM
                # until it is: clicking the operation's own header is what puts it there.
                page.locator(target).first.click()
            elif action == "fill":
                page.locator(target).first.fill(value)
            elif action == "scroll":
                page.locator(target).first.scroll_into_view_if_needed()
            else:
                raise ValueError(f"{capture.name}: unknown capture step « {action} »")
            page.wait_for_timeout(300)
        if capture.ready_selector:
            page.wait_for_selector(capture.ready_selector, timeout=180_000)
        page.wait_for_timeout(4000)  # let the animations settle
        page.screenshot(path=str(capture.path))
        browser.close()


ENGINES = {"chrome": by_chrome, "playwright": by_playwright}


# --- Writing down what was photographed -------------------------------------


def _git_revision() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


def record(capture: Capture) -> dict:
    """The manifest entry for an image that has just been written."""
    if capture.data_source == "third_party":
        raise ValueError(
            f"{capture.name}: a capture does not redistribute someone else's work. "
            "Photograph the product against data this repository may publish."
        )
    entry = {
        "sha256": hashlib.sha256(capture.path.read_bytes()).hexdigest(),
        "written": datetime.now(tz=UTC).date().isoformat(),
        "source": SOURCE,
        "command": f"uv run python {SOURCE} --only {capture.name}",
        "target": capture.target,
        "viewport": list(VIEWPORT),
        "device_scale_factor": DEVICE_SCALE_FACTOR,
        "app_state": capture.app_state,
        "data_source": capture.data_source,
        "demonstrates_behaviour": capture.demonstrates_behaviour,
    }
    revision = _git_revision()
    if revision:
        entry["git_revision"] = revision
    if capture.paired_with:
        entry["paired_with"] = f"{capture.paired_with}.png"
    if capture.depends_on:
        entry["depends_on"] = list(capture.depends_on)
    return entry


def write_manifest(entries: dict[str, dict]) -> None:
    """Merge into the manifest. An image nobody re-took keeps the entry it had."""
    payload: dict = {"schema": "image-manifest/1", "images": {}}
    if MANIFEST.exists():
        with contextlib.suppress(json.JSONDecodeError):
            payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload.setdefault("schema", "image-manifest/1")
    payload.setdefault("images", {})
    payload["images"].update(entries)
    payload["images"] = dict(sorted(payload["images"].items()))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline=""
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="a single capture, by name")
    parser.add_argument(
        "--check", action="store_true", help="take nothing; report what the manifest is missing"
    )
    arguments = parser.parse_args(argv)

    wanted = [c for c in CAPTURES if not arguments.only or c.name == arguments.only]
    if not wanted:
        print("no capture selected", file=sys.stderr)
        return 1

    if arguments.check:
        known = {}
        if MANIFEST.exists():
            known = json.loads(MANIFEST.read_text(encoding="utf-8")).get("images", {})
        stale = [
            c.name
            for c in wanted
            if not c.path.exists()
            or known.get(c.path.name, {}).get("sha256")
            != hashlib.sha256(c.path.read_bytes()).hexdigest()
        ]
        for name in stale:
            print(f"  {name} is missing or does not match its manifest entry")
        return 1 if stale else 0

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    prepare()
    entries: dict[str, dict] = {}
    health = f"{BASE_URL}{HEALTH_ROUTE}" if HEALTH_ROUTE else None
    with Serving(SERVE_COMMAND, health):
        for capture in wanted:
            if capture.is_second_surface and not _answers(capture.target):
                print(
                    f"{capture.name:24} skipped: nothing answers {capture.target}. "
                    f"Start it with: {capture.served_by}",
                    file=sys.stderr,
                )
                continue
            ENGINES[capture.engine](capture)
            entries[capture.path.name] = record(capture)
            print(f"{capture.name:24} {capture.path.relative_to(ROOT_DIR)}")
    write_manifest(entries)
    print(f"{len(entries)} capture(s), {MANIFEST.relative_to(ROOT_DIR)} updated")
    print("Read every image before committing it: no key, no token, no address on screen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
