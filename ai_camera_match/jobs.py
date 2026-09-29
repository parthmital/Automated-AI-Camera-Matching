"""Background subprocess jobs: the solver environment installer and the solver worker.

No bpy here; operators poll a Job from a modal timer so Blender's UI stays responsive.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

WORKER_DIR = Path(__file__).parent / "worker"

TORCH_INDEXES = {
    "CU128": "https://download.pytorch.org/whl/cu128",
    "CU126": "https://download.pytorch.org/whl/cu126",
    "CPU": "https://download.pytorch.org/whl/cpu",
    "PYPI": None,
}

# Lines worth showing when a step fails: pip errors and Python exception lines.
ERROR_LINE = re.compile(r"^(ERROR: .+|[A-Za-z_.]*(Error|Exception)\b.*)")

INSTALL_CHECK = (
    "import torch, geocalib, moge, aicm_worker; "
    "gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None; "
    "print(f'PyTorch {torch.__version__}, ' + (f'GPU: {gpu}' if gpu else 'CPU only'))"
)


def env_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def process_env(cache_dir: Path, offline: bool = False) -> dict[str, str]:
    """Environment for child processes: caches beside the solver env, worker on the path."""
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONHOME", "PYTHONPATH"}}
    env.update(
        PYTHONPATH=str(WORKER_DIR),
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        PIP_CACHE_DIR=str(cache_dir / "pip"),
        TORCH_HOME=str(cache_dir / "torch"),
        HF_HOME=str(cache_dir / "huggingface"),
    )
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def install_steps(
    base_python: str, env_dir: Path, torch_index: str
) -> list[tuple[str, list[str]]]:
    """(label, command) pairs that build or update the solver virtual environment.

    An existing environment is never re-created: running venv over it with a different base
    Python swaps its interpreter and breaks the compiled packages already installed there.
    """
    python = str(env_python(env_dir))
    pip = [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--progress-bar",
        "off",
    ]
    torch = pip + ["torch", "torchvision"]
    if TORCH_INDEXES[torch_index]:
        torch += ["--index-url", TORCH_INDEXES[torch_index]]
    steps = []
    if not env_python(env_dir).exists():
        steps.append(
            ("Creating virtual environment", [base_python, "-m", "venv", str(env_dir)])
        )
    return steps + [
        ("Upgrading pip", pip + ["--upgrade", "pip"]),
        ("Installing PyTorch (large download)", torch),
        (
            "Installing solver dependencies",
            pip + ["-r", str(WORKER_DIR / "requirements.txt")],
        ),
        (
            "Installing GeoCalib and MoGe",
            pip + ["--no-deps", "-r", str(WORKER_DIR / "requirements-nodeps.txt")],
        ),
        (
            "Checking installation",
            [python, "-W", "ignore::FutureWarning", "-c", INSTALL_CHECK],
        ),
    ]


def _clock() -> str:
    return time.strftime("%H:%M:%S")


@dataclass
class Job:
    """Runs commands one after another on a background thread and writes a readable log.

    Worker progress lines ({"progress": ..., "message": ...}) update ``progress`` and
    ``message``; an {"error": ...} line or the last error-looking output becomes ``error``.
    """

    title: str
    steps: list[tuple[str, list[str]]]
    env: dict[str, str]
    log_path: Path
    progress: float = 0.0
    message: str = "Starting"
    error: str = ""
    last_output: str = ""
    returncode: int | None = None
    _last_error_line: str = ""
    _process: subprocess.Popen | None = None
    _cancelled: bool = False

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._run, daemon=True).start()

    def _write(self, log: TextIO, text: str) -> None:
        log.write(f"[{_clock()}] {text}\n")
        log.flush()

    def _handle_line(self, log: TextIO, raw: bytes) -> None:
        # Progress bars redraw with carriage returns; keep only the final state.
        text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        line = next((part for part in reversed(text.split("\r")) if part.strip()), "")
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        if isinstance(event, dict) and "progress" in event:
            self.progress, self.message = float(event["progress"]), str(
                event["message"]
            )
            self._write(log, self.message)
        elif isinstance(event, dict) and "error" in event:
            self.error = str(
                event["error"]
            )  # written once, in the closing "Failed:" line
        else:
            self._write(log, line)
            self.last_output = line.strip()
            if ERROR_LINE.match(self.last_output):
                self._last_error_line = self.last_output

    def _run(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        started = time.perf_counter()
        with open(self.log_path, "w", encoding="utf-8") as log:
            log.write(
                f"AI Camera Match {self.title} log\nStarted {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            for index, (label, command) in enumerate(self.steps):
                if self._cancelled:
                    break
                step_started = time.perf_counter()
                if len(self.steps) > 1:
                    self.progress, self.message = index / len(self.steps), label
                log.write("\n")
                self._write(log, f"Step {index + 1}/{len(self.steps)}: {label}")
                self._write(log, f"$ {subprocess.list2cmdline(command)}")
                try:
                    self._process = subprocess.Popen(
                        command,
                        env=self.env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        creationflags=flags,
                    )
                except OSError as error:
                    self.error = f"{label}: could not start {command[0]} ({error.strerror or error})"
                    self._finish(log, 1, started)
                    return
                # Read bytes: text mode's universal newlines would split progress bars at each \r.
                for raw in self._process.stdout:
                    self._handle_line(log, raw)
                code = self._process.wait()
                seconds = time.perf_counter() - step_started
                if self._cancelled:
                    break
                if code != 0:
                    self._write(
                        log, f"Step failed with exit code {code} after {seconds:.1f} s"
                    )
                    detail = (
                        self.error
                        or self._last_error_line
                        or self.last_output
                        or f"exit code {code}"
                    )
                    self.error = (
                        f"{label} failed: {detail}" if len(self.steps) > 1 else detail
                    )
                    self._finish(log, code, started)
                    return
                self._write(log, f"Step finished in {seconds:.1f} s")
            if self._cancelled:
                self.error = "Cancelled"
                self._finish(log, 1, started)
            else:
                self.progress = 1.0
                self._finish(log, 0, started)

    def _finish(self, log: TextIO, code: int, started: float) -> None:
        seconds = time.perf_counter() - started
        outcome = "Finished successfully" if code == 0 else f"Failed: {self.error}"
        log.write(f"\n{outcome} ({seconds:.1f} s)\n")
        log.flush()
        self.returncode = code

    @property
    def done(self) -> bool:
        return self.returncode is not None

    def cancel(self) -> None:
        self._cancelled = True
        if self._process and self._process.poll() is None:
            self._process.kill()
