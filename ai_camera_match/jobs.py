"""Background subprocess jobs: the solver environment installer and the solver worker.

No bpy here; operators poll a Job from a modal timer so Blender's UI stays responsive.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

WORKER_DIR = Path(__file__).parent / "worker"

TORCH_INDEXES = {
    "CU128": "https://download.pytorch.org/whl/cu128",
    "CU126": "https://download.pytorch.org/whl/cu126",
    "CPU": "https://download.pytorch.org/whl/cpu",
    "PYPI": None,
}


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
    """(label, command) pairs that build the solver virtual environment."""
    python = str(env_python(env_dir))
    pip = [python, "-m", "pip", "install", "--disable-pip-version-check"]
    torch = pip + ["torch", "torchvision"]
    if TORCH_INDEXES[torch_index]:
        torch += ["--index-url", TORCH_INDEXES[torch_index]]
    return [
        ("Creating virtual environment", [base_python, "-m", "venv", str(env_dir)]),
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
            [
                python,
                "-c",
                "import torch, geocalib, moge, aicm_worker; "
                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())",
            ],
        ),
    ]


@dataclass
class Job:
    """Runs commands one after another, collecting output lines on a background thread."""

    steps: list[tuple[str, list[str]]]
    env: dict[str, str]
    log_path: Path
    lines: "queue.Queue[str]" = field(default_factory=queue.Queue)
    label: str = ""
    step: int = 0
    returncode: int | None = None
    _process: subprocess.Popen | None = None
    _cancelled: bool = False

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with open(self.log_path, "w", encoding="utf-8") as log:
            for self.step, (label, command) in enumerate(self.steps):
                if self._cancelled:
                    break
                self.label = label
                log.write(f"\n$ {subprocess.list2cmdline(command)}\n")
                log.flush()
                try:
                    self._process = subprocess.Popen(
                        command,
                        env=self.env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        creationflags=flags,
                    )
                except OSError as error:
                    self.lines.put(json.dumps({"error": f"{label}: {error}"}))
                    self.returncode = 1
                    return
                for line in self._process.stdout:
                    log.write(line)
                    log.flush()
                    self.lines.put(line.rstrip())
                code = self._process.wait()
                if code != 0:
                    self.returncode = code
                    return
        self.returncode = 1 if self._cancelled else 0

    @property
    def done(self) -> bool:
        return self.returncode is not None

    def cancel(self) -> None:
        self._cancelled = True
        if self._process and self._process.poll() is None:
            self._process.kill()

    def drain(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                return out
