import importlib.util
import sys
import time
from pathlib import Path

import pytest

# jobs.py has no bpy dependency; load it without importing the add-on package (which needs bpy).
_spec = importlib.util.spec_from_file_location(
    "aicm_jobs", Path(__file__).resolve().parents[1] / "ai_camera_match" / "jobs.py"
)
jobs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = jobs
_spec.loader.exec_module(jobs)


def python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def run(job: "jobs.Job", timeout: float = 30.0) -> "jobs.Job":
    job.start()
    deadline = time.time() + timeout
    while not job.done:
        assert time.time() < deadline, "job did not finish"
        time.sleep(0.05)
    return job


def make_job(tmp_path, steps, title="solve"):
    return jobs.Job(title=title, steps=steps, env=None, log_path=tmp_path / "job.log")


def test_progress_events_update_state_and_read_cleanly(tmp_path):
    code = (
        "import json, sys\n"
        "print(json.dumps({'progress': 0.5, 'message': 'Finding the floor'}), flush=True)\n"
        "sys.stderr.write('Downloading: 10%\\rDownloading: 60%\\rDownloading: 100%\\n')\n"
        "print('Result line', flush=True)\n"
    )
    job = run(make_job(tmp_path, [("Solving", python(code))]))

    assert job.returncode == 0
    assert (job.progress, job.message, job.error) == (1.0, "Finding the floor", "")
    text = job.log_path.read_text(encoding="utf-8")
    assert text.startswith("AI Camera Match solve log\nStarted ")
    assert "] Finding the floor\n" in text
    assert '"progress"' not in text  # protocol lines are rewritten, not dumped
    assert (
        "] Downloading: 100%\n" in text and "] Downloading: 10%" not in text
    )  # bar collapsed
    assert "] Step finished in " in text
    assert "Finished successfully (" in text


def test_worker_error_event_becomes_the_message(tmp_path):
    code = (
        "import json, sys\n"
        "sys.stderr.write('Traceback (most recent call last):\\nValueError: raw detail\\n')\n"
        "print(json.dumps({'error': 'The GPU ran out of memory.'}), flush=True)\n"
        "sys.exit(1)\n"
    )
    job = run(make_job(tmp_path, [("Solving", python(code))]))

    assert job.returncode == 1
    assert job.error == "The GPU ran out of memory."
    text = job.log_path.read_text(encoding="utf-8")
    assert "ValueError: raw detail" in text  # the traceback stays in the log
    assert "Failed: The GPU ran out of memory." in text


def test_failed_step_reports_the_error_line_not_the_last_line(tmp_path):
    ok = python("print('fine')")
    failing = python(
        "import sys\n"
        "print('ImportError: Failed to load PyTorch C extensions:')\n"
        "print('    or by running Python from a different directory.')\n"
        "sys.exit(1)\n"
    )
    job = run(
        make_job(
            tmp_path,
            [("Upgrading pip", ok), ("Checking installation", failing)],
            "install",
        )
    )

    assert job.returncode == 1
    assert (
        job.error
        == "Checking installation failed: ImportError: Failed to load PyTorch C extensions:"
    )
    text = job.log_path.read_text(encoding="utf-8")
    assert (
        "Step 1/2: Upgrading pip" in text and "Step 2/2: Checking installation" in text
    )
    assert "Step failed with exit code 1" in text


def test_missing_executable_is_reported(tmp_path):
    job = run(
        make_job(
            tmp_path,
            [("Creating virtual environment", ["no-such-python-exe"])],
            "install",
        )
    )
    assert job.returncode == 1
    assert job.error.startswith(
        "Creating virtual environment: could not start no-such-python-exe"
    )


def test_cancel_stops_the_process(tmp_path):
    job = make_job(tmp_path, [("Solving", python("import time; time.sleep(30)"))])
    job.start()
    time.sleep(0.5)
    job.cancel()
    run(job, timeout=10)
    assert (job.returncode, job.error) == (1, "Cancelled")
    assert "Failed: Cancelled" in job.log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "line",
    ["ERROR: No matching distribution found for torch", "torch.OutOfMemoryError: CUDA"],
)
def test_error_line_pattern(line):
    assert jobs.ERROR_LINE.match(line)
