"""Operators: install the solver environment, solve an image, import a saved solution."""

from __future__ import annotations

import time
from pathlib import Path

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import jobs, scene
from .properties import prefs, user_dir


class JobOperator:
    """Runs a jobs.Job modally (UI) or blocking (execute, e.g. from a background script)."""

    _job = None
    _timer = None

    def make_job(self, context) -> jobs.Job | None:
        raise NotImplementedError

    def on_finish(self, context) -> set[str]:
        raise NotImplementedError

    def _poll_job(self, context) -> None:
        wm = context.window_manager
        wm.aicm_progress, wm.aicm_status = self._job.progress, self._job.message

    def _begin(self, context) -> bool:
        wm = context.window_manager
        if wm.aicm_busy:
            self.report({"ERROR"}, "Another AI Camera Match job is already running")
            return False
        self._job = self.make_job(context)
        if self._job is None:
            return False
        wm.aicm_busy, wm.aicm_progress, wm.aicm_status = True, 0.0, "Starting"
        self._job.start()
        return True

    def _end(self, context) -> set[str]:
        wm = context.window_manager
        self._poll_job(context)
        wm.aicm_busy = False
        if self._job.returncode == 0:
            return self.on_finish(context)
        if self._job.error == "Cancelled":
            wm.aicm_status = "Cancelled"
            self.report({"WARNING"}, "Cancelled")
        else:
            wm.aicm_status = f"Failed: {self._job.error}"
            detail = wm.aicm_status.rstrip(".")
            self.report({"ERROR"}, f"{detail}. Full log: {self._job.log_path}")
        return {"CANCELLED"}

    def execute(self, context):
        if not self._begin(context):
            return {"CANCELLED"}
        while not self._job.done:
            time.sleep(0.2)
            self._poll_job(context)
        return self._end(context)

    def invoke(self, context, event):
        if context.window is None:
            return self.execute(context)
        if not self._begin(context):
            return {"CANCELLED"}
        self._timer = context.window_manager.event_timer_add(
            0.25, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            self._job.cancel()
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        self._poll_job(context)
        for area in context.screen.areas if context.screen else []:
            area.tag_redraw()
        if not self._job.done:
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        return self._end(context)


class AICM_OT_install_environment(JobOperator, bpy.types.Operator):
    """Create the solver environment and install PyTorch, GeoCalib and MoGe into it"""

    bl_idname = "aicm.install_environment"
    bl_label = "Install Solver Environment"

    def make_job(self, context):
        if not bpy.app.online_access:
            self.report(
                {"ERROR"},
                "Enable Preferences > System > Network > Allow Online Access first",
            )
            return None
        p = prefs(context)
        env_dir = p.resolved_env_dir()
        return jobs.Job(
            title="install",
            steps=jobs.install_steps(p.resolved_base_python(), env_dir, p.torch_index),
            env=jobs.process_env(p.cache_dir()),
            log_path=user_dir() / "logs" / "install.log",
        )

    def on_finish(self, context):
        wm = context.window_manager
        wm.aicm_status = f"Solver environment ready ({self._job.last_output})"
        self.report({"INFO"}, wm.aicm_status)
        return {"FINISHED"}


def solve_dir(image: Path) -> Path:
    """Next to the .blend when saved, so plates and meshes travel with the project."""
    if bpy.data.filepath:
        return Path(bpy.path.abspath("//")) / "camera_match" / image.stem
    return user_dir() / "solves" / image.stem


class AICM_OT_solve(JobOperator, bpy.types.Operator):
    """Estimate the camera for the image with GeoCalib and MoGe, then build the matched scene"""

    bl_idname = "aicm.solve"
    bl_label = "Match Camera"
    bl_options = {"REGISTER", "UNDO"}

    def make_job(self, context):
        p, settings = prefs(context), context.scene.aicm
        image = Path(bpy.path.abspath(settings.image))
        if not settings.image or not image.is_file():
            self.report({"ERROR"}, "Choose an image file first")
            return None
        if not p.env_ready():
            self.report(
                {"ERROR"},
                "Install the solver environment in the add-on preferences first",
            )
            return None
        depth_model = settings.depth_model_id()
        if not depth_model:
            self.report({"ERROR"}, "Enter a custom depth model")
            return None

        self._out_dir = solve_dir(image)
        command = [
            str(jobs.env_python(p.resolved_env_dir())),
            "-m",
            "aicm_worker",
            "--image",
            str(image),
            "--out-dir",
            str(self._out_dir),
            "--camera-model",
            settings.camera_model,
            "--depth-model",
            depth_model,
            "--depth-resolution",
            str(settings.depth_resolution),
            "--default-height",
            f"{settings.default_height:.3g}",
            "--device",
            p.device,
        ]
        if not settings.import_mesh:
            command.append("--no-mesh")
        return jobs.Job(
            title="solve",
            steps=[("Solving", command)],
            env=jobs.process_env(p.cache_dir(), offline=not bpy.app.online_access),
            log_path=self._out_dir / "solve.log",
        )

    def on_finish(self, context):
        settings = context.scene.aicm
        wm = context.window_manager
        try:
            camera = scene.build_scene(
                context,
                self._out_dir / "solution.json",
                settings.add_floor,
                settings.import_mesh,
            )
        except Exception as error:  # surface scene-building problems to the user
            wm.aicm_status = f"Solved, but building the scene failed: {error}"
            self.report({"ERROR"}, wm.aicm_status)
            return {"CANCELLED"}
        solution = scene.load_solution(self._out_dir / "solution.json")
        for warning in solution["warnings"]:
            self.report({"WARNING"}, warning)
        wm.aicm_status = f"Matched {camera.name}: {scene.summary(solution, camera)}"
        self.report({"INFO"}, wm.aicm_status)
        return {"FINISHED"}


class AICM_OT_import_solution(bpy.types.Operator, ImportHelper):
    """Build the matched scene from a solution.json written by the solver"""

    bl_idname = "aicm.import_solution"
    bl_label = "Import Camera Match Solution"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    add_floor: BoolProperty(name="Floor Plane", default=True)
    import_mesh: BoolProperty(name="Proxy Mesh", default=True)

    def execute(self, context):
        try:
            camera = scene.build_scene(
                context, Path(self.filepath), self.add_floor, self.import_mesh
            )
        except (OSError, ValueError, KeyError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        solution = scene.load_solution(Path(self.filepath))
        self.report(
            {"INFO"}, f"Imported {camera.name}: {scene.summary(solution, camera)}"
        )
        return {"FINISHED"}


class AICM_OT_open_folder(bpy.types.Operator):
    """Open the folder holding the last solve's plate, mesh, .fspy and log"""

    bl_idname = "aicm.open_folder"
    bl_label = "Open Solve Folder"

    def execute(self, context):
        image = context.scene.aicm.image
        folder = solve_dir(Path(bpy.path.abspath(image))) if image else user_dir()
        if not folder.exists():
            self.report(
                {"ERROR"},
                f"Nothing solved for this image yet ({folder} does not exist)",
            )
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=str(folder))
        return {"FINISHED"}


CLASSES = (
    AICM_OT_install_environment,
    AICM_OT_solve,
    AICM_OT_import_solution,
    AICM_OT_open_folder,
)
