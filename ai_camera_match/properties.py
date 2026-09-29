"""Add-on preferences (solver environment) and per-scene solve settings."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)

from . import jobs

DEPTH_MODELS = [
    (
        "Ruicheng/moge-2-vitl-normal",
        "MoGe-2 ViT-L",
        "331M parameters, most accurate MoGe-2 (default)",
    ),
    ("Ruicheng/moge-2-vitb-normal", "MoGe-2 ViT-B", "104M parameters, faster"),
    (
        "Ruicheng/moge-2-vits-normal",
        "MoGe-2 ViT-S",
        "35M parameters, fastest, for low-memory GPUs",
    ),
    (
        "CUSTOM",
        "Custom",
        "A MoGe-2 or MoGe-3 Hugging Face repo id or local checkpoint path",
    ),
]


def user_dir() -> Path:
    return Path(bpy.utils.extension_path_user(__package__, path="", create=True))


def prefs(context=None) -> "AICM_Preferences":
    return (context or bpy.context).preferences.addons[__package__].preferences


class AICM_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    env_dir: StringProperty(
        name="Solver Environment",
        description="Virtual environment holding PyTorch, GeoCalib and MoGe. Empty uses the extension's user folder",
        subtype="DIR_PATH",
    )
    base_python: StringProperty(
        name="Base Python",
        description="Python 3.10+ used to create the environment. Empty uses Blender's bundled Python",
        subtype="FILE_PATH",
    )
    torch_index: EnumProperty(
        name="PyTorch Build",
        items=[
            ("CU128", "CUDA 12.8", "NVIDIA GPUs including RTX 50 series"),
            ("CU126", "CUDA 12.6", "NVIDIA GPUs from GTX 900 series onwards"),
            ("CPU", "CPU only", "No GPU acceleration; solving takes much longer"),
            (
                "PYPI",
                "PyPI default",
                "macOS (Apple Silicon uses Metal) or Linux default CUDA build",
            ),
        ],
        default="PYPI" if sys.platform == "darwin" else "CU126",
    )
    device: EnumProperty(
        name="Device",
        items=[
            ("auto", "Auto", "GPU when available"),
            ("cuda", "CUDA", ""),
            ("mps", "Metal", ""),
            ("cpu", "CPU", ""),
        ],
        default="auto",
    )

    def resolved_env_dir(self) -> Path:
        return (
            Path(bpy.path.abspath(self.env_dir)) if self.env_dir else user_dir() / "env"
        )

    def resolved_base_python(self) -> str:
        return (
            bpy.path.abspath(self.base_python) if self.base_python else sys.executable
        )

    def cache_dir(self) -> Path:
        return self.resolved_env_dir().parent / "cache"

    def env_ready(self) -> bool:
        return jobs.env_python(self.resolved_env_dir()).exists()

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "env_dir")
        layout.prop(self, "base_python")
        layout.prop(self, "torch_index")
        layout.prop(self, "device")

        wm = context.window_manager
        box = layout.box()
        if wm.aicm_busy:
            box.progress(factor=wm.aicm_progress, text=wm.aicm_status)
        else:
            state = "installed" if self.env_ready() else "not installed"
            box.label(
                text=f"Solver environment: {state}",
                icon="CHECKMARK" if self.env_ready() else "ERROR",
            )
            box.label(text="Needs about 6 GB of disk and an internet connection.")
            box.operator("aicm.install_environment", icon="IMPORT")
            if wm.aicm_status:
                box.label(text=wm.aicm_status)


class AICM_SceneSettings(bpy.types.PropertyGroup):
    image: StringProperty(
        name="Image", description="Photo to match", subtype="FILE_PATH"
    )
    camera_model: EnumProperty(
        name="Lens",
        items=[
            ("pinhole", "Pinhole", "No lens distortion (most photos)"),
            ("simple_radial", "Simple Radial", "Mild barrel or pincushion distortion"),
            ("radial", "Radial", "Stronger polynomial distortion"),
            (
                "simple_divisional",
                "Fisheye (Divisional)",
                "Strong wide-angle distortion",
            ),
        ],
        default="pinhole",
    )
    estimate_height: BoolProperty(
        name="Estimate Height and Scale",
        description="Run MoGe for metric depth to find the floor and the camera height",
        default=True,
    )
    depth_model: EnumProperty(
        name="Depth Model", items=DEPTH_MODELS, default=DEPTH_MODELS[0][0]
    )
    custom_depth_model: StringProperty(
        name="Model", description="Hugging Face repo id or checkpoint path"
    )
    depth_resolution: IntProperty(
        name="Depth Resolution",
        description="Long edge in pixels for depth estimation and the proxy mesh",
        default=1024,
        min=256,
        max=4096,
    )
    default_height: FloatProperty(
        name="Fallback Height",
        description="Camera height when no floor is found or depth is off",
        default=1.6,
        min=0.0,
        unit="LENGTH",
    )
    add_floor: BoolProperty(
        name="Floor Plane", description="Add a wireframe floor at Z = 0", default=True
    )
    import_mesh: BoolProperty(
        name="Proxy Mesh",
        description="Import the textured depth mesh for shadows, reflections and reference",
        default=True,
    )

    def depth_model_id(self) -> str:
        if not self.estimate_height:
            return "none"
        return (
            self.custom_depth_model.strip()
            if self.depth_model == "CUSTOM"
            else self.depth_model
        )


CLASSES = (AICM_Preferences, AICM_SceneSettings)


def register_runtime_props():
    wm = bpy.types.WindowManager
    wm.aicm_busy = BoolProperty(default=False, options={"SKIP_SAVE"})
    wm.aicm_progress = FloatProperty(
        default=0.0, min=0.0, max=1.0, options={"SKIP_SAVE"}
    )
    wm.aicm_status = StringProperty(default="", options={"SKIP_SAVE"})
    bpy.types.Scene.aicm = bpy.props.PointerProperty(type=AICM_SceneSettings)


def unregister_runtime_props():
    del bpy.types.Scene.aicm
    for name in ("aicm_busy", "aicm_progress", "aicm_status"):
        delattr(bpy.types.WindowManager, name)
