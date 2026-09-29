"""Build the matched camera, plate, floor and proxy mesh from a solver solution."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from mathutils import Matrix

SENSOR_MM = 36.0


def load_solution(path: Path) -> dict:
    solution = json.loads(path.read_text(encoding="utf-8"))
    if solution.get("format") != "ai-camera-match-solution":
        raise ValueError(f"{path.name} is not an AI Camera Match solution")
    return solution


def _collection(scene: bpy.types.Scene, name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)
    return collection


def _object(collection, name: str, data) -> bpy.types.Object:
    """Reuse an object of the same name and type so re-solving updates it in place."""
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type != (
        "CAMERA" if isinstance(data, bpy.types.Camera) else "MESH"
    ):
        raise ValueError(
            f"'{name}' already exists and is not a {type(data).__name__}; rename it and retry"
        )
    if obj is None:
        obj = bpy.data.objects.new(name, data)
    elif obj.data is not data:
        old = obj.data
        obj.data = data
        if old.users == 0:
            (
                bpy.data.cameras
                if isinstance(old, bpy.types.Camera)
                else bpy.data.meshes
            ).remove(old)
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    return obj


def _setup_camera(
    camera: bpy.types.Camera, solution: dict, plate: bpy.types.Image
) -> None:
    width, height = solution["image"]["width"], solution["image"]["height"]
    intrinsics = solution["intrinsics"]
    horizontal = width >= height
    camera.type = "PERSP"
    camera.lens_unit = "MILLIMETERS"
    camera.sensor_fit = "HORIZONTAL" if horizontal else "VERTICAL"
    camera.sensor_width = camera.sensor_height = SENSOR_MM
    camera.lens = intrinsics["focal_px"] * SENSOR_MM / (width if horizontal else height)
    # The solvers assume a centred principal point, which is Blender's zero lens shift.
    camera.shift_x = camera.shift_y = 0.0
    camera.clip_end = max(camera.clip_end, 1000.0)

    camera.show_background_images = True
    background = next(
        (bg for bg in camera.background_images if bg.image == plate), None
    )
    for bg in camera.background_images:
        bg.show_background_image = False
    if background is None:
        background = camera.background_images.new()
        background.image = plate
    background.show_background_image = True
    background.frame_method = "FIT"
    background.display_depth = "BACK"
    background.alpha = 1.0


def _add_floor(collection, name: str) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    half = 10.0
    mesh.from_pydata(
        [(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)],
        [],
        [(0, 1, 2, 3)],
    )
    floor = _object(collection, name, mesh)
    floor.display_type = "WIRE"
    floor.matrix_world = Matrix.Identity(4)
    return floor


def _import_proxy_mesh(collection, path: Path, name: str) -> list[bpy.types.Object]:
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in imported:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
        if obj.type == "MESH":
            obj.name = name
    return imported


def _look_through(context) -> None:
    screen = getattr(context, "screen", None)
    for area in screen.areas if screen else []:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"


def build_scene(
    context, solution_path: Path, add_floor: bool = True, import_mesh: bool = True
) -> bpy.types.Object:
    """Create or update the matched setup for a solution.json and return the camera object."""
    solution_path = Path(bpy.path.abspath(str(solution_path))).resolve()
    solution = load_solution(solution_path)
    folder = solution_path.parent
    scene = context.scene
    stem = Path(solution["source_image"]).stem
    collection = _collection(scene, f"Camera Match {stem}")

    plate_path = folder / solution["plate"]
    plate = bpy.data.images.load(str(plate_path), check_existing=True)
    plate.reload()
    plate.name = f"{stem}_plate"

    camera_name = f"{stem}_camera"
    camera_data = bpy.data.cameras.get(camera_name) or bpy.data.cameras.new(camera_name)
    _setup_camera(camera_data, solution, plate)
    camera = _object(collection, camera_name, camera_data)
    camera.matrix_world = Matrix(solution["camera"]["matrix_world"])
    camera["ai_camera_match"] = json.dumps(
        {
            k: solution[k]
            for k in (
                "intrinsics",
                "orientation",
                "camera",
                "ground",
                "models",
                "warnings",
            )
        }
    )

    scene.camera = camera
    scene.render.resolution_x = solution["image"]["width"]
    scene.render.resolution_y = solution["image"]["height"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0

    if add_floor:
        _add_floor(collection, f"{stem}_floor")
    if import_mesh and solution.get("proxy_mesh"):
        _import_proxy_mesh(collection, folder / solution["proxy_mesh"], f"{stem}_proxy")

    _look_through(context)
    return camera
