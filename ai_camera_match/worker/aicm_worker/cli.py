"""Command-line solver: one image in, a camera solution JSON (plus plate, mesh, .fspy) out.

Run from the Blender add-on or directly:

    python -m aicm_worker --image photo.jpg --out-dir solve/

Progress is streamed to stdout as JSON lines ({"progress": 0.4, "message": "..."}); a failure
ends with {"error": "..."}. Readable details (setup, results, tracebacks) go to stderr.
The solution is written to <out-dir>/solution.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import __version__
from .geometry import (
    camera_matrix_world,
    fit_ground_plane,
    fov_from_focal,
    roll_pitch_from_up,
    world_from_cv_rotation,
)

CAMERA_MODELS = {
    "pinhole": "Pinhole",
    "simple_radial": "Simple Radial",
    "radial": "Radial",
    "simple_divisional": "Fisheye (Divisional)",
}
DEFAULT_DEPTH_MODEL = "Ruicheng/moge-2-vitl-normal"


def report(progress: float, message: str) -> None:
    print(json.dumps({"progress": progress, "message": message}), flush=True)


def log(message: str = "") -> None:
    print(message, file=sys.stderr, flush=True)


def quiet_third_party_output() -> None:
    """Keep the log to what matters: no library deprecation notices or download bars."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.captureWarnings(True)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def friendly_error(error: Exception) -> str:
    """Short, actionable message for the Blender status bar."""
    name = type(error).__name__
    if name == "OutOfMemoryError":
        return "The GPU ran out of memory. Choose a smaller Depth Model or lower Depth Resolution."
    if name in {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}:
        return "Model weights are not downloaded yet. Allow online access in Blender for the first solve."
    if isinstance(error, (FileNotFoundError, ValueError)):
        return str(error)
    return f"{name}: {error}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aicm_worker", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--camera-model", choices=list(CAMERA_MODELS), default="pinhole"
    )
    parser.add_argument(
        "--depth-model",
        default=DEFAULT_DEPTH_MODEL,
        help="MoGe-2/3 Hugging Face repo id or checkpoint path, or 'none' to skip depth.",
    )
    parser.add_argument(
        "--depth-resolution",
        type=int,
        default=1024,
        help="Long edge in pixels for MoGe.",
    )
    parser.add_argument(
        "--default-height",
        type=float,
        default=1.6,
        help="Camera height (m) without a floor.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu or mps.")
    parser.add_argument(
        "--no-fp16", action="store_true", help="Run MoGe in full precision."
    )
    parser.add_argument("--no-mesh", action="store_true", help="Skip the proxy mesh.")
    parser.add_argument(
        "--no-fspy", action="store_true", help="Skip the .fspy project."
    )
    return parser.parse_args(argv)


def read_rgb(path: Path) -> np.ndarray:
    import cv2

    # imdecode handles non-ASCII Windows paths and applies EXIF orientation, like cv2.imread.
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def resize_long_edge(rgb: np.ndarray, long_edge: int) -> np.ndarray:
    import cv2

    height, width = rgb.shape[:2]
    scale = long_edge / max(height, width)
    if scale >= 1.0:
        return rgb
    size = (round(width * scale), round(height * scale))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def to_gltf(vectors: np.ndarray) -> np.ndarray:
    """Blender Z-up world vectors to glTF Y-up, undone by Blender's glTF importer."""
    return np.stack([vectors[..., 0], vectors[..., 2], -vectors[..., 1]], axis=-1)


def save_proxy_mesh(
    path: Path, rgb, geometry, rotation: np.ndarray, height: float
) -> None:
    """Textured mesh of the point map in Blender world space, built as in MoGe's infer script."""
    from moge.utils.io import save_glb

    try:
        import utils3d_moge as utils3d
    except ImportError:
        import utils3d

    rows, cols = geometry.depth.shape
    mask = geometry.mask & ~utils3d.np.depth_map_edge(geometry.depth, rtol=0.04)
    maps = [
        geometry.points_cv,
        rgb.astype(np.float32) / 255,
        utils3d.np.uv_map(rows, cols),
    ]
    if geometry.normals_cv is not None:
        maps.append(geometry.normals_cv)
    faces, vertices, _colors, uvs, *normals = utils3d.np.build_mesh_from_map(
        *maps, mask=mask, tri=True
    )

    vertices = vertices @ rotation.T + [0.0, 0.0, height]
    normals = to_gltf(normals[0] @ rotation.T) if normals else None
    save_glb(path, to_gltf(vertices), faces, uvs * [1, -1] + [0, 1], rgb, normals)


def log_context(args: argparse.Namespace, device) -> None:
    import torch

    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    log(
        f"AI Camera Match worker {__version__}, PyTorch {torch.__version__}, device {device}{gpu}"
    )
    log(f"Image: {args.image}")
    log(
        f"Lens model: {CAMERA_MODELS[args.camera_model]}; depth model: {args.depth_model}; "
        f"depth resolution: {args.depth_resolution} px"
    )
    log(f"Output folder: {args.out_dir}")


def log_summary(solution: dict) -> None:
    i, o, c = solution["intrinsics"], solution["orientation"], solution["camera"]
    u = o["uncertainty_deg"]
    lens_mm = 36 * i["focal_px"] / solution["image"]["width"]
    log()
    log("Result")
    log(
        f"  Lens:        {lens_mm:.1f} mm on a 36 mm wide sensor ({i['focal_px']:.0f} px), "
        f"FoV {np.rad2deg(i['hfov']):.1f}° x {np.rad2deg(i['vfov']):.1f}°"
    )
    if i["distortion"]:
        log(f"  Distortion:  {', '.join(f'{k:.4f}' for k in i['distortion'])}")
    for name in ("roll", "pitch"):
        spread = f" (± {u[name]:.2f}°)" if name in u else ""
        log(f"  {name.title() + ':':<13}{o[name + '_deg']:.2f}°{spread}")
    if c["height_source"] == "ground_plane":
        g = solution["ground"]
        log(
            f"  Height:      {c['height_m']:.2f} m above the floor "
            f"({g['inliers']:,} floor points, tilt {g['tilt_deg']:.1f}°)"
        )
    else:
        log(f"  Height:      {c['height_m']:.2f} m (fallback, no floor measured)")
    for warning in solution["warnings"]:
        log(f"  Warning:     {warning}")
    files = [solution[k] for k in ("plate", "proxy_mesh", "fspy") if solution[k]]
    log(f"  Files:       {', '.join(files + ['solution.json'])}")
    log(f"  Time:        {solution['seconds']:.1f} s on {solution['device']}")


def solve(args: argparse.Namespace) -> dict:
    from .models import moge_version, pick_device, run_geocalib, run_moge

    started = time.perf_counter()
    device = pick_device(args.device)
    log_context(args, device)
    if args.depth_model.lower() != "none":
        moge_version(args.depth_model)  # reject an unusable model before any slow work
    args.out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    report(0.05, "Reading image")
    rgb = read_rgb(args.image)
    height_px, width_px = rgb.shape[:2]
    log(f"Image size: {width_px} x {height_px} px")

    report(
        0.15,
        f"Calibrating lens and horizon with GeoCalib ({CAMERA_MODELS[args.camera_model]} lens)",
    )
    calibration = run_geocalib(rgb, args.camera_model, device)
    plate_rgb = (
        calibration.undistorted_rgb if calibration.undistorted_rgb is not None else rgb
    )
    hfov = fov_from_focal(calibration.focal_px, width_px)
    vfov = fov_from_focal(calibration.focal_px, height_px)
    roll, pitch = roll_pitch_from_up(calibration.up_cv)
    rotation = world_from_cv_rotation(calibration.up_cv)

    plate_path = args.out_dir / "plate.png"
    write_png(plate_path, plate_rgb)

    camera_height, height_source, ground, mesh_path = (
        args.default_height,
        "default",
        None,
        None,
    )
    if args.depth_model.lower() != "none":
        report(0.45, f"Estimating depth with MoGe ({Path(args.depth_model).name})")
        small = resize_long_edge(plate_rgb, args.depth_resolution)
        geometry = run_moge(
            small, args.depth_model, float(np.rad2deg(hfov)), device, not args.no_fp16
        )

        report(0.8, "Finding the floor")
        normals = (
            geometry.normals_cv @ rotation.T
            if geometry.normals_cv is not None
            else None
        )
        plane = fit_ground_plane(geometry.points_cv @ rotation.T, normals)
        if plane is None:
            warnings.append(
                f"No floor found below the camera, so the camera uses the fallback height of {args.default_height:.2f} m."
            )
        else:
            camera_height, height_source = plane.height, "ground_plane"
            ground = {
                "tilt_deg": plane.tilt_deg,
                "inliers": plane.inliers,
                "candidates": plane.candidates,
                "normal": plane.normal.tolist(),
            }
            if plane.tilt_deg > 5.0:
                warnings.append(
                    f"The floor is tilted {plane.tilt_deg:.1f}° against the horizon. "
                    "Check the horizon, or whether a table or slope was taken as the floor."
                )

        if not args.no_mesh:
            report(0.9, "Writing the proxy mesh")
            mesh_path = args.out_dir / "proxy_mesh.glb"
            save_proxy_mesh(mesh_path, small, geometry, rotation, camera_height)

    matrix_world = camera_matrix_world(calibration.up_cv, camera_height)

    fspy_path = None
    if not args.no_fspy:
        from .fspy import write_fspy_project

        fspy_path = args.out_dir / f"{args.image.stem}.fspy"
        write_fspy_project(
            fspy_path,
            plate_path.read_bytes(),
            width_px,
            height_px,
            hfov,
            vfov,
            matrix_world,
        )

    if calibration.distortion and np.abs(calibration.distortion).max() > 1e-6:
        warnings.append(
            "Lens distortion was removed: the camera matches the undistorted plate.png, not the original photo."
        )

    return {
        "format": "ai-camera-match-solution",
        "version": 1,
        "worker_version": __version__,
        "source_image": str(args.image.resolve()),
        "plate": plate_path.name,
        "proxy_mesh": mesh_path.name if mesh_path else None,
        "fspy": fspy_path.name if fspy_path else None,
        "image": {"width": width_px, "height": height_px},
        "intrinsics": {
            "camera_model": args.camera_model,
            "focal_px": calibration.focal_px,
            "hfov": hfov,
            "vfov": vfov,
            "principal_point_px": [width_px / 2.0, height_px / 2.0],
            "distortion": calibration.distortion,
        },
        "orientation": {
            "roll_deg": float(np.rad2deg(roll)),
            "pitch_deg": float(np.rad2deg(pitch)),
            "up_cv": calibration.up_cv.tolist(),
            "uncertainty_deg": calibration.uncertainty_deg,
        },
        "camera": {
            "matrix_world": matrix_world.tolist(),
            "height_m": camera_height,
            "height_source": height_source,
        },
        "ground": ground,
        "models": {
            "calibration": "GeoCalib",
            "depth": None if args.depth_model.lower() == "none" else args.depth_model,
        },
        "device": str(device),
        "seconds": round(time.perf_counter() - started, 2),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quiet_third_party_output()
    try:
        solution = solve(args)
    except Exception as error:  # reported to the add-on as a single JSON line
        logging.exception("Solve failed. Details:")
        print(json.dumps({"error": friendly_error(error)}), flush=True)
        return 1
    (args.out_dir / "solution.json").write_text(
        json.dumps(solution, indent=2), encoding="utf-8"
    )
    log_summary(solution)
    report(1.0, "Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
