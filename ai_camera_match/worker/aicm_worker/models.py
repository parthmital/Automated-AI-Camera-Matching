"""Thin wrappers around the third-party models.

- GeoCalib (https://github.com/cvg/GeoCalib): Veicht, Sarlin, Lindenberger, Pollefeys,
  "GeoCalib: Learning Single-image Calibration with Geometric Optimization", ECCV 2024.
  Code Apache-2.0, weights CC-BY 4.0.
- MoGe (https://github.com/microsoft/MoGe): Wang et al., MoGe / MoGe-2 / MoGe-3,
  Microsoft Research. Code and weights MIT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# GeoCalib ships two checkpoints; distortion-aware camera models need the "distorted" one.
GEOCALIB_WEIGHTS = {
    "pinhole": "pinhole",
    "simple_radial": "distorted",
    "radial": "distorted",
    "simple_divisional": "distorted",
}


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class Calibration:
    focal_px: float
    up_cv: np.ndarray
    distortion: list[float]
    uncertainty_deg: dict[str, float]
    undistorted_rgb: np.ndarray | None
    """Undistorted plate (H, W, 3) uint8, or None for the pinhole model."""


def run_geocalib(
    rgb: np.ndarray, camera_model: str, device: torch.device
) -> Calibration:
    """Estimate focal length, lens distortion and gravity from an (H, W, 3) uint8 RGB image."""
    from geocalib import GeoCalib

    model = GeoCalib(weights=GEOCALIB_WEIGHTS[camera_model]).to(device)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
    result = model.calibrate(image, camera_model=camera_model)

    camera = result["camera"]
    # GeoCalib predicts a single focal length (fx == fy) with the principal point at the centre.
    focal = float(camera.f[0, 1])
    distortion = (
        camera.dist[0, : camera.num_dist_params()].tolist()
        if camera_model != "pinhole"
        else []
    )
    undistorted = None
    if camera_model != "pinhole":
        plate = camera.undistort_image(image[None])[0]
        undistorted = (
            (plate.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
            .round()
            .astype(np.uint8)
        )

    uncertainty = {
        name: float(np.rad2deg(result[f"{name}_uncertainty"][0].item()))
        for name in ("roll", "pitch", "vfov")
        if f"{name}_uncertainty" in result
    }
    calibration = Calibration(
        focal_px=focal,
        up_cv=result["gravity"].vec3d[0].double().cpu().numpy(),
        distortion=distortion,
        uncertainty_deg=uncertainty,
        undistorted_rgb=undistorted,
    )
    del model
    free_memory(device)
    return calibration


def moge_version(model_id: str) -> str:
    """MoGe model class version for a Hugging Face repo id or local checkpoint path."""
    name = model_id.lower().replace("_", "-")
    if "moge-3" in name:
        return "v3"
    if "moge-2" in name:
        return "v2"
    raise ValueError(
        f"'{model_id}' is not a MoGe-2 or MoGe-3 model. Metric scale needs MoGe-2 or later; "
        "the name must contain 'moge-2' or 'moge-3'."
    )


@dataclass
class Geometry:
    points_cv: np.ndarray
    """(H, W, 3) metric points in the OpenCV camera frame; invalid pixels are inf."""
    normals_cv: np.ndarray | None
    mask: np.ndarray
    depth: np.ndarray


def run_moge(
    rgb: np.ndarray, model_id: str, hfov_deg: float, device: torch.device, fp16: bool
) -> Geometry:
    """Metric point map for an (H, W, 3) uint8 image, constrained to the given horizontal FoV."""
    from moge.model import import_model_class_by_version

    version = moge_version(model_id)
    model = (
        import_model_class_by_version(version)
        .from_pretrained(model_id)
        .to(device)
        .eval()
    )
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
    use_fp16 = fp16 and device.type == "cuda"
    with torch.no_grad():
        output = model.infer(image, fov_x=hfov_deg, use_fp16=use_fp16)

    geometry = Geometry(
        points_cv=output["points"].float().cpu().numpy(),
        normals_cv=(
            output["normal"].float().cpu().numpy() if "normal" in output else None
        ),
        mask=output["mask"].cpu().numpy(),
        depth=output["depth"].float().cpu().numpy(),
    )
    del model
    free_memory(device)
    return geometry
