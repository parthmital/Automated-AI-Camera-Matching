"""Camera and ground-plane geometry shared by the solver. Pure NumPy, no model code.

Frames used throughout:
- OpenCV camera: +X right, +Y down, +Z forward (GeoCalib and MoGe outputs).
- Blender camera: +X right, +Y up, -Z forward.
- Blender world: Z up; the solved camera looks along +Y (yaw = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# OpenCV camera axes expressed in Blender camera axes (and vice versa).
CV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0])


def world_from_cv_rotation(up_cv: np.ndarray) -> np.ndarray:
    """Rotation taking OpenCV camera vectors to a Z-up world with zero yaw.

    Args:
        up_cv: World up direction expressed in the OpenCV camera frame. GeoCalib's
            ``gravity.vec3d`` is this vector: (0, -1, 0) for a level camera.
    """
    up = np.asarray(up_cv, dtype=np.float64)
    up = up / np.linalg.norm(up)

    # World +Y is the camera's forward direction projected onto the horizontal plane.
    # Looking straight up or down leaves forward undefined, so fall back to camera up.
    forward = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.999:
        forward = np.array([0.0, -1.0, 0.0])
    y_axis = forward - (forward @ up) * up
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, up)
    return np.stack([x_axis, y_axis, up])


def roll_pitch_from_up(up_cv: np.ndarray) -> tuple[float, float]:
    """Roll and pitch in radians, using GeoCalib's sign conventions (pitch > 0 looks up)."""
    x, y, z = np.asarray(up_cv, dtype=np.float64) / np.linalg.norm(up_cv)
    pitch = float(np.arcsin(np.clip(z, -1.0, 1.0)))
    roll = float(np.arctan2(-x, -y))
    return roll, pitch


def camera_matrix_world(up_cv: np.ndarray, height: float) -> np.ndarray:
    """4x4 Blender ``matrix_world`` for a camera at ``(0, 0, height)``."""
    matrix = np.eye(4)
    matrix[:3, :3] = world_from_cv_rotation(up_cv) @ CV_TO_BLENDER_CAMERA
    matrix[2, 3] = height
    return matrix


def fov_from_focal(focal_px: float, size_px: float) -> float:
    """Field of view in radians for a focal length and image extent in pixels."""
    return float(2.0 * np.arctan(size_px / (2.0 * focal_px)))


@dataclass
class GroundPlane:
    height: float
    """Camera height above the observed floor in metres (median over floor inliers)."""
    normal: np.ndarray
    """Fitted plane normal in world space, pointing up."""
    tilt_deg: float
    """Angle between the fitted normal and the gravity-derived world up."""
    inliers: int
    candidates: int


def fit_ground_plane(
    points_world: np.ndarray,
    normals_world: np.ndarray | None = None,
    max_normal_angle_deg: float = 20.0,
    min_support_ratio: float = 0.5,
    min_inliers: int = 200,
    refine_iterations: int = 3,
) -> GroundPlane | None:
    """Find the floor below the camera in a world-oriented metric point cloud.

    Points are camera-relative (camera at the origin) and already rotated so +Z is up.
    Heights of upward-facing points below the camera are voted into bins; the lowest
    bin with at least ``min_support_ratio`` of the peak support seeds the floor, so a
    large tabletop does not win over a visible floor beneath it. A plane is then refitted
    to points near it a few times, which follows a floor that the depth model tilts slightly
    against the gravity estimate. The camera height is the median depth of the floor
    inliers below the camera, so the visible floor lands at Z = 0 on average.

    Returns None when too few points support any horizontal plane.
    """
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] < 0.0)
    if normals_world is not None:
        normals = np.asarray(normals_world, dtype=np.float64).reshape(-1, 3)
        valid &= normals[:, 2] > np.cos(np.deg2rad(max_normal_angle_deg))
    candidates = points[valid]
    if len(candidates) < min_inliers:
        return None

    # Bin width scales with scene size: 2% of the median distance, clamped to 1-25 cm.
    distance = np.median(np.linalg.norm(candidates, axis=1))
    tolerance = float(np.clip(0.02 * distance, 0.01, 0.25))
    heights = candidates[:, 2]
    edges = np.arange(heights.min(), heights.max() + 2 * tolerance, tolerance)
    counts, edges = np.histogram(heights, bins=edges)
    # Count each bin together with its neighbours so a plane split across a bin edge still wins.
    support = np.convolve(counts, np.ones(3), mode="same")
    floor_bin = int(np.argmax(support >= min_support_ratio * support.max()))
    floor_z = 0.5 * (edges[floor_bin] + edges[floor_bin + 1])

    normal, offset = np.array([0.0, 0.0, 1.0]), -floor_z
    for _ in range(refine_iterations):
        near = candidates[np.abs(candidates @ normal + offset) <= 1.5 * tolerance]
        if len(near) < min_inliers:
            return None
        centroid = near.mean(axis=0)
        normal = np.linalg.svd(near - centroid, full_matrices=False)[2][-1]
        normal = normal if normal[2] > 0 else -normal
        if normal[2] < np.cos(np.deg2rad(max_normal_angle_deg)):
            return None
        offset = -(normal @ centroid)

    return GroundPlane(
        height=float(-np.median(near[:, 2])),
        normal=normal,
        tilt_deg=float(np.rad2deg(np.arccos(np.clip(normal[2], -1.0, 1.0)))),
        inliers=int(len(near)),
        candidates=int(len(candidates)),
    )
