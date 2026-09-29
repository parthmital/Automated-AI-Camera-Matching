import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "ai_camera_match" / "worker")
)

from aicm_worker.fspy import write_fspy_project  # noqa: E402
from aicm_worker.geometry import (  # noqa: E402
    camera_matrix_world,
    fit_ground_plane,
    fov_from_focal,
    roll_pitch_from_up,
    world_from_cv_rotation,
)


def geocalib_up(roll: float, pitch: float) -> np.ndarray:
    """GeoCalib's Gravity.from_rp: the world up vector in the OpenCV camera frame."""
    sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
    return np.array([-sr * cp, -cr * cp, sp])


def test_level_camera_looks_along_world_y():
    matrix = camera_matrix_world(geocalib_up(0.0, 0.0), height=1.6)
    rotation = matrix[:3, :3]
    np.testing.assert_allclose(
        rotation @ [0, 0, -1], [0, 1, 0], atol=1e-12
    )  # Blender camera forward
    np.testing.assert_allclose(rotation @ [0, 1, 0], [0, 0, 1], atol=1e-12)  # camera up
    np.testing.assert_allclose(matrix[:3, 3], [0, 0, 1.6])


@pytest.mark.parametrize(
    "roll_deg, pitch_deg", [(0, -30), (10, 20), (-25, -60), (170, 5)]
)
def test_rotation_matches_roll_and_pitch(roll_deg, pitch_deg):
    roll, pitch = np.deg2rad(roll_deg), np.deg2rad(pitch_deg)
    rotation = camera_matrix_world(geocalib_up(roll, pitch), 0.0)[:3, :3]

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    forward = rotation @ [0, 0, -1]
    assert forward[2] == pytest.approx(np.sin(pitch))  # positive pitch looks up
    assert forward[0] == pytest.approx(0.0, abs=1e-12)  # zero yaw
    # The camera's right axis tilts out of the horizon exactly as GeoCalib's up vector says.
    assert (rotation @ [1, 0, 0])[2] == pytest.approx(
        -np.sin(roll) * np.cos(pitch), abs=1e-12
    )
    assert roll_pitch_from_up(geocalib_up(roll, pitch)) == pytest.approx((roll, pitch))


def test_world_rotation_maps_up_to_z():
    up = geocalib_up(0.3, -0.4)
    np.testing.assert_allclose(world_from_cv_rotation(up) @ up, [0, 0, 1], atol=1e-12)


def test_straight_down_camera_is_defined():
    rotation = camera_matrix_world(np.array([0.0, 0.0, -1.0]), 0.0)[:3, :3]
    np.testing.assert_allclose(rotation @ [0, 0, -1], [0, 0, -1], atol=1e-12)
    assert np.isfinite(rotation).all()


def test_fov_from_focal():
    assert fov_from_focal(500.0, 1000.0) == pytest.approx(np.pi / 2)


def plane_points(z: float, count: int, rng, extent: float = 3.0) -> np.ndarray:
    xy = rng.uniform(-extent, extent, size=(count, 2)) + [0, extent + 1]
    return np.column_stack([xy, np.full(count, z) + rng.normal(0, 0.005, count)])


def test_ground_plane_prefers_floor_under_larger_table():
    rng = np.random.default_rng(0)
    floor = plane_points(-1.5, 3000, rng)
    table = plane_points(-0.7, 5000, rng, extent=1.0)
    wall = np.column_stack(
        [rng.uniform(-3, 3, 4000), np.full(4000, 7.0), rng.uniform(-1.5, 1.5, 4000)]
    )
    points = np.concatenate([floor, table, wall, np.full((100, 3), np.inf)])

    plane = fit_ground_plane(points)
    assert plane is not None
    assert plane.height == pytest.approx(1.5, abs=0.01)
    assert plane.tilt_deg < 1.0


def test_ground_plane_uses_normals_to_reject_walls():
    rng = np.random.default_rng(1)
    floor = plane_points(-1.2, 2000, rng)
    wall = np.column_stack(
        [rng.uniform(-3, 3, 8000), np.full(8000, 5.0), rng.uniform(-3.0, -0.1, 8000)]
    )
    normals = np.concatenate(
        [np.tile([0, 0, 1.0], (2000, 1)), np.tile([0, -1.0, 0], (8000, 1))]
    )

    plane = fit_ground_plane(np.concatenate([floor, wall]), normals)
    assert plane.height == pytest.approx(1.2, abs=0.01)
    assert plane.candidates == 2000


def test_ground_plane_follows_slightly_tilted_floor():
    rng = np.random.default_rng(3)
    floor = plane_points(0.0, 6000, rng)
    tilt = np.deg2rad(4.0)
    floor[:, 2] = -1.4 + np.tan(tilt) * floor[:, 1]  # rises by 7 cm per metre ahead

    plane = fit_ground_plane(floor)
    assert plane.tilt_deg == pytest.approx(4.0, abs=0.2)
    assert plane.inliers > 0.95 * len(floor)
    # Height is measured where the floor is seen, so the visible floor centres on Z = 0.
    assert plane.height == pytest.approx(-np.median(floor[:, 2]), abs=0.02)


def test_ground_plane_none_without_points_below_camera():
    rng = np.random.default_rng(2)
    assert fit_ground_plane(plane_points(2.0, 5000, rng)) is None


def test_fspy_project_layout(tmp_path):
    matrix = camera_matrix_world(geocalib_up(0.1, -0.2), 1.7)
    path = tmp_path / "shot.fspy"
    write_fspy_project(path, b"IMAGE", 1920, 1080, 1.0, 0.6, matrix)

    data = path.read_bytes()
    # Header checks mirror fSpy-Blender's parser (fspy_blender/fspy.py).
    assert struct.unpack("<I", data[:4])[0] == 2037412710
    version, state_size, image_size = struct.unpack("<III", data[4:16])
    assert (version, image_size) == (1, 5)
    state = json.loads(data[16 : 16 + state_size])
    assert data[16 + state_size :] == b"IMAGE"
    params = state["cameraParameters"]
    np.testing.assert_allclose(params["cameraTransform"]["rows"], matrix)
    assert params["horizontalFieldOfView"] == 1.0
    assert (params["imageWidth"], params["imageHeight"]) == (1920, 1080)
    assert params["principalPoint"] == {"x": 0.0, "y": 0.0}
    assert state["calibrationSettingsBase"]["referenceDistanceUnit"] == "Meters"
