"""Write solved cameras as fSpy project files.

Binary layout follows fSpy's project_file_format.md (https://github.com/stuffmatic/fSpy,
GPL-3.0, Per Gantelius). The state holds only the fields read by the fSpy-Blender importer
(https://github.com/stuffmatic/fSpy-Blender), so the file is meant for that importer and
other .fspy readers, not for re-editing in the fSpy app, which re-solves from control points.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

FSPY_FILE_ID = b"fspy"
FSPY_PROJECT_VERSION = 1


def write_fspy_project(
    path: str | Path,
    image_bytes: bytes,
    image_width: int,
    image_height: int,
    hfov: float,
    vfov: float,
    matrix_world: np.ndarray,
) -> None:
    """Write an fSpy project with a centred principal point and metre units."""
    matrix_world = np.asarray(matrix_world, dtype=np.float64)
    state = {
        "cameraParameters": {
            # fSpy's relative image-plane coordinates, where (0, 0) is the image centre.
            "principalPoint": {"x": 0.0, "y": 0.0},
            "viewTransform": {"rows": np.linalg.inv(matrix_world).tolist()},
            "cameraTransform": {"rows": matrix_world.tolist()},
            "horizontalFieldOfView": float(hfov),
            "verticalFieldOfView": float(vfov),
            "imageWidth": int(image_width),
            "imageHeight": int(image_height),
        },
        "calibrationSettingsBase": {"referenceDistanceUnit": "Meters"},
    }
    state_bytes = json.dumps(state).encode("utf-8")
    header = FSPY_FILE_ID + struct.pack(
        "<III", FSPY_PROJECT_VERSION, len(state_bytes), len(image_bytes)
    )
    Path(path).write_bytes(header + state_bytes + image_bytes)
