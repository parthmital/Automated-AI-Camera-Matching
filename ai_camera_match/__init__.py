# SPDX-FileCopyrightText: 2026 Parth Mital
# SPDX-License-Identifier: GPL-3.0-or-later

"""AI Camera Match: fSpy-style camera matching from a single photo, without vanishing lines.

Combines GeoCalib (camera intrinsics and gravity) and MoGe (metric geometry) in a separate
Python environment, then builds the matched camera, plate, floor and proxy mesh in Blender.
See THIRD_PARTY_NOTICES.md for the projects this builds on.
"""

import bpy

from . import operators, properties, ui

CLASSES = properties.CLASSES + operators.CLASSES + ui.CLASSES


def menu_import(self, context):
    self.layout.operator(
        operators.AICM_OT_import_solution.bl_idname, text="AI Camera Match (.json)"
    )


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    properties.register_runtime_props()
    bpy.types.TOPBAR_MT_file_import.append(menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    properties.unregister_runtime_props()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
