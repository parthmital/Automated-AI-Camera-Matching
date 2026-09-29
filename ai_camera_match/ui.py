"""3D Viewport sidebar panel (N panel > Camera Match)."""

import bpy

from .properties import prefs


class AICM_PT_main(bpy.types.Panel):
    bl_label = "AI Camera Match"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Camera Match"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.aicm
        wm = context.window_manager

        if not prefs(context).env_ready():
            box = layout.box()
            box.label(text="Solver environment not installed", icon="ERROR")
            box.operator("preferences.addon_show", text="Open Preferences").module = (
                __package__
            )

        layout.prop(settings, "image")
        layout.prop(settings, "camera_model")

        layout.prop(settings, "estimate_height")
        col = layout.column()
        col.active = settings.estimate_height
        col.prop(settings, "depth_model")
        if settings.depth_model == "CUSTOM":
            col.prop(settings, "custom_depth_model")
        col.prop(settings, "depth_resolution")
        col.prop(settings, "import_mesh")
        layout.prop(settings, "default_height")
        layout.prop(settings, "add_floor")

        if wm.aicm_busy:
            layout.progress(factor=wm.aicm_progress, text=wm.aicm_status)
            layout.label(text="Press Esc to cancel")
        else:
            layout.operator("aicm.solve", icon="CAMERA_DATA")
            if wm.aicm_status:
                layout.label(text=wm.aicm_status)
        row = layout.row(align=True)
        row.operator("aicm.import_solution", icon="IMPORT", text="Import Solution")
        row.operator("aicm.open_folder", icon="FILE_FOLDER", text="")


CLASSES = (AICM_PT_main,)
