# Import all the node classes from their respective files
from .omni_nodes import (
    OmniPartLoaderNode,
    OmniPartSegmentImage,
    OmniPartCombineSegments,
    OmniPartHideSegments,
    OmniPartGenerate3DDataNode,
    OmniPartSaveMeshNode,
    OmniPartExportObjects,
)

from .omni_texture import OmniPartSaveTexturedMesh

# Map the internal class names to the names ComfyUI will use
NODE_CLASS_MAPPINGS = {
    "OmniPartLoader": OmniPartLoaderNode,
    "OmniPartSegmentImage": OmniPartSegmentImage,
    "OmniPartCombineSegments": OmniPartCombineSegments,
    "OmniPartHideSegments": OmniPartHideSegments,
    "OmniPartGenerate3DData": OmniPartGenerate3DDataNode,
    "OmniPartSaveMesh": OmniPartSaveMeshNode,
    "OmniPartSaveTexturedMesh": OmniPartSaveTexturedMesh,
    "OmniPartExportObjects": OmniPartExportObjects,
}

# Map the internal class names to the names that will be displayed in the menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniPartLoader": "Load OmniPart Models",
    "OmniPartSegmentImage": "Segment Image",
    "OmniPartCombineSegments": "Combine Segments",
    "OmniPartHideSegments": "Hide Segments",
    "OmniPartGenerate3DData": "Generate 3D Data",
    "OmniPartSaveMesh": "Save Untextured Mesh",
    "OmniPartSaveTexturedMesh": "Save Textured Mesh",
    "OmniPartExportObjects": "Export Objects from Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']