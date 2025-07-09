# __init__.py

from .omni_nodes import (
    OmniPartLoaderNode,
    OmniPartSegmenterNode,
    OmniPartMergeNode,
    OmniPartGenerate3DDataNode,
    OmniPartSaveMeshNode
)

# Import the new debug node
from .omni_texture import OmniPartTextureMeshNode

NODE_CLASS_MAPPINGS = {
    "OmniPartLoader": OmniPartLoaderNode,
    "OmniPartSegmenter": OmniPartSegmenterNode,
    "OmniPartMergeSegments": OmniPartMergeNode,
    "OmniPartGenerate3DData": OmniPartGenerate3DDataNode,
    "OmniPartSaveMesh": OmniPartSaveMeshNode,
    "OmniPartTextureMesh (Debug)": OmniPartTextureMeshNode, # Add the new node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniPartLoader": "Load OmniPart Models",
    "OmniPartSegmenter": "Segment Image (OmniPart)",
    "OmniPartMergeSegments": "Merge Segments (OmniPart)",
    "OmniPartGenerate3DData": "Generate 3D Data (OmniPart)",
    "OmniPartSaveMesh": "Save Mesh (OmniPart)",
    "OmniPartTextureMesh (Debug)": "Texture Mesh (Debug)", # Add its display name
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
