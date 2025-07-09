import os
import sys
import trimesh
import folder_paths
import numpy as np
import torch
import gc

# Add the custom node's directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import utilities from the project's modules
from modules.part_synthesis.process_utils import save_parts_outputs
from omni_nodes import cleanup_memory

def _get_random_int():
    """Helper to get a random integer for directory names."""
    return np.random.randint(10000)

class OmniPartTextureMeshNode:
    """
    Saves the generated 3D parts as a single, textured mesh file.
    This node leverages the underlying OmniPart pipeline's capability to
    texture the mesh parts based on the input image.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "omni_3d_data": ("OMNIPART_3D_DATA",),
                "output_prefix": ("STRING", {"default": "omni_parts_textured"}),
                "simplify_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_file_path",)
    FUNCTION = "save_textured_mesh"
    CATEGORY = "OmniPart"

    def save_textured_mesh(self, omni_3d_data, output_prefix, simplify_ratio):
        # 🧹 Perform aggressive cleanup before node execution to free VRAM
        print("Performing full cleanup: Clearing garbage and emptying CUDA cache.")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Original cleanup call for any other specific actions
        cleanup_memory()

        synthesis_output = omni_3d_data["synthesis_output"]
        
        # Use a temporary directory for intermediate part files
        temp_dir = folder_paths.get_temp_directory()
        parts_output_dir = os.path.join(temp_dir, f"{output_prefix}_{_get_random_int()}")
        os.makedirs(parts_output_dir, exist_ok=True)
        
        # 1. Save individual parts, with texturing enabled.
        # This relies on the synthesis_output containing all necessary information
        # (geometry, masks, image context) for the texturing process.
        save_parts_outputs(
            synthesis_output,
            output_dir=parts_output_dir,
            simplify_ratio=simplify_ratio,
            save_video=False,
            save_glb=True,
            textured=True  # Enable texturing
        )
        
        # 2. Merge the textured parts into a single scene
        scene_list = []
        part_files = sorted([
            os.path.join(parts_output_dir, f) 
            for f in os.listdir(parts_output_dir) if f.startswith('part') and f.endswith('.glb')
        ])
        
        if not part_files:
            raise Exception("No textured part .glb files were generated to merge.")
            
        for part_path in part_files:
            # Load the textured mesh part. Trimesh automatically handles materials/textures within the GLB.
            part_mesh = trimesh.load(part_path, force='mesh')
            scene_list.append(part_mesh)

        if not scene_list:
            raise Exception("Could not load any valid meshes from the output directory.")

        # 3. Create a scene and export the final combined textured mesh
        combined_scene = trimesh.Scene(scene_list)
        
        final_output_dir = folder_paths.get_output_directory()
        final_mesh_path = os.path.join(final_output_dir, f"{output_prefix}_merged.glb")
        combined_scene.export(final_mesh_path)

        print("Textured mesh saving node complete.")
        cleanup_memory()
        
        return (final_mesh_path,)