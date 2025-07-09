import os
import sys
import torch
import numpy as np
from PIL import Image
import gc
import shutil
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import folder_paths
import comfy.model_management as mm # Import ComfyUI's memory manager

# Add the custom node's directory to the Python path
# This allows us to import from the sub-directories like 'modules'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now we can import from the OmniPart modules
from modules.label_2d_mask.label_parts import (
    prepare_image, get_sam_mask, get_mask, clean_segment_edges,
    resize_and_pad_to_square, size_th as DEFAULT_SIZE_TH
)
from modules.label_2d_mask.visualizer import Visualizer
from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
from modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline
# --- FIX: Removed the unnecessary 'merge_parts' import ---
from modules.part_synthesis.process_utils import save_parts_outputs
from modules.inference_utils import (
    load_img_mask, prepare_bbox_gen_input, prepare_part_synthesis_input,
    gen_mesh_from_bounds, vis_voxel_coords
)
from segment_anything import SamAutomaticMaskGenerator, build_sam
from transformers import AutoModelForImageSegmentation
import cv2
import trimesh

# --- Global Variables for Models and Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# --- Class-based holder for models to improve state persistence ---
class OmniPartModels:
    sam_mask_generator = None
    rmbg_model = None
    bbox_gen_model = None
    part_synthesis_pipeline = None

# Global instance to hold the models
MODELS = OmniPartModels()


# --- Helper Functions ---

def tensor_to_pil(tensor):
    """Converts a ComfyUI IMAGE tensor (B, H, W, C) to a PIL image."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    image_np = tensor.mul(255).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(image_np)

def pil_to_tensor(image):
    """Converts a PIL image (H, W, C) to a ComfyUI IMAGE tensor (1, H, W, C)."""
    image_np = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def cleanup_memory(*models_to_unload):
    """Unloads models and clears CUDA cache using ComfyUI's manager."""
    for model in models_to_unload:
        if model is not None:
            if hasattr(model, 'predictor') and hasattr(model.predictor, 'model'):
                 model.predictor.model.to('cpu')
            elif hasattr(model, 'to'):
                 model.to('cpu')

    gc.collect()
    if torch.cuda.is_available():
        mm.soft_empty_cache()


# --- Node Definitions ---

class OmniPartLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"force_reload": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("loaded",)
    FUNCTION = "load_models"
    CATEGORY = "OmniPart"

    def load_models(self, force_reload):
        ckpt_dir = os.path.join(current_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

        if MODELS.sam_mask_generator is None or force_reload:
            print("Loading SAM model...")
            sam_ckpt_path = hf_hub_download("omnipart/OmniPart_modules", "sam_vit_h_4b8939.pth", local_dir=ckpt_dir)
            sam_model = build_sam(checkpoint=sam_ckpt_path).to(device=DEVICE)
            MODELS.sam_mask_generator = SamAutomaticMaskGenerator(sam_model)
            cleanup_memory(MODELS.sam_mask_generator)

        if MODELS.rmbg_model is None or force_reload:
            print("Loading BriaRMBG 2.0 model...")
            MODELS.rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
            cleanup_memory(MODELS.rmbg_model)

        if MODELS.part_synthesis_pipeline is None or force_reload:
            print("Loading PartSynthesis model...")
            MODELS.part_synthesis_pipeline = OmniPartImageTo3DPipeline.from_pretrained('omnipart/OmniPart')
            cleanup_memory(MODELS.part_synthesis_pipeline)

        if MODELS.bbox_gen_model is None or force_reload:
            print("Loading BboxGen model...")
            partfield_ckpt_path = hf_hub_download("omnipart/OmniPart_modules", "partfield_encoder.ckpt", local_dir=ckpt_dir)
            bbox_gen_ckpt_path = hf_hub_download("omnipart/OmniPart_modules", "bbox_gen.ckpt", local_dir=ckpt_dir)
            config_path = os.path.join(current_dir, "configs", "bbox_gen.yaml")
            bbox_gen_config = OmegaConf.load(config_path).model.args
            bbox_gen_config.partfield_encoder_path = partfield_ckpt_path
            MODELS.bbox_gen_model = BboxGen(bbox_gen_config)
            MODELS.bbox_gen_model.load_state_dict(torch.load(bbox_gen_ckpt_path), strict=False)
            MODELS.bbox_gen_model.eval().half()
            cleanup_memory(MODELS.bbox_gen_model)

        print("All OmniPart models loaded successfully.")
        return (True,)


class OmniPartSegmenterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "loader": ("BOOLEAN", {"forceInput": True}),
                "image": ("IMAGE",),
                "size_threshold": ("INT", {"default": 1800, "min": 100, "max": 8000, "step": 100}),
            }
        }
    RETURN_TYPES = ("IMAGE", "OMNIPART_STATE")
    FUNCTION = "segment_image"
    CATEGORY = "OmniPart"

    def segment_image(self, loader, image, size_threshold):
        if MODELS.sam_mask_generator is None or MODELS.rmbg_model is None:
            raise Exception("Models not loaded. Please use the OmniPartLoader node first.")

        MODELS.sam_mask_generator.predictor.model.to(DEVICE)
        MODELS.rmbg_model.to(DEVICE)
        
        pil_image = tensor_to_pil(image)
        output_dir = folder_paths.get_temp_directory()
        img_name = f"omni_input_{np.random.randint(100000)}"
        
        processed_image = prepare_image(pil_image, rmbg_net=MODELS.rmbg_model)
        processed_image = resize_and_pad_to_square(processed_image)
        white_bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
        white_bg_img = Image.alpha_composite(white_bg, processed_image.convert("RGBA"))
        image_np = np.array(white_bg_img.convert('RGB'))
        
        rgba_path = os.path.join(output_dir, f"{img_name}_processed.png")
        processed_image.save(rgba_path)
        
        visual = Visualizer(image_np)
        group_ids, _ = get_sam_mask(image_np, MODELS.sam_mask_generator, visual, None, rgba_image=processed_image, img_name=img_name, save_dir=output_dir, size_threshold=size_threshold)
        
        seg_vis_np = np.ones_like(image_np) * 255
        unique_ids = np.unique(group_ids)
        unique_ids = unique_ids[unique_ids >= 0]
        for i, unique_id in enumerate(unique_ids):
            color = np.array([(i * 50 + 80) % 256, (i * 120 + 40) % 256, (i * 180 + 20) % 256])
            mask = (group_ids == unique_id)
            seg_vis_np[mask] = color
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
                cv2.putText(seg_vis_np, str(unique_id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        seg_img_pil = Image.fromarray(seg_vis_np)
        
        omni_state = {
            "image_np": image_np, "processed_image_path": rgba_path, "group_ids": group_ids,
            "original_group_ids": group_ids, "img_name": img_name, "output_dir": output_dir,
            "size_threshold": size_threshold
        }
        cleanup_memory(MODELS.sam_mask_generator, MODELS.rmbg_model)
        return (pil_to_tensor(seg_img_pil), omni_state)


class OmniPartMergeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "omni_state": ("OMNIPART_STATE",),
                "merge_string": ("STRING", {"multiline": True, "default": "0,1;3,4"}),
            }
        }
    RETURN_TYPES = ("IMAGE", "OMNIPART_STATE")
    FUNCTION = "merge_segments"
    CATEGORY = "OmniPart"

    def merge_segments(self, omni_state, merge_string):
        if MODELS.sam_mask_generator is None:
            raise Exception("Models not loaded. Please use the OmniPartLoader node first.")

        MODELS.sam_mask_generator.predictor.model.to(DEVICE)
        
        image_np, group_ids, img_name, output_dir, size_threshold = (
            omni_state["image_np"], omni_state["original_group_ids"], omni_state["img_name"],
            omni_state["output_dir"], omni_state["size_threshold"]
        )
        processed_image = Image.open(omni_state["processed_image_path"])
        
        merge_groups = []
        if merge_string:
            for group_set in merge_string.split(';'):
                try:
                    ids = [int(x) for x in group_set.split(',') if x.strip()]
                    if ids: merge_groups.append(ids)
                except ValueError:
                    print(f"Could not parse merge group: {group_set}")
        
        visual = Visualizer(image_np)
        new_group_ids, _ = get_sam_mask(image_np, MODELS.sam_mask_generator, visual, merge_groups=merge_groups, existing_group_ids=group_ids, rgba_image=processed_image, skip_split=True, img_name=img_name, save_dir=output_dir, size_threshold=size_threshold)
        new_group_ids = clean_segment_edges(new_group_ids)
        
        merged_vis_np = np.ones_like(image_np) * 255
        new_unique_ids = np.unique(new_group_ids)
        new_unique_ids = new_unique_ids[new_unique_ids >= 0]
        for i, unique_id in enumerate(new_unique_ids):
            color = np.array([(i * 50 + 80) % 256, (i * 120 + 40) % 256, (i * 180 + 20) % 256])
            mask = (new_group_ids == unique_id)
            merged_vis_np[mask] = color
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
                cv2.putText(merged_vis_np, str(unique_id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        merged_img_pil = Image.fromarray(merged_vis_np)
        
        save_mask = new_group_ids + 1
        save_mask = save_mask.reshape(518, 518, 1).repeat(3, axis=-1)
        mask_save_path = os.path.join(output_dir, f"{img_name}_mask.tiff")
        cv2.imwrite(mask_save_path, save_mask.astype(np.float32))

        omni_state.update({"group_ids": new_group_ids, "mask_path": mask_save_path})
        cleanup_memory(MODELS.sam_mask_generator)
        return (pil_to_tensor(merged_img_pil), omni_state)


class OmniPartGenerate3DDataNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "omni_state": ("OMNIPART_STATE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                # --- ADDED CONTROLS FOR MESH DETAIL ---
                "voxel_steps": ("INT", {"default": 25, "min": 10, "max": 100, "step": 1, "tooltop": "Steps for initial shape generation. Higher is more defined."}),
                "voxel_cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 15.0, "step": 0.5, "tooltop": "CFG for initial shape generation."}),
                "slat_steps": ("INT", {"default": 25, "min": 10, "max": 100, "step": 1, "tooltop": "Steps for final detail generation. Higher is more detailed."}),
                "slat_cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 15.0, "step": 0.5, "tooltop": "CFG for final detail generation."}),
            }
        }
    RETURN_TYPES = ("OMNIPART_3D_DATA",)
    FUNCTION = "generate_3d_data"
    CATEGORY = "OmniPart"

    def generate_3d_data(self, omni_state, seed, voxel_steps, voxel_cfg, slat_steps, slat_cfg):
        if MODELS.bbox_gen_model is None or MODELS.part_synthesis_pipeline is None:
            raise Exception("Models not loaded. Please use the OmniPartLoader node first.")
        
        MODELS.bbox_gen_model.to(DEVICE)
        MODELS.part_synthesis_pipeline.to(DEVICE)
        
        img_path, mask_path, output_dir = omni_state["processed_image_path"], omni_state["mask_path"], omni_state["output_dir"]
        
        img_white_bg, img_black_bg, ordered_mask_input, _ = load_img_mask(img_path, mask_path)
        
        # Use the new user-defined parameters for voxel generation
        voxel_coords = MODELS.part_synthesis_pipeline.get_coords(
            img_black_bg, 
            num_samples=1, 
            seed=seed, 
            sparse_structure_sampler_params={"steps": voxel_steps, "cfg_strength": voxel_cfg}
        )
        voxel_coords_path = os.path.join(output_dir, "voxel_coords.npy")
        np.save(voxel_coords_path, voxel_coords.cpu().numpy())
        
        bbox_gen_input = prepare_bbox_gen_input(voxel_coords_path, img_white_bg, ordered_mask_input)
        bbox_gen_output = MODELS.bbox_gen_model.generate(bbox_gen_input)
        bboxes_path = os.path.join(output_dir, "bboxes.npy")
        np.save(bboxes_path, bbox_gen_output['bboxes'][0])
        
        cleanup_memory(MODELS.bbox_gen_model)
        
        part_synthesis_input = prepare_part_synthesis_input(voxel_coords_path, bboxes_path, ordered_mask_input)
        
        # Use the new user-defined parameters for detail generation
        part_synthesis_output = MODELS.part_synthesis_pipeline.get_slat(
            img_black_bg,
            part_synthesis_input['coords'],
            [part_synthesis_input['part_layouts']],
            part_synthesis_input['masks'],
            seed=seed,
            slat_sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg},
            formats=['mesh', 'gaussian'],
            preprocess_image=False,
        )
        
        omni_3d_data = {"synthesis_output": part_synthesis_output}
        cleanup_memory(MODELS.part_synthesis_pipeline)
        return (omni_3d_data,)


class OmniPartSaveMeshNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "omni_3d_data": ("OMNIPART_3D_DATA",),
                "output_prefix": ("STRING", {"default": "omni_untextured"}),
                "simplify_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_file_path",)
    FUNCTION = "save_mesh"
    CATEGORY = "OmniPart"

    def save_mesh(self, omni_3d_data, output_prefix, simplify_ratio):
        synthesis_output = omni_3d_data["synthesis_output"]
        
        output_dir = folder_paths.get_output_directory()
        final_output_dir = os.path.join(output_dir, f"{output_prefix}_{np.random.randint(10000)}")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Save untextured mesh only
        save_parts_outputs(
            synthesis_output,
            output_dir=final_output_dir,
            simplify_ratio=simplify_ratio,
            save_video=False,
            save_glb=True,
            textured=False # Explicitly set to False
        )
        
        # Find all generated .glb files and merge them
        glb_files = [os.path.join(final_output_dir, f) for f in os.listdir(final_output_dir) if f.endswith('.glb')]
        if not glb_files:
            raise Exception("No .glb files were generated to merge.")
            
        # The original code saves parts as part1.glb, part2.glb etc.
        # We need a function to merge these into one final mesh.
        
        # Let's manually merge them here
        all_meshes = [trimesh.load(p) for p in glb_files]
        if not all_meshes:
            raise Exception("Could not load any valid meshes from the output directory.")

        combined_mesh = trimesh.util.concatenate(all_meshes)
        
        final_mesh_path = os.path.join(final_output_dir, "untextured_mesh.glb")
        combined_mesh.export(final_mesh_path)
        
        cleanup_memory()
        return (final_mesh_path,)
