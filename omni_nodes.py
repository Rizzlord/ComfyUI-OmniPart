import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter
import gc
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import folder_paths
import comfy.model_management as mm
import cv2
import trimesh

# Add the custom node's directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now we can import from the OmniPart modules
from modules.label_2d_mask.label_parts import (
    prepare_image, get_sam_mask, clean_segment_edges,
    resize_and_pad_to_square
)
from modules.label_2d_mask.visualizer import Visualizer
from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
from modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline
from modules.part_synthesis.process_utils import save_parts_outputs
from modules.inference_utils import (
    load_img_mask, prepare_bbox_gen_input, prepare_part_synthesis_input
)
from segment_anything import SamAutomaticMaskGenerator, build_sam
from transformers import AutoModelForImageSegmentation


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
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    image_np = tensor.mul(255).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(image_np)

def pil_to_tensor(image):
    image_np = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor

def cleanup_memory(*models_to_unload):
    for model in models_to_unload:
        if model is not None:
            if hasattr(model, 'predictor') and hasattr(model.predictor, 'model'):
                 model.predictor.model.to('cpu')
            elif hasattr(model, 'to'):
                 model.to('cpu')
    gc.collect()
    if torch.cuda.is_available():
        mm.soft_empty_cache()

def upscale_pil_image(pil_image, upscale_model):
    if not pil_image or not callable(upscale_model):
        return pil_image
    print("Upscaling image with provided model...")
    image_rgb_pil = pil_image.convert("RGB")
    image_tensor_bhwc = pil_to_tensor(image_rgb_pil)
    image_tensor_bchw = image_tensor_bhwc.permute(0, 3, 1, 2)
    try:
        upscaled_tensor_bchw = upscale_model(image_tensor_bchw)
        upscaled_tensor_bhwc = upscaled_tensor_bchw.permute(0, 2, 3, 1)
        upscaled_pil = tensor_to_pil(upscaled_tensor_bhwc)
        print(f"Upscaled to {upscaled_pil.width}x{upscaled_pil.height}")
        return upscaled_pil
    except Exception as e:
        print(f"ERROR: Upscale failed: {e}")
        return pil_image

def visualize_mask(image_np, group_ids, visualize=True):
    if not visualize:
        max_id = np.max(group_ids)
        if max_id < 0:
            mask_img_uint8 = np.zeros(group_ids.shape, dtype=np.uint8)
        else:
            mask_img = (group_ids + 1) * (255.0 / (max_id + 1))
            mask_img[group_ids == -1] = 0
            mask_img_uint8 = mask_img.astype(np.uint8)
        return Image.fromarray(mask_img_uint8).convert("RGB")
    vis_np = np.ones_like(image_np) * 255
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    for i, unique_id in enumerate(unique_ids):
        color = np.array([(i * 50 + 80) % 256, (i * 120 + 40) % 256, (i * 180 + 20) % 256])
        mask = (group_ids == unique_id)
        vis_np[mask] = color
    for i, unique_id in enumerate(unique_ids):
        mask = (group_ids == unique_id)
        if not np.any(mask):
            continue
        dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        _, _, _, max_loc = cv2.minMaxLoc(dist_map)
        center_x, center_y = max_loc
        text = str(unique_id)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        text_size, _ = cv2.getTextSize(text, font_face, font_scale, 2)
        text_origin = (center_x - text_size[0] // 2, center_y + text_size[1] // 2)
        cv2.putText(vis_np, text, text_origin, font_face, font_scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(vis_np, text, text_origin, font_face, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(vis_np)

def resize_and_fit(pil_image, max_size):
    if pil_image.width == 0 or pil_image.height == 0:
        return pil_image
    ratio = min(max_size / pil_image.width, max_size / pil_image.height)
    new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
    return pil_image.resize(new_size, Image.Resampling.LANCZOS)

def draw_gradient_circle(image_array, center_x, center_y, radius, color_center, color_edge):
    x_min, x_max = max(0, center_x - radius), min(image_array.shape[1], center_x + radius)
    y_min, y_max = max(0, center_y - radius), min(image_array.shape[0], center_y + radius)
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= radius:
                interp_factor = dist / radius if radius > 0 else 0
                new_color = [
                    int(color_center[i] * (1 - interp_factor) + color_edge[i] * interp_factor)
                    for i in range(3)
                ]
                image_array[y, x] = [new_color[0], new_color[1], new_color[2], 255]
    return image_array

def create_texture_bleed_image(image_pil, radius=5):
    if radius <= 0:
        return image_pil
    image_np = np.array(image_pil)
    image_bgra = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGRA)
    alpha_channel = image_bgra[:, :, 3]
    transparent_mask = (alpha_channel == 0).astype(np.uint8)
    image_bgr = image_bgra[:, :, :3]
    inpainted_bgr = cv2.inpaint(image_bgr, transparent_mask, radius, cv2.INPAINT_TELEA)
    final_bgra = np.dstack((inpainted_bgr, alpha_channel))
    final_rgba = cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(final_rgba)

def smooth_edges(pil_image, radius):
    if radius <= 0 or pil_image.mode != 'RGBA':
        return pil_image
    alpha = pil_image.split()[3]
    blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=radius))
    new_image = pil_image.copy()
    new_image.putalpha(blurred_alpha)
    return new_image

# --- Node Definitions ---

class OmniPartLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
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

class OmniPartSegmentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loader": ("BOOLEAN", {"forceInput": True}),
                "image": ("IMAGE",),
                "size_threshold": ("INT", {"default": 1800, "min": 100, "max": 8000, "step": 100}),
                "texture_bleed_radius": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "visualize": ("BOOLEAN", {"default": True}),
                "output_resolution": (['original', '512', '1024', '2048', '4096'],),
            },
            "optional": {
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "upscale_model": ("UPSCALE_MODEL",),
                "edge_smoothing": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("IMAGE", "OMNIPART_STATE")
    FUNCTION = "segment_image"
    CATEGORY = "OmniPart"

    def segment_image(self, loader, image, size_threshold, texture_bleed_radius, visualize, output_resolution, enable_upscale=False, upscale_model=None, edge_smoothing=1.0):
        if not all([MODELS.sam_mask_generator, MODELS.rmbg_model]):
            raise Exception("Models not loaded. Please use the OmniPartLoader node first.")
        MODELS.sam_mask_generator.predictor.model.to(DEVICE)
        MODELS.rmbg_model.to(DEVICE)
        pil_image = tensor_to_pil(image)
        output_dir = folder_paths.get_temp_directory()
        img_name = f"omni_input_{np.random.randint(100000)}"
        processed_image = prepare_image(pil_image, rmbg_net=MODELS.rmbg_model)
        processed_image = resize_and_pad_to_square(processed_image)
        
        processed_image = smooth_edges(processed_image, edge_smoothing)
        
        texture_ready_image = create_texture_bleed_image(processed_image, texture_bleed_radius)
        
        white_bg = Image.new("RGBA", texture_ready_image.size, (255, 255, 255, 255))
        white_bg_img = Image.alpha_composite(white_bg, texture_ready_image.convert("RGBA"))
        image_np = np.array(white_bg_img.convert('RGB'))
        rgba_path = os.path.join(output_dir, f"{img_name}_processed.png")
        texture_ready_image.save(rgba_path)
        visual = Visualizer(image_np)
        group_ids, _ = get_sam_mask(image_np, MODELS.sam_mask_generator, visual, None, rgba_image=texture_ready_image, img_name=img_name, save_dir=output_dir, size_threshold=size_threshold)
        seg_img_pil = visualize_mask(image_np, group_ids, visualize)
        if enable_upscale:
            seg_img_pil = upscale_pil_image(seg_img_pil, upscale_model)
        resolution_int = 0 if output_resolution == 'original' else int(output_resolution)
        if resolution_int > 0:
             seg_img_pil = seg_img_pil.resize((resolution_int, resolution_int), Image.Resampling.LANCZOS)
        omni_state = {
            "image_np": image_np, "processed_image_path": rgba_path, "group_ids": group_ids,
            "original_group_ids": np.copy(group_ids), "img_name": img_name, "output_dir": output_dir,
            "size_threshold": size_threshold
        }
        cleanup_memory(MODELS.sam_mask_generator, MODELS.rmbg_model)
        return (pil_to_tensor(seg_img_pil), omni_state)

class OmniPartCombineSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omni_state": ("OMNIPART_STATE",),
                "merge_string": ("STRING", {"multiline": True, "default": "0,1;3,4", "placeholder": "Combine parts, e.g., 0,1; 3,4,5"}),
                "visualize": ("BOOLEAN", {"default": True}),
                "output_resolution": (['original', '512', '1024', '2048', '4096'],),
            },
            "optional": {
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "upscale_model": ("UPSCALE_MODEL",),
            }
        }
    RETURN_TYPES = ("IMAGE", "OMNIPART_STATE")
    FUNCTION = "combine_segments"
    CATEGORY = "OmniPart"

    def combine_segments(self, omni_state, merge_string, visualize, output_resolution, enable_upscale=False, upscale_model=None):
        if not MODELS.sam_mask_generator:
            raise Exception("Models not loaded. Please use the OmniPartLoader node first.")
        MODELS.sam_mask_generator.predictor.model.to(DEVICE)
        image_np = omni_state["image_np"]
        group_ids = omni_state["original_group_ids"]
        img_name = omni_state["img_name"]
        output_dir = omni_state["output_dir"]
        size_threshold = omni_state["size_threshold"]
        processed_image = Image.open(omni_state["processed_image_path"])
        merge_groups = []
        if merge_string:
            for group_set in merge_string.split(';'):
                try:
                    ids = [int(x.strip()) for x in group_set.split(',') if x.strip()]
                    if ids: merge_groups.append(ids)
                except ValueError:
                    print(f"Warning: Could not parse merge group: '{group_set}'")
        visual = Visualizer(image_np)
        new_group_ids, _ = get_sam_mask(image_np, MODELS.sam_mask_generator, visual, merge_groups=merge_groups, existing_group_ids=group_ids, rgba_image=processed_image, skip_split=True, img_name=img_name, save_dir=output_dir, size_threshold=size_threshold)
        new_group_ids = clean_segment_edges(new_group_ids)
        merged_img_pil = visualize_mask(image_np, new_group_ids, visualize)
        if enable_upscale:
            merged_img_pil = upscale_pil_image(merged_img_pil, upscale_model)
        resolution_int = 0 if output_resolution == 'original' else int(output_resolution)
        if resolution_int > 0:
            merged_img_pil = merged_img_pil.resize((resolution_int, resolution_int), Image.Resampling.LANCZOS)
        omni_state["group_ids"] = new_group_ids
        cleanup_memory(MODELS.sam_mask_generator)
        return (pil_to_tensor(merged_img_pil), omni_state)

class OmniPartHideSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omni_state": ("OMNIPART_STATE",),
                "hide_string": ("STRING", {"multiline": False, "default": "", "placeholder": "Segments to hide, e.g., 1,2"}),
                "visualize": ("BOOLEAN", {"default": True}),
                "output_resolution": (['original', '512', '1024', '2048', '4096'],),
            },
            "optional": {
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "upscale_model": ("UPSCALE_MODEL",),
                "add_placeholders": ("BOOLEAN", {"default": False}),
                "placeholder_radius": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "OMNIPART_STATE")
    RETURN_NAMES = ("mask_visualization", "image_with_hidden_parts", "omni_state")
    FUNCTION = "hide_segments"
    CATEGORY = "OmniPart"

    def hide_segments(self, omni_state, hide_string, visualize, output_resolution, enable_upscale=False, upscale_model=None, add_placeholders=False, placeholder_radius=10):
        image_np = omni_state["image_np"]
        group_ids = np.copy(omni_state["group_ids"])
        hide_ids = []
        if hide_string:
            try:
                hide_ids = [int(x.strip()) for x in hide_string.split(',') if x.strip()]
            except ValueError:
                print(f"Warning: Could not parse hide IDs: '{hide_string}'")
        if hide_ids:
            for segment_id in hide_ids:
                group_ids[group_ids == segment_id] = -1
        mask_visualization_pil = visualize_mask(image_np, group_ids, visualize)
        source_rgba_pil = Image.open(omni_state["processed_image_path"]).convert("RGBA")
        source_rgba_np = np.array(source_rgba_pil)
        pixels_to_hide = (group_ids == -1)
        source_rgba_np[pixels_to_hide] = [255, 255, 255, 255]
        
        if add_placeholders and placeholder_radius > 0:
            color_purple = [128, 0, 128]
            color_black = [0, 0, 0]
            h, w, _ = source_rgba_np.shape
            corners = [
                (placeholder_radius, placeholder_radius),
                (w - 1 - placeholder_radius, placeholder_radius),
                (placeholder_radius, h - 1 - placeholder_radius),
                (w - 1 - placeholder_radius, h - 1 - placeholder_radius),
            ]
            for cx, cy in corners:
                source_rgba_np = draw_gradient_circle(source_rgba_np, cx, cy, placeholder_radius, color_purple, color_black)
        
        image_with_hidden_parts_pil = Image.fromarray(source_rgba_np)

        if enable_upscale:
            mask_visualization_pil = upscale_pil_image(mask_visualization_pil, upscale_model)
            image_with_hidden_parts_pil = upscale_pil_image(image_with_hidden_parts_pil, upscale_model)
        resolution_int = 0 if output_resolution == 'original' else int(output_resolution)
        if resolution_int > 0:
            mask_visualization_pil = mask_visualization_pil.resize((resolution_int, resolution_int), Image.Resampling.LANCZOS)
            image_with_hidden_parts_pil = image_with_hidden_parts_pil.resize((resolution_int, resolution_int), Image.Resampling.LANCZOS)
        omni_state["group_ids"] = group_ids
        return (pil_to_tensor(mask_visualization_pil), pil_to_tensor(image_with_hidden_parts_pil), omni_state)

class OmniPartExportObjects:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omni_state": ("OMNIPART_STATE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "output_directory": ("STRING", {"default": "OmniPart_Objects"}),
                "filename_prefix": ("STRING", {"default": "object"}),
                "canvas_background": (['white', 'black', 'alpha'],),
                "output_resolution": (['512', '1024', '2048', '4096'],),
            },
            "optional": {
                "padding": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "edge_smoothing": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("IMAGE_BATCH", "object_count", "output_path")
    FUNCTION = "export_objects"
    CATEGORY = "OmniPart"

    def export_objects(self, omni_state, upscale_model, output_directory, filename_prefix, canvas_background, output_resolution, padding=5, edge_smoothing=1.0):
        group_ids = omni_state["group_ids"]
        source_image_path = omni_state["processed_image_path"]
        full_output_path = os.path.join(folder_paths.get_output_directory(), output_directory)
        os.makedirs(full_output_path, exist_ok=True)
        source_rgba_pil = Image.open(source_image_path).convert("RGBA")
        source_rgba_np = np.array(source_rgba_pil)
        object_ids = [uid for uid in np.unique(group_ids) if uid >= 0]
        image_batch_list = []
        count = 0
        for obj_id in object_ids:
            mask = (group_ids == obj_id)
            object_canvas_np = np.zeros_like(source_rgba_np)
            object_canvas_np[mask] = source_rgba_np[mask]
            isolated_object_pil = Image.fromarray(object_canvas_np)
            bbox = isolated_object_pil.getbbox()
            if not bbox:
                continue
            img_width, img_height = isolated_object_pil.size
            left = max(0, bbox[0] - padding)
            upper = max(0, bbox[1] - padding)
            right = min(img_width, bbox[2] + padding)
            lower = min(img_height, bbox[3] + padding)
            cropped_object_pil = isolated_object_pil.crop((left, upper, right, lower))
            
            smoothed_object_pil = smooth_edges(cropped_object_pil, edge_smoothing)

            upscaled_object_pil = smoothed_object_pil
            if callable(upscale_model):
                rgb_part = smoothed_object_pil.convert("RGB")
                alpha_part = smoothed_object_pil.split()[3]
                upscaled_rgb_pil = upscale_pil_image(rgb_part, upscale_model)
                upscaled_alpha_pil = alpha_part.resize(upscaled_rgb_pil.size, Image.Resampling.LANCZOS)
                upscaled_rgb_pil.putalpha(upscaled_alpha_pil)
                upscaled_object_pil = upscaled_rgb_pil
            
            fitted_object_pil = resize_and_fit(upscaled_object_pil, 512)
            if canvas_background == 'white':
                bg_color = (255, 255, 255, 255)
            elif canvas_background == 'black':
                bg_color = (0, 0, 0, 255)
            else:
                bg_color = (0, 0, 0, 0)
            base_canvas_pil = Image.new('RGBA', (512, 512), bg_color)
            paste_x = (512 - fitted_object_pil.width) // 2
            paste_y = (512 - fitted_object_pil.height) // 2
            base_canvas_pil.paste(fitted_object_pil, (paste_x, paste_y), fitted_object_pil)
            final_image_pil = base_canvas_pil
            target_res_int = int(output_resolution)
            if final_image_pil.size != (target_res_int, target_res_int):
                final_image_pil = final_image_pil.resize((target_res_int, target_res_int), Image.Resampling.LANCZOS)
            save_path = os.path.join(full_output_path, f"{filename_prefix}_{obj_id}.png")
            final_image_pil.save(save_path)
            image_batch_list.append(pil_to_tensor(final_image_pil.convert("RGB")))
            count += 1
        if not image_batch_list:
            print("Export Objects: No objects found to export.")
            fallback_tensor = torch.zeros((1, 64, 64, 3))
            return (fallback_tensor, 0, full_output_path)
        final_batch = torch.cat(image_batch_list, dim=0)
        print(f"Exported {count} objects to {full_output_path}")
        return (final_batch, count, full_output_path)

class OmniPartGenerate3DDataNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omni_state": ("OMNIPART_STATE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "voxel_steps": ("INT", {"default": 25, "min": 10, "max": 100, "step": 1}),
                "voxel_cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 15.0, "step": 0.5}),
                "slat_steps": ("INT", {"default": 25, "min": 10, "max": 100, "step": 1}),
                "slat_cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 15.0, "step": 0.5}),
            }
        }
    RETURN_TYPES = ("OMNIPART_3D_DATA",)
    FUNCTION = "generate_3d_data"
    CATEGORY = "OmniPart"

    def generate_3d_data(self, omni_state, seed, voxel_steps, voxel_cfg, slat_steps, slat_cfg):
        if not all([MODELS.bbox_gen_model, MODELS.part_synthesis_pipeline]):
            raise Exception("Models not loaded. Please use the OmniPartLoader node first.")
        output_dir = omni_state["output_dir"]
        img_name = omni_state["img_name"]
        final_group_ids = omni_state["group_ids"]
        save_mask = (final_group_ids + 1).reshape(final_group_ids.shape[0], final_group_ids.shape[1], 1).repeat(3, axis=-1)
        mask_save_path = os.path.join(output_dir, f"{img_name}_mask.tiff")
        cv2.imwrite(mask_save_path, save_mask.astype(np.float32))
        omni_state["mask_path"] = mask_save_path
        MODELS.bbox_gen_model.to(DEVICE)
        MODELS.part_synthesis_pipeline.to(DEVICE)
        img_path, mask_path = omni_state["processed_image_path"], omni_state["mask_path"]
        img_white_bg, img_black_bg, ordered_mask_input, _ = load_img_mask(img_path, mask_path)
        voxel_coords = MODELS.part_synthesis_pipeline.get_coords(
            img_black_bg, num_samples=1, seed=seed, 
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
        part_synthesis_output = MODELS.part_synthesis_pipeline.get_slat(
            img_black_bg, part_synthesis_input['coords'], [part_synthesis_input['part_layouts']],
            part_synthesis_input['masks'], seed=seed,
            slat_sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg},
            formats=['mesh', 'gaussian'], preprocess_image=False,
        )
        omni_3d_data = {"synthesis_output": part_synthesis_output}
        cleanup_memory(MODELS.part_synthesis_pipeline)
        return (omni_3d_data,)

class OmniPartSaveMeshNode:
    @classmethod
    def INPUT_TYPES(cls):
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
        save_parts_outputs(
            synthesis_output, output_dir=final_output_dir,
            simplify_ratio=simplify_ratio, save_video=False, save_glb=True, textured=False
        )
        glb_files = [os.path.join(final_output_dir, f) for f in os.listdir(final_output_dir) if f.endswith('.glb')]
        if not glb_files: raise Exception("No .glb files were generated to merge.")
        all_meshes = [trimesh.load(p) for p in glb_files]
        if not all_meshes: raise Exception("Could not load any valid meshes from the output directory.")
        combined_mesh = trimesh.util.concatenate(all_meshes)
        final_mesh_path = os.path.join(final_output_dir, "untextured_mesh.glb")
        combined_mesh.export(final_mesh_path)
        cleanup_memory()
        return (final_mesh_path,)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "OmniPartLoader": OmniPartLoaderNode,
    "OmniPartSegmentImage": OmniPartSegmentImage,
    "OmniPartCombineSegments": OmniPartCombineSegments,
    "OmniPartHideSegments": OmniPartHideSegments,
    "OmniPartExportObjects": OmniPartExportObjects,
    "OmniPartGenerate3DData": OmniPartGenerate3DDataNode,
    "OmniPartSaveMesh": OmniPartSaveMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniPartLoader": "Load OmniPart Models",
    "OmniPartSegmentImage": "Segment Image",
    "OmniPartCombineSegments": "Combine Segments",
    "OmniPartHideSegments": "Hide Segments",
    "OmniPartExportObjects": "Export Objects from Image",
    "OmniPartGenerate3DData": "Generate 3D Data",
    "OmniPartSaveMesh": "Save Untextured Mesh",
}