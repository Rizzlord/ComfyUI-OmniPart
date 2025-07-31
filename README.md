ComfyUI-OmniPart: 2D Image to 3D Model Workflow
This repository contains a set of custom nodes for ComfyUI that implement the OmniPart model, allowing you to convert a single 2D image into a fully segmented, editable 3D model.

The workflow is designed to be interactive and modular, giving you fine-grained control over the segmentation, combination, and generation process. It's an ideal tool for concept artists, game developers, and anyone looking to quickly turn 2D ideas into 3D assets.

(A screenshot of a complete node workflow would be great here!)

Features
Automatic Image Segmentation: Uses the Segment Anything Model (SAM) to automatically break down an input image into numbered parts.

Interactive Part Management: Easily combine, hide, or remove segments with simple text inputs before generating the final model.

3D Model Generation: Creates both untextured and fully textured 3D models in the .glb format.

2D Asset Exporting: Includes a powerful node to isolate, upscale, and export each segmented object as a separate 2D image, perfect for creating sprite sheets or concept boards.

VRAM Management: Includes controls for texture rendering quality to prevent out-of-memory errors on consumer-grade GPUs.

Built-in Upscaling: Supports using any ComfyUI-compatible upscale model to enhance the quality of 2D image outputs.

Installation
Navigate to your ComfyUI installation directory.

Go into the ComfyUI/custom_nodes/ folder.

Clone this repository:

git clone [<your_repository_url_here>](https://github.com/Rizzlord/ComfyUI-OmniPart)

Start or restart ComfyUI. The first time you run a workflow, the necessary models (SAM, OmniPart, etc.) will be downloaded automatically from Hugging Face. This may take some time.

Workflow & Node Guide
The nodes are designed to be used in a specific sequence. Here is a breakdown of a typical workflow from start to finish.

1. Load OmniPart Models
This is the first and most important node. It handles the loading of all the necessary AI models into memory.

Inputs:

force_reload: A boolean to force the node to reload all models from disk.

Purpose: You must connect the output of this node to the first node in your processing chain (Segment Image) to ensure the models are ready.

2. Segment Image
This node takes your input image and performs the initial automatic segmentation.

Inputs:

image: The 2D image you want to convert.

size_threshold: Filters out small, noisy segments. A higher value results in fewer, larger parts. A lower value keeps more fine details.

output_resolution: A dropdown to select the final resolution of the preview image.

enable_upscale / upscale_model: Optionally upscale the preview image.

Outputs:

IMAGE: A preview image showing the colored, numbered segments.

OMNIPART_STATE: The core data (including segmentation masks) that gets passed to the next node.

3. Combine Segments
Use this node to merge multiple small segments into logical, larger parts.

Inputs:

omni_state: The state from the previous node.

merge_string: A text field where you define which segments to combine.

Format: Use commas to group segments and semicolons to separate groups.

Example: 1,2; 3,4,5 will combine segments 1 and 2 into a new part, and segments 3, 4, and 5 into another new part.

Outputs:

IMAGE: An updated preview showing the new, combined segments.

OMNIPART_STATE: The updated state with the merged segments.

4. Hide Segments
This node allows you to remove unwanted parts from the image before 3D generation.

Inputs:

omni_state: The state from the previous node.

hide_string: A comma-separated list of segment numbers to remove (e.g., 2,5).

add_placeholders / placeholder_radius: Adds small circles to the corners of the output image. Enable this if your final 3D model appears incorrectly scaled or off-center.

Outputs:

mask_visualization: The standard colored segment preview.

image_with_hidden_parts: The original image with the hidden parts removed (filled with white).

OMNIPART_STATE: The updated state.

5. Export Objects from Image
A powerful utility node for creating 2D assets. It takes the final segmented image and exports each remaining part as a separate file.

Inputs:

omni_state: The state from the Hide Segments node.

upscale_model: (Required) An upscale model to enhance the detail of each object.

padding: Adds a small margin around each object to prevent it from looking cut off.

canvas_background: Choose whether the final exported images have a white, black, or transparent (alpha) background.

output_resolution: The final square resolution for each exported image.

Outputs:

IMAGE_BATCH: A batch of all the exported object images for previewing in ComfyUI.

object_count: The number of images successfully exported.

6. Generate 3D Data
This is the core node that processes the final segmentation and generates the 3D representation.

Inputs:

omni_state: The final state after all your segment editing.

seed, voxel_steps, slat_steps, etc.: Parameters that control the quality and detail of the 3D generation. Higher values can lead to better results but take more time.

Outputs:

OMNIPART_3D_DATA: The generated 3D data to be passed to a save node.

7. Save Nodes
Save Untextured Mesh: Saves the result as a simple, uncolored .glb 3D file.

Save Textured Mesh: Saves the result as a fully textured .glb file based on the original image colors.

Important: This node is very VRAM-intensive. If you get an "out of memory" error, lower the render_resolution (e.g., to 512) and render_views (e.g., to 50) to reduce the workload.

Troubleshooting
CUDA Out of Memory Error: This almost always happens on the Save Textured Mesh node. The texture baking process is extremely demanding. Solution: Lower the render_resolution and render_views values on the node.

Models Not Downloading: The nodes require an internet connection to download models from Hugging Face on the first run. If it fails, check your connection and try again.

Incorrect 3D Model Scaling: If your final 3D model is small, off-center, or strangely scaled, go back to the Hide Segments node and enable the add_placeholders option. This ensures the generator sees the full canvas boundaries.

Credits
This node pack is an implementation of the official OmniPart research. All credit for the underlying models and 3D generation technology goes to the original authors.

OmniPart Project Page: https://omnipart.github.io/

Original Paper: https://arxiv.org/abs/2404.18023
