"""SeedVR2 diffusion upscaling nodes for ComfyUI.

Provides dedicated loader and upscaler nodes for SeedVR2 super-resolution model,
following ComfyUI's standard loader/sampler pattern.

Nodes:
- MfluxSeedVR2Loader: Load SeedVR2 model with quantization options
- MfluxSeedVR2Upscaler: Upscale images using diffusion-based super-resolution
"""
import os
import time
import uuid
import numpy as np
import torch
from PIL import Image

import folder_paths

from .utils.tensor_utils import pil_to_comfy_tensor
from .utils.memory_utils import clear_mlx_memory

# --- Environment Variable Guards ---
_skip_mlx_import = os.environ.get("MFLUX_COMFY_DISABLE_MLX_IMPORT") == "1"
_skip_mflux_import = os.environ.get("MFLUX_COMFY_DISABLE_MFLUX_IMPORT") == "1"

# --- mflux Imports with Guards ---
SeedVR2 = None
ModelConfig = None
ScaleFactor = None

if not _skip_mflux_import:
    try:
        from mflux.models.seedvr2 import SeedVR2
        from mflux.models.common.config import ModelConfig
        from mflux.utils.scale_factor import ScaleFactor
    except ImportError:
        pass

# --- Model Cache ---
_seedvr2_cache = {}


def _save_tensor_to_temp(tensor: torch.Tensor, prefix: str = "seedvr2_input") -> str:
    """Save a ComfyUI IMAGE tensor to a temporary PNG file.

    Args:
        tensor: ComfyUI IMAGE tensor [B, H, W, C] in [0, 1] float32
        prefix: Filename prefix

    Returns:
        Absolute path to the saved PNG file
    """
    in_dir = folder_paths.get_input_directory()
    fname = f"{prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.png"
    out_path = os.path.join(in_dir, fname)

    # Convert tensor to numpy
    array = tensor.cpu().numpy()

    # Handle batch dimension: [B, H, W, C] -> [H, W, C]
    if array.ndim == 4:
        array = array[0]

    # Convert to uint8
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    # Save as RGB PNG
    img = Image.fromarray(array, mode='RGB')
    img.save(out_path)

    return out_path


def _calculate_dimensions(input_w: int, input_h: int, resolution) -> tuple[int, int]:
    """Calculate output dimensions matching SeedVR2Util.preprocess_image() pattern.

    Args:
        input_w: Input image width
        input_h: Input image height
        resolution: Either a ScaleFactor or int target resolution

    Returns:
        Tuple of (output_width, output_height)
    """
    if ScaleFactor is not None and isinstance(resolution, ScaleFactor):
        # Scale based on shortest edge
        target_res = resolution.get_scaled_value(min(input_w, input_h))
    else:
        target_res = resolution

    # Scale proportionally based on shortest edge
    scale = target_res / min(input_w, input_h)
    true_w = int(input_w * scale)
    true_h = int(input_h * scale)

    # Round to even (multiple of 2)
    true_w = (true_w // 2) * 2
    true_h = (true_h // 2) * 2

    return true_w, true_h


class MfluxSeedVR2Loader:
    """Load SeedVR2 model with quantization options.

    Returns a SEEDVR2_MODEL that can be connected to the upscaler node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quantize": (["4", "8", "None"], {
                    "default": "4",
                    "tooltip": "4-bit: Fastest, lowest memory. 8-bit: Better quality. None: Full precision."
                }),
            },
            "optional": {
                "clear_cache_after_use": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Clear after each",
                    "label_off": "Keep cached",
                    "tooltip": "Clear MLX cache after each upscale. OFF keeps model loaded for faster subsequent runs."
                }),
            }
        }

    RETURN_TYPES = ("SEEDVR2_MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "mflux"
    FUNCTION = "load_model"

    def load_model(self, quantize: str, clear_cache_after_use: bool = True):
        """Load SeedVR2 model with caching."""
        if SeedVR2 is None or ModelConfig is None:
            raise ImportError(
                "mflux is not installed or SeedVR2 not available. "
                "Please ensure you have 'mflux>=0.15.5' installed."
            )

        # Parse quantize value
        q_val = None if quantize == "None" else int(quantize)

        # Cache key
        cache_key = ("seedvr2", q_val)

        if cache_key not in _seedvr2_cache:
            # Clear old cache entries
            _seedvr2_cache.clear()

            print(f"[MFlux-SeedVR2] Loading model (Quantize: {quantize})")

            # Load model
            instance = SeedVR2(
                quantize=q_val,
                model_path=None,
                model_config=ModelConfig.seedvr2_3b(),
            )

            _seedvr2_cache[cache_key] = instance
            print("[MFlux-SeedVR2] Model loaded successfully.")

        # Return model and cache setting as dict wrapped in tuple
        model_data = {
            "model": _seedvr2_cache[cache_key],
            "clear_cache": clear_cache_after_use,
        }
        return (model_data,)


class MfluxSeedVR2Upscaler:
    """Upscale images using SeedVR2 diffusion-based super-resolution.

    Takes a loaded SEEDVR2_MODEL and any ComfyUI IMAGE, returns upscaled IMAGE.
    SeedVR2 is a dedicated super-resolution model using NaDiT transformer architecture.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SEEDVR2_MODEL", {
                    "tooltip": "SeedVR2 model from MfluxSeedVR2Loader"
                }),
                "image": ("IMAGE", {
                    "tooltip": "ComfyUI IMAGE to upscale (from any source)"
                }),
                "scale_mode": (["Multiplier", "Longest Side"], {
                    "default": "Multiplier",
                    "tooltip": "How to specify output size"
                }),
                "multiplier": (["1x", "2x", "4x"], {
                    "default": "4x",
                    "tooltip": "Scale multiplier (when mode is Multiplier)"
                }),
                "longest_side": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 8192,
                    "tooltip": "Target pixels for longest edge (when mode is Longest Side)"
                }),
                "softness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "0.0 = sharpest, 1.0 = maximum softness"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "dimensions",)
    CATEGORY = "mflux"
    FUNCTION = "upscale"

    def upscale(self, model, image: torch.Tensor, scale_mode: str, multiplier: str,
                longest_side: int, softness: float, seed: int):
        """Upscale an image using SeedVR2 diffusion-based super-resolution."""
        if ScaleFactor is None:
            raise ImportError(
                "mflux ScaleFactor not available. "
                "Please ensure you have 'mflux>=0.15.5' installed."
            )

        # Extract model and cache setting from model dict
        model_data = model
        seedvr2_model = model_data["model"]
        should_clear = model_data["clear_cache"]

        temp_path = None
        try:
            # Get input dimensions from tensor [B, H, W, C]
            input_h, input_w = image.shape[1], image.shape[2]

            # Save tensor to temp file (SeedVR2 requires file path)
            temp_path = _save_tensor_to_temp(image)

            # Determine resolution parameter
            if scale_mode == "Multiplier":
                mult_int = int(multiplier.replace("x", ""))
                resolution = ScaleFactor(value=mult_int)
            else:
                resolution = longest_side

            # Calculate output dimensions for display
            output_w, output_h = _calculate_dimensions(input_w, input_h, resolution)
            dims_string = f"{input_w}x{input_h} -> {output_w}x{output_h}"

            print(f"[MFlux-SeedVR2] Upscaling: {dims_string}, softness={softness}")

            # Generate upscaled image
            result = seedvr2_model.generate_image(
                seed=seed,
                image_path=temp_path,
                resolution=resolution,
                softness=softness,
            )

            # Extract PIL image from result
            if hasattr(result, 'image'):
                pil_image = result.image
            else:
                pil_image = result

            # Convert to ComfyUI tensor
            output_tensor = pil_to_comfy_tensor(pil_image)

            return (output_tensor, dims_string)

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Best effort cleanup

            # Clear MLX memory based on user setting
            if should_clear:
                clear_mlx_memory()
