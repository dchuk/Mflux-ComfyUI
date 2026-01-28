"""Z-Image Turbo nodes for ComfyUI.

Provides dedicated loader and sampler nodes for Z-Image Turbo model generation,
following ComfyUI's standard loader/sampler pattern.

Nodes:
- MfluxZImageLoader: Load Z-Image Turbo model with quantization options
- MfluxZImageSampler: Generate images from text prompts
- MfluxZImageImg2Img: Transform images with text-guided generation
"""
import os
import time
import uuid
import numpy as np
import torch
from PIL import Image

import folder_paths
from folder_paths import models_dir

from .utils.tensor_utils import pil_to_comfy_tensor, comfy_tensor_to_pil
from .utils.memory_utils import clear_mlx_memory

# --- Environment Variable Guards ---
_skip_mlx_import = os.environ.get("MFLUX_COMFY_DISABLE_MLX_IMPORT") == "1"
_skip_mflux_import = os.environ.get("MFLUX_COMFY_DISABLE_MFLUX_IMPORT") == "1"

# --- mflux Imports with Guards ---
ZImageTurbo = None
ModelConfig = None

if not _skip_mflux_import:
    try:
        from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
        from mflux.models.common.config import ModelConfig
    except ImportError:
        pass

# --- HuggingFace Hub Import ---
try:
    from huggingface_hub import scan_cache_dir
except ImportError:
    scan_cache_dir = None

# --- Model Cache ---
_model_cache = {}

# --- Mflux Directory ---
mflux_dir = os.path.join(models_dir, "Mflux")
if not os.path.exists(mflux_dir):
    os.makedirs(mflux_dir, exist_ok=True)


def _is_model_directory(path: str) -> bool:
    """Check if a directory appears to be a model root."""
    indicators = ["config.json", "transformer", "vae", "text_encoder", "text_encoder_2", "model_index.json"]
    try:
        items = os.listdir(path)
        return any(item in items for item in indicators)
    except Exception:
        return False


def _get_cached_models() -> list[str]:
    """Get list of cached model repo IDs from HuggingFace cache."""
    if scan_cache_dir is None:
        return []
    try:
        cache_info = scan_cache_dir()
        # Filter for models only, return sorted list of repo_ids
        models = [repo.repo_id for repo in cache_info.repos if repo.repo_type == "model"]
        return sorted(models, key=str.lower)
    except Exception:
        return []


def _scan_zimage_models() -> list[str]:
    """Scan for available Z-Image models in local folder, cache, and provide aliases."""
    options = []

    # 1. Scan local models folder (recursive)
    if os.path.exists(mflux_dir):
        for root, dirs, files in os.walk(mflux_dir):
            if _is_model_directory(root):
                rel_path = os.path.relpath(root, mflux_dir)
                # Filter for Z-Image related models
                if "z-image" in rel_path.lower() or "zimage" in rel_path.lower():
                    options.append(f"📁 {rel_path}")
                dirs[:] = []  # Stop recursing

    # 2. Check system cache for Z-Image models
    cached_models = _get_cached_models()
    for repo in cached_models:
        if "z-image" in repo.lower() or "zimage" in repo.lower():
            options.append(f"🟢 {repo}")

    # 3. Add well-known Z-Image models and aliases
    known_models = [
        "filipstrand/Z-Image-Turbo-mflux-4bit",
    ]
    for model in known_models:
        display = f"☁️ {model}"
        if display not in options:
            options.append(display)

    # 4. Add the official alias
    options.append("☁️ z-image-turbo")

    # Sort and ensure we have at least one option
    options = sorted(set(options), key=str.lower)
    return options if options else ["z-image-turbo"]


def _save_tensor_to_temp(tensor: torch.Tensor, prefix: str = "zimage_temp") -> str:
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


class MfluxZImageLoader:
    """Load Z-Image Turbo model with quantization options.

    Returns a ZIMAGE_MODEL that can be connected to sampler nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_options = _scan_zimage_models()
        return {
            "required": {
                "model": (model_options, {
                    "default": model_options[0] if model_options else "z-image-turbo",
                    "tooltip": "📁 = Local (ComfyUI/models/Mflux)\n🟢 = Cached (HuggingFace System Cache)\n☁️ = Alias/Shortcut (May trigger download)"
                }),
                "quantize": (["4", "8", "None"], {
                    "default": "4",
                    "tooltip": "4-bit: Fastest, lowest memory. 8-bit: Better quality, more memory. None: Full precision, highest quality and memory."
                }),
            }
        }

    RETURN_TYPES = ("ZIMAGE_MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "MFlux/ZImage"
    FUNCTION = "load_model"

    def load_model(self, model: str, quantize: str):
        """Load Z-Image Turbo model with caching."""
        if ZImageTurbo is None or ModelConfig is None:
            raise ImportError(
                "mflux is not installed or failed to load. "
                "Please ensure you have 'mflux>=0.15.5' installed."
            )

        # Clean up visual indicators from model name
        clean_model = model.replace("🟢 ", "").replace("📁 ", "").replace("☁️ ", "")

        # Resolve model path if it's a local folder reference
        model_path = None
        if "📁 " in model:
            full_path = os.path.join(mflux_dir, clean_model)
            if os.path.exists(full_path):
                model_path = full_path

        # Parse quantize value
        q_val = None if quantize == "None" else int(quantize)

        # Cache key
        cache_key = (clean_model, q_val)

        if cache_key not in _model_cache:
            # Clear old cache entries
            _model_cache.clear()

            print(f"[MFlux-ZImage] Loading model: '{clean_model}' (Quantize: {quantize})")

            # Create model config
            model_config = ModelConfig.z_image_turbo()

            # Load model
            instance = ZImageTurbo(
                quantize=q_val,
                model_path=model_path,
                model_config=model_config,
            )

            _model_cache[cache_key] = instance
            print(f"[MFlux-ZImage] Model loaded successfully.")

        return (_model_cache[cache_key],)


class MfluxZImageSampler:
    """Generate images from text prompts using Z-Image Turbo.

    Takes a loaded ZIMAGE_MODEL and generates images based on text prompts.
    Z-Image Turbo is optimized for fast generation with good quality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ZIMAGE_MODEL", {
                    "tooltip": "Z-Image Turbo model from MfluxZImageLoader"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of a beautiful landscape at golden hour",
                    "tooltip": "Text description of the image to generate"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible generation"
                }),
                "steps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Number of inference steps. Z-Image Turbo works well with 4-12 steps."
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Image width in pixels"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Image height in pixels"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "MFlux/ZImage"
    FUNCTION = "generate"

    def generate(self, model, prompt: str, seed: int, steps: int, width: int, height: int):
        """Generate an image from text prompt using Z-Image Turbo."""
        try:
            print(f"[MFlux-ZImage] Generating: seed={seed}, steps={steps}, size={width}x{height}")

            # Generate image
            # Note: Z-Image Turbo uses guidance=0.0 internally, no guidance parameter
            result = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
            )

            # Extract PIL image from result
            if hasattr(result, 'image'):
                pil_image = result.image
            else:
                pil_image = result

            # Convert to ComfyUI tensor
            image_tensor = pil_to_comfy_tensor(pil_image)

            return (image_tensor,)

        finally:
            # Clear MLX memory after generation
            clear_mlx_memory()


class MfluxZImageImg2Img:
    """Transform images with text-guided generation using Z-Image Turbo.

    Takes an input image and generates a new version based on the text prompt,
    with denoise controlling how much the original image influences the result.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ZIMAGE_MODEL", {
                    "tooltip": "Z-Image Turbo model from MfluxZImageLoader"
                }),
                "init_image": ("IMAGE", {
                    "tooltip": "Starting image for transformation. The generated image will be based on this."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic transformation of the image",
                    "tooltip": "Text description guiding the transformation"
                }),
                "denoise": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "0.0 = Keep original image, 1.0 = Ignore original completely. 0.4-0.7 typically works well."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible generation"
                }),
                "steps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Number of inference steps"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "MFlux/ZImage"
    FUNCTION = "generate"

    def generate(self, model, init_image: torch.Tensor, prompt: str, denoise: float, seed: int, steps: int):
        """Generate an image based on input image and text prompt."""
        temp_path = None
        try:
            # Get dimensions from input tensor [B, H, W, C]
            height = init_image.shape[1]
            width = init_image.shape[2]

            # Save input image to temp file
            temp_path = _save_tensor_to_temp(init_image, "zimage_input")

            # Map denoise to image_strength (inverted: higher denoise = lower strength)
            # denoise 0.0 -> image_strength 1.0 (keep image)
            # denoise 1.0 -> image_strength 0.0 (ignore image)
            image_strength = 1.0 - denoise

            print(f"[MFlux-ZImage] Img2Img: seed={seed}, steps={steps}, size={width}x{height}, denoise={denoise}")

            # Generate image with img2img
            result = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                image_path=temp_path,
                image_strength=image_strength,
            )

            # Extract PIL image from result
            if hasattr(result, 'image'):
                pil_image = result.image
            else:
                pil_image = result

            # Convert to ComfyUI tensor
            image_tensor = pil_to_comfy_tensor(pil_image)

            return (image_tensor,)

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Best effort cleanup

            # Clear MLX memory
            clear_mlx_memory()
