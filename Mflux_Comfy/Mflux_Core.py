import random
import json
import os
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import comfy.utils  # Required for the progress bar
from importlib import import_module

# --- MFLUX Imports with Guards (Fixes CI/Linux/Missing Dependency Regressions) ---
_skip_mlx_import = os.environ.get("MFLUX_COMFY_DISABLE_MLX_IMPORT") == "1"
_skip_mflux_import = os.environ.get("MFLUX_COMFY_DISABLE_MFLUX_IMPORT") == "1"

try:
    if _skip_mlx_import:
        raise ImportError("Skipping MLX import via env var")
    import mlx.core as mx
except ImportError:
    mx = None

try:
    if _skip_mflux_import:
        raise ImportError("Skipping Mflux import via env var")
    from mflux.models.common.config import ModelConfig
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
    from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
    from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
    from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
    from mflux.models.fibo.variants.txt2img.fibo import FIBO
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
    from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
    from mflux.callbacks.instances.memory_saver import MemorySaver

    # Optional: Try to import ControlnetUtil
    try:
        from mflux.controlnet.controlnet_util import ControlnetUtil
    except ImportError:
        ControlnetUtil = None

except ImportError:
    # Dummy classes to allow the node to load in ComfyUI/CI even if dependencies are missing
    Flux1 = None
    Flux1Controlnet = None
    Flux1Fill = None
    Flux1Depth = None
    Flux1Redux = None
    FIBO = None
    QwenImage = None
    ZImageTurbo = None
    ModelConfig = None
    CallbackRegistry = None
    ControlnetUtil = None
    MemorySaver = None

from .Mflux_Pro import MfluxControlNetPipeline

# Cache for loaded models
model_cache = {}

DEFAULT_CONTROLNET_MODELS = [
    "InstantX/FLUX.1-dev-Controlnet-Canny",
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
]

def get_available_controlnet_models() -> list[str]:
    """Return known ControlNet model repo IDs, falling back to defaults."""
    if ModelConfig is None:
        return DEFAULT_CONTROLNET_MODELS.copy()

    try:
        model_cfg = import_module("mflux.config.model_config")
        available = getattr(model_cfg, "AVAILABLE_MODELS", {})
        models = []
        for cfg in available.values():
            control_name = getattr(cfg, "controlnet_model", None)
            if control_name and control_name not in models:
                models.append(control_name)
        return models or DEFAULT_CONTROLNET_MODELS.copy()
    except Exception:
        return DEFAULT_CONTROLNET_MODELS.copy()

class ComfyUIProgressBarCallback:
    """Callback to update ComfyUI progress bar during mflux generation."""
    def __init__(self, total_steps):
        self.pbar = comfy.utils.ProgressBar(total_steps)

    def call_in_loop(self, t, **kwargs):
        self.pbar.update(1)

def _get_mflux_version() -> str:
    try:
        from mflux.utils.version_util import VersionUtil
        return VersionUtil.get_mflux_version()
    except Exception:
        return 'unknown'

def is_third_party_model(model_string: str) -> bool:
    """Check if the model name looks like a third-party HF repo."""
    prefixes = ["filipstrand/", "akx/", "Freepik/", "shuttleai/", "Tongyi-MAI/", "dhairyashil/", "briaai/"]
    return any(str(model_string).startswith(p) for p in prefixes)

def infer_quant_bits(name: str | None) -> int | None:
    if not name:
        return None
    s = str(name).lower()
    for b in (8, 6, 5, 4, 3):
        if f"{b}-bit" in s or f"{b}bit" in s:
            return b
    if "mflux-4bit" in s:
        return 4
    return None

def load_or_create_model(model_string, quantize, model_path, lora_paths, lora_scales, variant="txt2img", controlnet_path=None, base_model_hint="dev"):
    """
    Load or create a model instance (Flux, FIBO, Qwen, Z-Image) based on unified logic.
    `model_string` can be an alias (e.g., "schnell", "qwen") or a path/HF ID.
    `base_model_hint` is used to disambiguate or force a specific model type.
    """
    if not (Flux1 or FIBO or QwenImage or ZImageTurbo):
        raise ImportError("MFlux is not installed or failed to load.")

    effective_model_path = model_path if model_path else None

    if effective_model_path and quantize is None:
        quantize = infer_quant_bits(effective_model_path)

    # Cache key needs to be comprehensive for different model types
    key = (model_string, quantize, effective_model_path, tuple(lora_paths), tuple(lora_scales), variant, controlnet_path, base_model_hint)
    if key not in model_cache:
        model_cache.clear() # Clear cache to save VRAM

        print(f"[MFlux-ComfyUI] Loading model: '{model_string}' (Hint: {base_model_hint}, Variant: {variant})")

        # Determine target class and config_base_model string
        target_class = None
        config_base_model_name = None

        if base_model_hint == "qwen":
            target_class = QwenImage
            config_base_model_name = "qwen-image"
        elif base_model_hint == "fibo":
            target_class = FIBO
            config_base_model_name = "fibo"
        elif base_model_hint == "z-image-turbo":
            target_class = ZImageTurbo
            config_base_model_name = "z-image-turbo"
        else: # Default to Flux or derive from model_string
            if "z-image" in str(model_string).lower() or base_model_hint == "z-image": # Fallback for z-image if hint is generic
                target_class = ZImageTurbo
                config_base_model_name = "z-image-turbo" # Default z-image config
            elif "fibo" in str(model_string).lower():
                target_class = FIBO
                config_base_model_name = "fibo"
            elif "qwen" in str(model_string).lower():
                target_class = QwenImage
                config_base_model_name = "qwen-image"
            else: # Assume Flux
                target_class = Flux1
                config_base_model_name = base_model_hint # e.g. "dev", "schnell"

        # Resolve ModelConfig
        # For specific Flux variants (Fill, Depth, Redux, ControlNet)
        if target_class == Flux1: # Only apply variant configs to Flux
            if variant == "fill":
                model_config = ModelConfig.dev_fill()
            elif variant == "depth":
                model_config = ModelConfig.dev_depth()
            elif variant == "redux":
                model_config = ModelConfig.dev_redux()
            elif variant == "controlnet":
                if controlnet_path == "InstantX/FLUX.1-dev-Controlnet-Canny":
                    model_config = ModelConfig.dev_controlnet_canny()
                elif "upscaler" in str(controlnet_path).lower():
                    model_config = ModelConfig.dev_controlnet_upscaler()
                else:
                    model_config = ModelConfig.from_name(model_string, base_model=config_base_model_name)
            else: # Standard txt2img Flux
                model_config = ModelConfig.from_name(model_string, base_model=config_base_model_name)
        else: # For FIBO, Qwen, Z-Image - use the resolved base_model_name
            model_config = ModelConfig.from_name(model_string, base_model=config_base_model_name)


        common_args = {
            "quantize": quantize,
            "model_path": effective_model_path,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
        }

        instance = target_class(model_config=model_config, **common_args)
        model_cache[key] = instance
        print(f"[MFlux-ComfyUI] Loaded {target_class.__name__} for '{model_string}'.")

    return model_cache[key]

def get_lora_info(loras_pipeline):
    if loras_pipeline and hasattr(loras_pipeline, 'lora_paths') and hasattr(loras_pipeline, 'lora_scales'):
        return loras_pipeline.lora_paths, loras_pipeline.lora_scales
    return [], []

def generate_image(prompt, model_string, seed, width, height, steps, guidance, quantize="None", metadata=True,
                   model_path=None, img2img_pipeline=None, loras_pipeline=None, controlnet_pipeline=None,
                   base_model_hint="dev", negative_prompt="", optimizations=None,
                   masked_image_path=None, depth_image_path=None, redux_image_paths=None,
                   redux_image_strengths=None):

    q_val = None if quantize in (None, "None") else int(quantize)
    lora_paths, lora_scales = get_lora_info(loras_pipeline)

    # Determine variant and ControlNet settings
    variant = "txt2img"
    controlnet_path = None
    controlnet_strength = 1.0
    controlnet_image_path = None
    img2img_image_obj = None # For MfluxImg2ImgPipeline object
    img2img_strength = None

    if img2img_pipeline:
        variant = "img2img" # Consider img2img as a variant
        img2img_image_obj = img2img_pipeline # Pass the whole pipeline object
        img2img_strength = img2img_pipeline.image_strength

    if masked_image_path:
        variant = "fill"
    elif depth_image_path:
        variant = "depth"
    elif redux_image_paths:
        variant = "redux"
    elif controlnet_pipeline is not None and isinstance(controlnet_pipeline, MfluxControlNetPipeline):
        variant = "controlnet"
        controlnet_path = controlnet_pipeline.model_selection
        controlnet_strength = float(controlnet_pipeline.control_strength)
        controlnet_image_path = controlnet_pipeline.control_image_path

    # Load Model
    model_instance = load_or_create_model(
        model_string=model_string,
        quantize=q_val,
        model_path=model_path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        variant=variant,
        controlnet_path=controlnet_path,
        base_model_hint=base_model_hint
    )

    # Prepare Generation Arguments
    seed_val = int(seed) if seed != -1 else random.randint(0, 0xffffffffffffffff)

    gen_kwargs = {
        "seed": seed_val,
        "prompt": prompt,
        "num_inference_steps": steps,
        "height": height,
        "width": width,
    }

    # Model-specific argument routing
    if isinstance(model_instance, QwenImage):
        gen_kwargs["negative_prompt"] = negative_prompt
    elif isinstance(model_instance, ZImageTurbo):
        if "guidance" in gen_kwargs: del gen_kwargs["guidance"] # Z-Image doesn't use guidance
    elif isinstance(model_instance, (Flux1, Flux1Fill, Flux1Depth, Flux1Redux, Flux1Controlnet, FIBO)):
        gen_kwargs["guidance"] = guidance
    else: # Fallback for any other model type
        gen_kwargs["guidance"] = guidance


    # Add variant-specific arguments
    if variant == "fill":
        if img2img_image_obj: gen_kwargs["image_path"] = img2img_image_obj.image_path # For fill, source image
        gen_kwargs["masked_image_path"] = masked_image_path
    elif variant == "depth":
        if img2img_image_obj: gen_kwargs["image_path"] = img2img_image_obj.image_path # For depth, source image
        gen_kwargs["depth_image_path"] = depth_image_path
    elif variant == "redux":
        gen_kwargs["redux_image_paths"] = redux_image_paths
        if redux_image_strengths:
            gen_kwargs["redux_image_strengths"] = redux_image_strengths
    elif variant == "controlnet":
        gen_kwargs["controlnet_image_path"] = controlnet_image_path
        gen_kwargs["controlnet_strength"] = controlnet_strength
    elif variant == "img2img": # Standard img2img
        if img2img_image_obj:
            gen_kwargs["image_path"] = img2img_image_obj.image_path
            gen_kwargs["image_strength"] = img2img_strength
    # else: # txt2img, no extra args beyond common ones

    print(f"[MFlux-ComfyUI] Generating with {model_instance.__class__.__name__} ({variant}) seed: {seed_val}, steps: {steps}")

    # Register Progress Bar Callback
    if CallbackRegistry:
        model_instance.callbacks = CallbackRegistry()
        pbar = ComfyUIProgressBarCallback(total_steps=steps)
        model_instance.callbacks.register(pbar)

    # Apply Optimizations
    low_ram = optimizations.get("low_ram", False) if optimizations else False
    vae_tiling = optimizations.get("vae_tiling", False) if optimizations else False
    vae_tiling_split = optimizations.get("vae_tiling_split", "horizontal") if optimizations else "horizontal"

    if isinstance(model_instance, Flux1) and vae_tiling: # Apply VAE tiling only to Flux
        model_instance.vae.decoder.enable_tiling = True
        model_instance.vae.decoder.split_direction = vae_tiling_split
        print(f"[MFlux-ComfyUI] VAE tiling enabled: {vae_tiling_split}")

    if low_ram and MemorySaver:
        # MemorySaver needs the model instance
        memory_saver = MemorySaver(model=model_instance, keep_transformer=True, cache_limit_bytes=1000**3)
        model_instance.callbacks.register(memory_saver)
        print("[MFlux-ComfyUI] Low RAM optimization (MemorySaver) enabled.")

    # Generate
    try:
        # Pass optimizations to gen_kwargs if the backend supports them directly (though we handle them above)
        # This is a fallback if the backend ever changes to handle them internally.
        # For now, we apply them directly to the model instance.
        # gen_kwargs["low_ram"] = low_ram # Already handled by MemorySaver
        # gen_kwargs["vae_tiling"] = vae_tiling # Already handled by Flux.vae.decoder
        generated_result = model_instance.generate_image(**gen_kwargs)
    except Exception as e:
        print(f"[MFlux-ComfyUI] Error during generation: {e}")
        raise e

    # Process Output
    if hasattr(generated_result, "image"):
        pil_image = generated_result.image
    else:
        pil_image = generated_result # Assume PIL Image if not an object

    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    return (image_tensor,)

def save_images_with_metadata(images, prompt, model_alias, quantize, model_path, seed, height, width, steps, guidance,
                             lora_paths, lora_scales, image_path, image_strength, filename_prefix="Mflux",
                             full_prompt=None, extra_pnginfo=None, base_model_hint=None, negative_prompt_used="",
                             vae_tiling=False, vae_tiling_split="horizontal",
                             control_image_path=None, control_strength=None, control_model=None,
                             quantize_effective=None):

    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
    mflux_output_folder = os.path.join(full_output_folder, "MFlux")
    os.makedirs(mflux_output_folder, exist_ok=True)

    existing_files = os.listdir(mflux_output_folder)
    existing_counters = [
        int(f.split("_")[-1].split(".")[0])
        for f in existing_files
        if f.startswith(filename_prefix) and f.endswith(".png") and "_" in f
    ]
    counter = max(existing_counters, default=0) + 1

    results = list()
    for image_tensor in images: # Renamed from 'image' to 'image_tensor' for clarity
        i = 255. * image_tensor.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        metadata = PngInfo()
        if full_prompt:
            metadata.add_text("full_prompt", json.dumps(full_prompt))
        if extra_pnginfo:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        image_file = f"{filename_prefix}_{counter:05}.png"
        img.save(os.path.join(mflux_output_folder, image_file), pnginfo=metadata, compress_level=4)

        results.append({
            "filename": image_file,
            "subfolder": "MFlux",
            "type": "output"
        })

        metadata_jsonfile = os.path.join(mflux_output_folder, f"{filename_prefix}_{counter:05}.json")
        json_dict = {
            "prompt": prompt,
            "model_alias": model_alias, # Store the alias/path string used for generation
            "base_model_hint": base_model_hint, # Store the base model hint (e.g., "dev", "qwen")
            "quantize": quantize,
            "quantize_effective": quantize_effective,
            "seed": seed,
            "height": height,
            "width": width,
            "steps": steps,
            "guidance": guidance,
            "model_path": model_path, # Store the resolved model path
            "image_path": image_path, # For img2img or fill
            "image_strength": image_strength,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
            "control_image_path": control_image_path,
            "control_strength": control_strength,
            "control_model": control_model,
            "masked_image_path": extra_pnginfo.get("masked_image_path") if extra_pnginfo else None,
            "depth_image_path": extra_pnginfo.get("depth_image_path") if extra_pnginfo else None,
            "redux_image_paths": extra_pnginfo.get("redux_image_paths") if extra_pnginfo else None,
            "mflux_version": _get_mflux_version(),
            "negative_prompt_used": negative_prompt_used, # Store if negative prompt was used
            "vae_tiling": vae_tiling, # Store vae tiling settings
            "vae_tiling_split": vae_tiling_split,
        }
        with open(metadata_jsonfile, 'w') as metadata_file:
            json.dump(json_dict, metadata_file, indent=4)

        counter += 1

    return {"ui": {"images": results}, "counter": counter}
