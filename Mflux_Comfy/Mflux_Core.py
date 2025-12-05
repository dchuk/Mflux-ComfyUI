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
    from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo

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
    ZImageTurbo = None
    ModelConfig = None
    CallbackRegistry = None
    ControlnetUtil = None

from .Mflux_Pro import MfluxControlNetPipeline

# Cache for loaded models
flux_cache = {}

DEFAULT_CONTROLNET_MODELS = [
    "InstantX/FLUX.1-dev-Controlnet-Canny",
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
]

def get_available_controlnet_models() -> list[str]:
    """Return known ControlNet model repo IDs, falling back to defaults."""
    if ModelConfig is None:
        return DEFAULT_CONTROLNET_MODELS.copy()

    try:
        # Attempt to discover models from mflux config if available
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
        # Update ComfyUI progress bar by 1 step
        self.pbar.update(1)

def _get_mflux_version() -> str:
    try:
        from mflux.utils.version_util import VersionUtil
        return VersionUtil.get_mflux_version()
    except Exception:
        return 'unknown'

def is_third_party_model(model_name: str) -> bool:
    """Check if the model name looks like a third-party HF repo."""
    prefixes = ["filipstrand/", "akx/", "Freepik/", "shuttleai/", "Tongyi-MAI/"]
    return any(str(model_name).startswith(p) for p in prefixes)

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

def load_or_create_flux(model_name, quantize, path, lora_paths, lora_scales, variant="txt2img", controlnet_path=None, base_model="dev"):
    """
    Create or fetch a cached Flux model variant.
    """
    if Flux1 is None:
        raise ImportError("MFlux is not installed or failed to load.")

    effective_model_path = path if path else None

    if effective_model_path and quantize is None:
        quantize = infer_quant_bits(effective_model_path)

    # Cache key includes variant and controlnet_path
    key = (model_name, quantize, effective_model_path, tuple(lora_paths), tuple(lora_scales), variant, controlnet_path, base_model)

    if key not in flux_cache:
        flux_cache.clear() # Clear cache to save VRAM

        print(f"[MFlux-ComfyUI] Loading model: {model_name} (Base: {base_model}, Variant: {variant})")

        # Resolve ModelConfig
        # For Redux/Fill/Depth, we need specific configs
        if variant == "fill":
            model_config = ModelConfig.dev_fill() # Fill usually implies dev-fill
        elif variant == "depth":
            model_config = ModelConfig.dev_depth()
        elif variant == "redux":
            model_config = ModelConfig.dev_redux()
        elif variant == "controlnet":
            # If using standard canny, use the preset, otherwise generic dev
            if controlnet_path == "InstantX/FLUX.1-dev-Controlnet-Canny":
                model_config = ModelConfig.dev_controlnet_canny()
            elif "upscaler" in str(controlnet_path).lower():
                model_config = ModelConfig.dev_controlnet_upscaler()
            else:
                model_config = ModelConfig.from_name(model_name, base_model=base_model)
        else:
            model_config = ModelConfig.from_name(model_name, base_model=base_model)

        common_args = {
            "quantize": quantize,
            "model_path": effective_model_path,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
        }

        # Instantiate the correct class
        if "z-image" in str(model_name).lower() or "z-image" in str(base_model).lower():
             # Z-Image Turbo handling
             flux = ZImageTurbo(model_config=model_config, **common_args)
        elif variant == "fill":
            flux = Flux1Fill(model_config=model_config, **common_args)
        elif variant == "depth":
            flux = Flux1Depth(model_config=model_config, **common_args)
        elif variant == "redux":
            flux = Flux1Redux(model_config=model_config, **common_args)
        elif variant == "controlnet":
            # ControlNet needs the specific controlnet path if it's not the default in config
            # But mflux 0.13.1 Flux1Controlnet constructor doesn't take controlnet_path arg directly
            # if it's defined in model_config. However, for custom ones, we might need to handle it.
            # For now, we assume the ModelConfig handles the mapping or we use the default.
            flux = Flux1Controlnet(model_config=model_config, **common_args)
        else:
            flux = Flux1(model_config=model_config, **common_args)

        flux_cache[key] = flux

    return flux_cache[key]

def get_lora_info(Loras):
    if Loras:
        return Loras.lora_paths, Loras.lora_scales
    return [], []

def generate_image(prompt, model, seed, width, height, steps, guidance, quantize="None", metadata=True, Local_model="", image=None, Loras=None, ControlNet=None, base_model="dev", low_ram=False, vae_tiling=False, vae_tiling_split="horizontal", masked_image_path=None, depth_image_path=None, redux_image_paths=None, redux_image_strengths=None):

    # 1. Resolve Model Name
    model_resolved = model
    if Local_model:
        if "schnell" in str(Local_model).lower():
            model_resolved = "schnell"
        elif "dev" in str(Local_model).lower():
            model_resolved = "dev"
        elif "z-image" in str(Local_model).lower():
            model_resolved = "z-image-turbo"

    # If user selected the specific 4-bit repo in the dropdown, ensure we treat it as z-image
    if "z-image" in str(model).lower():
        model_resolved = model

    q_val = None if quantize in (None, "None") else int(quantize)
    lora_paths, lora_scales = get_lora_info(Loras)

    # 2. Determine Variant and ControlNet settings
    variant = "txt2img"
    controlnet_path = None
    controlnet_strength = 1.0
    controlnet_image_path = None

    if masked_image_path:
        variant = "fill"
    elif depth_image_path:
        variant = "depth"
    elif redux_image_paths:
        variant = "redux"
    elif ControlNet is not None and isinstance(ControlNet, MfluxControlNetPipeline):
        variant = "controlnet"
        controlnet_path = ControlNet.model_selection
        controlnet_strength = float(ControlNet.control_strength)
        controlnet_image_path = ControlNet.control_image_path

    # 3. Load Model
    flux = load_or_create_flux(
        model_name=model_resolved,
        quantize=q_val,
        path=Local_model if Local_model else None,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        variant=variant,
        controlnet_path=controlnet_path,
        base_model=base_model
    )

    # 4. Prepare Generation Arguments
    seed_val = int(seed) if seed != -1 else random.randint(0, 0xffffffffffffffff)

    gen_kwargs = {
        "seed": seed_val,
        "prompt": prompt,
        "num_inference_steps": steps,
        "height": height,
        "width": width,
        "guidance": guidance,
        # Restore parameters that were ignored in previous version
        # Note: Backend support for low_ram/vae_tiling varies by version,
        # but passing them is better than ignoring them.
        "low_ram": low_ram,
        "vae_tiling": vae_tiling,
        "vae_tiling_split": vae_tiling_split,
    }

    # Z-Image Turbo does not accept 'guidance'
    if isinstance(flux, ZImageTurbo):
        if "guidance" in gen_kwargs:
            del gen_kwargs["guidance"]

    # Add variant-specific arguments
    if variant == "fill":
        # Flux1Fill requires image_path and masked_image_path
        if image: gen_kwargs["image_path"] = image.image_path
        gen_kwargs["masked_image_path"] = masked_image_path
    elif variant == "depth":
        if image: gen_kwargs["image_path"] = image.image_path
        gen_kwargs["depth_image_path"] = depth_image_path
    elif variant == "redux":
        gen_kwargs["redux_image_paths"] = redux_image_paths
        if redux_image_strengths:
            gen_kwargs["redux_image_strengths"] = redux_image_strengths
    elif variant == "controlnet":
        gen_kwargs["controlnet_image_path"] = controlnet_image_path
        gen_kwargs["controlnet_strength"] = controlnet_strength
    else:
        # Standard txt2img / img2img
        if image:
            gen_kwargs["image_path"] = image.image_path
            gen_kwargs["image_strength"] = image.image_strength

    print(f"[MFlux-ComfyUI] Generating ({variant}) seed: {seed_val}, steps: {steps}, dims: {width}x{height}")

    # 5. Register Progress Bar Callback
    if CallbackRegistry:
        flux.callbacks = CallbackRegistry()
        pbar = ComfyUIProgressBarCallback(total_steps=steps)
        flux.callbacks.register(pbar)

    # 6. Generate
    try:
        generated_result = flux.generate_image(**gen_kwargs)
    except TypeError as e:
        # Retry without advanced kwargs if backend rejects them
        if "unexpected keyword argument" in str(e):
            print(f"[MFlux-ComfyUI] Warning: Backend rejected some arguments ({e}). Retrying with basics...")
            for k in ["low_ram", "vae_tiling", "vae_tiling_split"]:
                if k in gen_kwargs:
                    del gen_kwargs[k]
            generated_result = flux.generate_image(**gen_kwargs)
        else:
            raise e
    except Exception as e:
        print(f"[MFlux-ComfyUI] Error during generation: {e}")
        raise e

    # 7. Process Output
    # Handle different return types from mflux versions (object with .image or raw)
    if hasattr(generated_result, "image"):
        pil_image = generated_result.image
    else:
        pil_image = generated_result

    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    return (image_tensor,)

def save_images_with_metadata(images, prompt, model, quantize, Local_model, seed, height, width, steps, guidance, lora_paths, lora_scales, image_path, image_strength, filename_prefix="Mflux", full_prompt=None, extra_pnginfo=None, base_model=None, low_ram=False, control_image_path=None, control_strength=None, control_model=None, quantize_effective=None, vae_tiling=False, vae_tiling_split="horizontal"):

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
    for image in images:
        i = 255. * image.cpu().numpy().squeeze()
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
            "model": model,
            "base_model": base_model,
            "quantize": quantize,
            "quantize_effective": quantize_effective,
            "seed": seed,
            "height": height,
            "width": width,
            "steps": steps,
            "guidance": guidance,
            "Local_model": Local_model,
            "image_path": image_path,
            "image_strength": image_strength,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
            "control_image_path": control_image_path,
            "control_strength": control_strength,
            "masked_image_path": extra_pnginfo.get("masked_image_path") if extra_pnginfo else None,
            "depth_image_path": extra_pnginfo.get("depth_image_path") if extra_pnginfo else None,
            "redux_image_paths": extra_pnginfo.get("redux_image_paths") if extra_pnginfo else None,
            "mflux_version": _get_mflux_version(),
            "low_ram": low_ram,
            "vae_tiling": vae_tiling,
            "vae_tiling_split": vae_tiling_split,
        }
        with open(metadata_jsonfile, 'w') as metadata_file:
            json.dump(json_dict, metadata_file, indent=4)

        counter += 1

    return {"ui": {"images": results}, "counter": counter}