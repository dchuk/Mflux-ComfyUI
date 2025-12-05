import os
import json
from folder_paths import models_dir
from typing import Any, Dict, Optional

# --- MFLUX Imports with Guards ---
_skip_mflux_import = os.environ.get("MFLUX_COMFY_DISABLE_MFLUX_IMPORT") == "1"

try:
    if _skip_mflux_import:
        raise ImportError("Skipping Mflux import via env var")
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1
except ImportError:
    ModelConfig = None
    Flux1 = None

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

from .Mflux_Core import get_lora_info, generate_image, save_images_with_metadata, infer_quant_bits, is_third_party_model

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

mflux_dir = os.path.join(models_dir, "Mflux")
create_directory(mflux_dir)

def get_full_model_path(model_dir, model_name):
    return os.path.join(model_dir, model_name)

def _marker_path(dir_path: str) -> str:
    return os.path.join(dir_path, ".mflux_download.json")

def _write_marker(dir_path: str, repo_id: str):
    try:
        files_count = 0
        for root, _, files in os.walk(dir_path):
            files_count += len([f for f in files if not f.startswith(".")])
        data = {"repo_id": repo_id, "files_count": files_count}
        with open(_marker_path(dir_path), "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[MFlux-ComfyUI] Warning: Failed to write marker for {dir_path}: {e}")

def _has_marker(dir_path: str) -> bool:
    return os.path.exists(_marker_path(dir_path))

def _looks_like_model_root(dir_path: str) -> bool:
    """Heuristic to decide if a directory is a model root."""
    if _has_marker(dir_path):
        return True
    try:
        # Exclude any path that clearly lives under caches
        lowered = dir_path.lower()
        if any(seg in lowered for seg in ("/cache/", "/.cache/", "/huggingface/", "/download/")):
            return False
        entries = [e.name for e in os.scandir(dir_path) if e.is_dir()]
        # Typical mflux model roots have some of these components as immediate subfolders
        typical = {"vae", "tokenizer", "text_encoder", "text_encoder_2", "transformer"}
        if len(typical.intersection(set(entries))) >= 2:
            return True
        # Or the directory contains a non-trivial number of files at its root
        file_count = sum(1 for e in os.scandir(dir_path) if e.is_file())
        if file_count >= 5:
            return True
    except Exception:
        pass
    return False

def download_hg_model(model_version, force_redownload=False):
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required for model downloads. "
            "Activate your ComfyUI virtual environment and install it with: "
            "pip install 'huggingface_hub>=0.26.0'"
        )

    repo_id = model_version if "/" in model_version else (f"madroid/{model_version}" if "4bit" in model_version else f"AITRADER/{model_version}")
    model_checkpoint = get_full_model_path(mflux_dir, model_version)

    must_download = True
    if os.path.exists(model_checkpoint):
        if _has_marker(model_checkpoint) or _looks_like_model_root(model_checkpoint):
            if force_redownload:
                print(f"Model {model_version} exists but force_redownload=True. Re-downloading...")
            else:
                print(f"Model {model_version} already exists at {model_checkpoint}. Skipping download.")
                must_download = False
        else:
            print(f"Model folder exists without completion marker: {model_checkpoint}. Resuming download...")

    if must_download:
        print(f"Downloading {repo_id} to {model_checkpoint}...")
        try:
            snapshot_download(repo_id=repo_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
            _write_marker(model_checkpoint, repo_id)
        except Exception as e:
            raise RuntimeError(f"Failed to download '{repo_id}': {e}") from e

    return model_checkpoint

class MfluxModelsDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": ([
                    "flux.1-schnell-mflux-4bit",
                    "flux.1-dev-mflux-4bit",
                    "MFLUX.1-schnell-8-bit",
                    "MFLUX.1-dev-8-bit",
                    "filipstrand/FLUX.1-Krea-dev-mflux-4bit",
                    "akx/FLUX.1-Kontext-dev-mflux-4bit",
                    "Tongyi-MAI/Z-Image-Turbo",
                    "filipstrand/Z-Image-Turbo-mflux-4bit",
                ], {"default": "flux.1-schnell-mflux-4bit"}),
            },
            "optional": {
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Downloaded_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "download_model"

    def download_model(self, model_version, force_redownload=False):
        model_path = download_hg_model(model_version, force_redownload=force_redownload)
        return (model_path,)

class MfluxCustomModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["dev", "schnell", "z-image-turbo"], {"default": "schnell"}),
                "quantize": (["3", "4", "5", "6", "8"], {"default": "8"}),
            },
            "optional": {
                "Loras": ("MfluxLorasPipeline",),
                "custom_identifier": ("STRING", {"default": ""}),
                "base_model": (["dev", "schnell"], {"default": "dev"}),
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Custom_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "save_model"

    def save_model(self, model, quantize, Loras=None, custom_identifier="", base_model="dev"):
        if Flux1 is None or ModelConfig is None:
            raise ImportError("MFlux is not installed/loaded properly.")

        identifier = custom_identifier if custom_identifier else "default"
        save_dir = get_full_model_path(mflux_dir, f"Mflux-{model}-{quantize}bit-{identifier}")
        create_directory(save_dir)

        lora_paths, lora_scales = get_lora_info(Loras)

        # Support HF repo ids with base_model; else use alias via from_name
        if is_third_party_model(model) or "/" in str(model):
            model_config = ModelConfig.from_name(model_name=model, base_model=base_model)
        else:
            model_config = ModelConfig.from_name(model_name=model, base_model=None)

        # mflux 0.13.1 Flux1 constructor
        flux = Flux1(
            model_config=model_config,
            quantize=int(quantize),
            lora_paths=lora_paths,
            lora_scales=lora_scales
        )

        flux.save_model(save_dir)
        _write_marker(save_dir, f"custom:{model}:{quantize}")
        print(f"Model saved to {save_dir}")
        return (save_dir,)

class MfluxModelsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls.get_sorted_model_paths() or ["None"],),
            },
            "optional": {
                "free_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Local_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "load"

    @classmethod
    def get_sorted_model_paths(cls):
        candidates = set()
        try:
            if os.path.exists(mflux_dir):
                for entry in os.scandir(mflux_dir):
                    if not entry.is_dir():
                        continue
                    if _looks_like_model_root(entry.path):
                        candidates.add(entry.name)
        except Exception:
            return []
        return sorted(list(candidates))

    def load(self, model_name="", free_path=""):
        if free_path:
            if not os.path.exists(free_path):
                raise ValueError(f"Path does not exist: {free_path}")
            return (free_path,)

        if model_name and model_name != "None":
            return (get_full_model_path(mflux_dir, model_name),)

        return ("",)

class QuickMfluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Luxury food photograph"}),
                "model": (["dev", "schnell"], {"default": "schnell"}),
                "quantize": (["None", "3", "4", "5", "6", "8"], {"default": "8"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Local_model": ("PATH",),
                "Loras": ("MfluxLorasPipeline",),
                "img2img": ("MfluxImg2ImgPipeline",),
                "ControlNet": ("MfluxControlNetPipeline",),
                "base_model": (["dev", "schnell"], {"default": "dev"}),
                "low_ram": ("BOOLEAN", {"default": False}),
                "vae_tiling": ("BOOLEAN", {"default": False}),
                "vae_tiling_split": (["horizontal", "vertical"], {"default": "horizontal"}),
                # Presets
                "size_preset": (["Custom", "512x512", "768x1024", "1024x1024", "1024x768"], {"default": "Custom"}),
                "apply_size_preset": ("BOOLEAN", {"default": True}),
                "quality_preset": (["Balanced (25 steps)", "Fast (12 steps)", "High Quality (35 steps)", "Custom"], {"default": "Balanced (25 steps)"}),
                "apply_quality_preset": ("BOOLEAN", {"default": True}),
                "randomize_seed": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "full_prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "generate"

    def generate(self, prompt, model, seed, width, height, steps, guidance, quantize="None", metadata=True, Local_model="", img2img=None, Loras=None, ControlNet=None, base_model="dev", low_ram=False, full_prompt=None, extra_pnginfo=None, size_preset="Custom", apply_size_preset=True, quality_preset="Balanced (25 steps)", apply_quality_preset=True, randomize_seed=True, vae_tiling=False, vae_tiling_split="horizontal"):

        # Apply presets logic
        final_width, final_height = width, height
        if apply_size_preset and size_preset != "Custom" and "x" in size_preset:
            try:
                w, h = size_preset.split("x")
                final_width, final_height = int(w), int(h)
            except ValueError:
                pass

        final_steps, final_guidance = steps, guidance
        if apply_quality_preset and quality_preset != "Custom":
            if "Fast" in quality_preset:
                final_steps = 12
            elif "High Quality" in quality_preset:
                final_steps = 35
            else:
                final_steps = 25

        final_seed = -1 if randomize_seed else seed

        generated_images = generate_image(
            prompt, model, final_seed, final_width, final_height, final_steps, final_guidance, quantize, metadata,
            Local_model, img2img, Loras, ControlNet, base_model=base_model, low_ram=low_ram,
            vae_tiling=vae_tiling, vae_tiling_split=vae_tiling_split
        )

        if metadata:
            image_path = img2img.image_path if img2img else None
            image_strength = img2img.image_strength if img2img else None
            lora_paths, lora_scales = get_lora_info(Loras)

            quantize_effective = quantize
            if Local_model:
                detected = infer_quant_bits(Local_model)
                if detected: quantize_effective = f"{detected}-bit"

            control_image_path = None
            control_strength = None
            control_model = None
            if ControlNet:
                control_image_path = getattr(ControlNet, "control_image_path", None)
                control_strength = getattr(ControlNet, "control_strength", None)
                control_model = getattr(ControlNet, "model_selection", None)

            save_images_with_metadata(
                images=generated_images,
                prompt=prompt,
                model=model,
                quantize=quantize,
                quantize_effective=quantize_effective,
                Local_model=Local_model,
                seed=final_seed,
                height=final_height,
                width=final_width,
                steps=final_steps,
                guidance=final_guidance,
                image_path=image_path,
                image_strength=image_strength,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                control_image_path=control_image_path,
                control_strength=control_strength,
                control_model=control_model,
                full_prompt=full_prompt,
                extra_pnginfo=extra_pnginfo,
                base_model=base_model,
                vae_tiling=vae_tiling,
                vae_tiling_split=vae_tiling_split
            )

        return generated_images

class MfluxZImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A cinematic shot of..."}),
                "model": ([
                    "filipstrand/Z-Image-Turbo-mflux-4bit",
                    "Tongyi-MAI/Z-Image-Turbo",
                ], {"default": "filipstrand/Z-Image-Turbo-mflux-4bit"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 50, "tooltip": "Z-Image Turbo is optimized for 9 steps."}),
                "metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "quantize": (["None", "4", "8"], {"default": "None", "tooltip": "Use '4' if loading the full model and want to save RAM. Leave 'None' for pre-quantized models."}),
                "Local_model": ("PATH",),
                "Loras": ("MfluxLorasPipeline",),
                "img2img": ("MfluxImg2ImgPipeline",),
            },
            "hidden": {
                "full_prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "generate"

    # Z-Image specific generation function
    def generate(self, prompt, model, seed, width, height, steps, metadata=True, quantize="None", Local_model="", img2img=None, Loras=None, full_prompt=None, extra_pnginfo=None):

        # Force guidance to 0 for Z-Image
        guidance = 0.0

        # If user selected the 4-bit model but didn't set quantize, force it to 4 to match the weights
        if "4bit" in model and quantize == "None":
            quantize = "4"

        generated_images = generate_image(
            prompt, model, seed, width, height, steps, guidance, quantize, metadata,
            Local_model, img2img, Loras, ControlNet=None, base_model="dev", low_ram=False
        )

        if metadata:
            image_path = img2img.image_path if img2img else None
            image_strength = img2img.image_strength if img2img else None
            lora_paths, lora_scales = get_lora_info(Loras)

            quantize_effective = quantize
            if Local_model:
                detected = infer_quant_bits(Local_model)
                if detected: quantize_effective = f"{detected}-bit"

            save_images_with_metadata(
                images=generated_images,
                prompt=prompt,
                model=model,
                quantize=quantize,
                quantize_effective=quantize_effective,
                Local_model=Local_model,
                seed=seed,
                height=height,
                width=width,
                steps=steps,
                guidance=guidance,
                image_path=image_path,
                image_strength=image_strength,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                full_prompt=full_prompt,
                extra_pnginfo=extra_pnginfo,
                base_model="z-image" # Just for metadata reference
            )

        return generated_images