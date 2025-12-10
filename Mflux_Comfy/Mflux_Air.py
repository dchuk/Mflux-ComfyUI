import os
import json
from folder_paths import models_dir
from .Mflux_Core import get_lora_info, generate_image, save_images_with_metadata, infer_quant_bits, is_third_party_model

# --- MFLUX 0.13.1 Imports with CI Guards ---
try:
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.fibo.variants.txt2img.fibo import FIBO
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
    from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
    from mflux.callbacks.instances.memory_saver import MemorySaver
except ImportError:
    ModelConfig = None
    Flux1 = None
    FIBO = None
    QwenImage = None
    ZImageTurbo = None
    MemorySaver = None

try:
    from huggingface_hub import snapshot_download, scan_cache_dir
except ImportError:
    snapshot_download = None
    scan_cache_dir = None

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

mflux_dir = os.path.join(models_dir, "Mflux")
create_directory(mflux_dir)

def get_full_model_path(model_dir, model_name):
    return os.path.join(model_dir, model_name)

def download_hg_model(model_version, force_redownload=False):
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required. Please install it: pip install huggingface_hub")
    repo_id = model_version
    model_checkpoint = get_full_model_path(mflux_dir, model_version)
    if os.path.exists(model_checkpoint) and not force_redownload:
        print(f"Model {model_version} found at {model_checkpoint}")
        return model_checkpoint
    print(f"Downloading {repo_id} to {model_checkpoint}...")
    snapshot_download(repo_id=repo_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
    return model_checkpoint

# Helper to detect if a directory looks like a model root
def is_model_directory(path):
    indicators = ["config.json", "transformer", "vae", "text_encoder", "text_encoder_2", "model_index.json"]
    try:
        items = os.listdir(path)
        return any(item in items for item in indicators)
    except Exception:
        return False

# Helper to get list of cached model repo IDs
def get_cached_models():
    if scan_cache_dir is None:
        return []
    try:
        cache_info = scan_cache_dir()
        # Filter for models only, return sorted list of repo_ids
        models = [repo.repo_id for repo in cache_info.repos if repo.repo_type == "model"]
        return sorted(models, key=str.lower)
    except Exception:
        return []

class MfluxOptimizations:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_ram": ("BOOLEAN", {"default": False, "tooltip": "OFF = Fastest. ON = Saves RAM (use for 8GB Macs or 8/16-bit models) but slows generation by ~20%."}),
                "vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "OFF = Fastest. ON = Prevents crashes at high resolutions (>1536px) or during upscaling."}),
                "vae_tiling_split": (["horizontal", "vertical"], {"default": "horizontal", "tooltip": "Direction for VAE tiling. 'horizontal' is generally better for avoiding facial seams."}),
            }
        }

    RETURN_TYPES = ("MFLUX_OPTS",)
    RETURN_NAMES = ("optimizations",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "get_optimizations"

    def get_optimizations(self, low_ram, vae_tiling, vae_tiling_split):
        opts = {
            "low_ram": low_ram,
            "vae_tiling": vae_tiling,
            "vae_tiling_split": vae_tiling_split,
        }
        return (opts,)

class MfluxModelsDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit",
                    "dhairyashil/FLUX.1-dev-mflux-4bit",
                    "filipstrand/FLUX.1-Krea-dev-mflux-4bit",
                    "akx/FLUX.1-Kontext-dev-mflux-4bit",
                    "filipstrand/Z-Image-Turbo-mflux-4bit",
                    "briaai/Fibo-mlx-4bit",
                    "briaai/Fibo-mlx-8bit",
                    "filipstrand/Qwen-Image-mflux-6bit",
                ], {"default": "dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit"}),
            },
            "optional": {
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("downloaded_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "download_model"

    def download_model(self, model, force_redownload=False):
        model_path = download_hg_model(model, force_redownload=force_redownload)
        return (model_path,)

class MfluxCustomModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "schnell", "tooltip": "Model alias or path. E.g., 'schnell', '/path/to/model', or 'dhairyashil/FLUX.1-dev-mflux-4bit'"}),
                "quantize": (["3", "4", "5", "6", "8"], {"default": "8"}),
            },
            "optional": {
                "Loras": ("MfluxLorasPipeline",),
                "custom_identifier": ("STRING", {"default": ""}),
                "base_model": (["dev", "schnell", "qwen", "fibo", "z-image-turbo"], {"default": "dev", "tooltip": "Architecture hint. Required if 'model' is a custom path. Alias 'base_model' is ignored if 'model' is a Hugging Face ID."}),
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Custom_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "save_model"

    def save_model(self, model, quantize, Loras=None, custom_identifier="", base_model="dev"):
        if Flux1 is None and FIBO is None and QwenImage is None and ZImageTurbo is None:
            raise ImportError("mflux is not installed or failed to load.")

        identifier = custom_identifier if custom_identifier else "default"

        # Determine model type for saving directory
        model_type_hint = base_model
        if model_type_hint == "z-image-turbo":
            model_type_hint = "z_image_turbo"
        save_dir_name = f"Mflux-{model_type_hint}-{quantize}bit-{identifier}"
        save_dir = get_full_model_path(mflux_dir, save_dir_name)
        create_directory(save_dir)

        lora_paths, lora_scales = get_lora_info(Loras)
        quantize_int = int(quantize)

        # Determine which model class to use based on base_model hint
        target_class = None
        config_base_model = None
        if base_model == "qwen":
            target_class = QwenImage
            config_base_model = "qwen-image"
        elif base_model == "fibo":
            target_class = FIBO
            config_base_model = "fibo"
        elif base_model == "z-image-turbo":
            target_class = ZImageTurbo
            config_base_model = "z-image-turbo"
        else:
            # Default to Flux
            target_class = Flux1
            config_base_model = base_model

        if is_third_party_model(model) or "/" in str(model):
            model_config = ModelConfig.from_name(model_name=model, base_model=config_base_model)
        else:
            model_config = ModelConfig.from_name(model_name=model, base_model=config_base_model)

        instance = target_class(
            model_config=model_config,
            quantize=quantize_int,
            lora_paths=lora_paths,
            lora_scales=lora_scales
        )
        instance.save_model(save_dir)
        print(f"Model saved to {save_dir}")
        return (save_dir,)

class MfluxModelsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # 1. Find Local Models (Recursive)
        local_models = []
        if os.path.exists(mflux_dir):
            for root, dirs, files in os.walk(mflux_dir):
                if is_model_directory(root):
                    rel_path = os.path.relpath(root, mflux_dir)
                    # Add Folder Icon
                    local_models.append(f"📁 {rel_path}")
                    dirs[:] = []  # Stop recursing
        local_models.sort(key=str.lower)

        # 2. Check System Cache for ALL models
        # This lists the actual Repo IDs found in the cache
        cached_models = [f"🟢 {repo}" for repo in get_cached_models()]

        # 3. Aliases (Shortcuts)
        # These are abstract names that mflux resolves internally.
        # We mark them with a Cloud to indicate they are shortcuts/downloads.
        aliases = ["dev", "schnell", "qwen", "fibo", "z-image-turbo"]
        alias_options = [f"☁️ {alias}" for alias in aliases]

        final_options = local_models + cached_models + alias_options
        return {
            "required": {
                "model": (final_options or ["None"], {"default": final_options[0] if final_options else "None", "tooltip": "📁 = Local (ComfyUI/models/Mflux)\n🟢 = Cached (HuggingFace System Cache)\n☁️ = Alias/Shortcut (May trigger download)"}),
            },
            "optional": {
                "free_path": ("STRING", {"default": "", "tooltip": "Manually input an absolute model path to override the selection above."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_to_load",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "load"

    def load(self, model="", free_path=""):
        if free_path:
            if not os.path.exists(free_path):
                raise ValueError(f"Path does not exist: {free_path}")
            return (free_path,)

        if model and model != "None":
            # Clean up the visual indicators
            clean_name = model.replace("🟢 ", "").replace("📁 ", "").replace("☁️ ", "")

            # If it was a local folder (📁), resolve full path
            if "📁 " in model:
                full_path = get_full_model_path(mflux_dir, clean_name)
                if os.path.exists(full_path):
                    return (full_path,)
                else:
                    print(f"Warning: Local model path {full_path} not found. Passing name '{clean_name}' as string.")
                    return (clean_name,)

            # If it was a cached repo (🟢) or alias (☁️), return the string as-is
            return (clean_name,)

        return ("",)

class QuickMfluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Luxury food photograph"}),
                "model": ("STRING", {"default": "schnell", "tooltip": "1. ALIAS: Type 'dev', 'schnell', 'qwen', 'fibo', or 'z-image-turbo'. 'base_model' is ignored if alias is used.\n2. PATH: Type a path or connect the Downloader/Loader node. You MUST set 'base_model' to match the architecture (e.g., 'qwen', 'fibo')."}),
                "quantize": (["Auto", "None", "3", "4", "5", "6", "8"], {"default": "Auto", "tooltip": "Auto/None = Use model's native weights. Number = Force quantization."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Loras": ("MfluxLorasPipeline",),
                "img2img": ("MfluxImg2ImgPipeline",),
                "ControlNet": ("MfluxControlNetPipeline",),
                "base_model": (["dev", "schnell", "qwen", "fibo", "z-image-turbo"], {"default": "dev", "tooltip": "Architecture hint. Required if 'model' is a custom path. Alias 'base_model' is ignored if 'model' is a Hugging Face ID or known alias."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Negative prompt. Only used by Qwen models."}),
                "optimizations": ("MFLUX_OPTS", {"tooltip": "Connect MfluxOptimizations node here for hardware-specific settings."}),
            },
            "hidden": {
                # Legacy input hidden from UI but functional for old workflows
                "Local_model": ("PATH",),
                "full_prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "generate"

    def generate(self, prompt, model, seed, width, height, steps, guidance, quantize="Auto", metadata=True, img2img=None, Loras=None, ControlNet=None, base_model="dev", negative_prompt="", optimizations=None, Local_model=None, full_prompt=None, extra_pnginfo=None, size_preset="Custom", apply_size_preset=True, quality_preset="Balanced (25 steps)", apply_quality_preset=True, randomize_seed=True):
        # Safety check for empty prompt to prevent backend crash
        if not prompt or not prompt.strip():
            print("[MFlux-ComfyUI] Warning: Empty prompt detected. Using fallback '.' to prevent backend crash.")
            prompt = "."

        # Priority: 1. Local_model (legacy dock), 2. model (text widget/input)
        final_model = model
        if Local_model and isinstance(Local_model, str) and Local_model.strip():
            final_model = Local_model

        final_width, final_height = width, height
        final_steps, final_guidance = steps, guidance
        final_seed = -1 if randomize_seed else seed

        low_ram = optimizations.get("low_ram", False) if optimizations else False
        vae_tiling = optimizations.get("vae_tiling", False) if optimizations else False
        vae_tiling_split = optimizations.get("vae_tiling_split", "horizontal") if optimizations else "horizontal"

        # Handle Auto quantization
        q_val = None
        if quantize in ("Auto", "None"):
            q_val = None
        else:
            q_val = quantize

        generated_images = generate_image(
            prompt,
            final_model,
            final_seed,
            final_width,
            final_height,
            final_steps,
            final_guidance,
            q_val,
            metadata,
            model_path=final_model,
            img2img_pipeline=img2img,
            loras_pipeline=Loras,
            controlnet_pipeline=ControlNet,
            base_model_hint=base_model,
            negative_prompt=negative_prompt,
            low_ram=low_ram,
            vae_tiling=vae_tiling,
            vae_tiling_split=vae_tiling_split
        )

        if metadata:
            image_path = img2img.image_path if img2img else None
            image_strength = img2img.image_strength if img2img else None
            lora_paths, lora_scales = get_lora_info(Loras)

            quantize_effective = quantize
            if quantize in ("Auto", "None"):
                detected = infer_quant_bits(final_model)
                if detected:
                    quantize_effective = f"Auto ({detected}-bit)"
                else:
                    quantize_effective = "Auto (Native)"

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
                model_alias=final_model,
                quantize=quantize,
                quantize_effective=quantize_effective,
                model_path=final_model,
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
                base_model_hint=base_model,
                negative_prompt_used=negative_prompt,
                vae_tiling=vae_tiling,
                vae_tiling_split=vae_tiling_split,
                low_ram=low_ram
            )

        return generated_images

class MfluxZImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A cinematic shot of..."}),
                "model": ("STRING", {"default": "filipstrand/Z-Image-Turbo-mflux-4bit", "tooltip": "Model alias or path. E.g., 'z-image-turbo' or a Hugging Face ID."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 50, "tooltip": "Z-Image Turbo is optimized for 9 steps."}),
                "metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Full quantization range + Auto
                "quantize": (["Auto", "None", "3", "4", "5", "6", "8"], {"default": "Auto", "tooltip": "Auto/None = Use model's native weights. Number = Force quantization."}),
                "Loras": ("MfluxLorasPipeline",),
                "img2img": ("MfluxImg2ImgPipeline",),
                "optimizations": ("MFLUX_OPTS", {"tooltip": "Connect MfluxOptimizations node here for Low RAM mode."}),
            },
            "hidden": {
                # Legacy input hidden
                "Local_model": ("PATH",),
                "full_prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "generate"

    def generate(self, prompt, model, seed, width, height, steps, metadata=True, quantize="Auto", Loras=None, img2img=None, optimizations=None, Local_model=None, full_prompt=None, extra_pnginfo=None):
        # Safety check for empty prompt to prevent backend crash
        if not prompt or not prompt.strip():
            print("[MFlux-ComfyUI] Warning: Empty prompt detected. Using fallback '.' to prevent backend crash.")
            prompt = "."

        # Priority: 1. Local_model, 2. model
        final_model = model
        if Local_model and isinstance(Local_model, str) and Local_model.strip():
            final_model = Local_model

        guidance = 0.0

        # Handle Auto quantization
        q_val = None
        if quantize in ("Auto", "None"):
            q_val = None
        else:
            q_val = quantize

        low_ram = optimizations.get("low_ram", False) if optimizations else False
        vae_tiling = optimizations.get("vae_tiling", False) if optimizations else False
        vae_tiling_split = optimizations.get("vae_tiling_split", "horizontal") if optimizations else "horizontal"

        generated_images = generate_image(
            prompt,
            final_model,
            seed,
            width,
            height,
            steps,
            guidance,
            q_val,
            metadata,
            model_path=final_model,
            img2img_pipeline=img2img,
            loras_pipeline=Loras,
            controlnet_pipeline=None,
            base_model_hint="z-image-turbo",
            negative_prompt="",
            low_ram=low_ram,
            vae_tiling=vae_tiling,
            vae_tiling_split=vae_tiling_split
        )

        if metadata:
            image_path = img2img.image_path if img2img else None
            image_strength = img2img.image_strength if img2img else None
            lora_paths, lora_scales = get_lora_info(Loras)

            quantize_effective = quantize
            if quantize in ("Auto", "None"):
                detected = infer_quant_bits(final_model)
                if detected:
                    quantize_effective = f"Auto ({detected}-bit)"
                else:
                    quantize_effective = "Auto (Native)"

            save_images_with_metadata(
                images=generated_images,
                prompt=prompt,
                model_alias=final_model,
                quantize=quantize,
                quantize_effective=quantize_effective,
                model_path=final_model,
                seed=seed,
                height=height,
                width=width,
                steps=steps,
                guidance=guidance,
                image_path=image_path,
                image_strength=image_strength,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                control_image_path=None,  # Z-Image doesn't support ControlNet
                control_strength=None,
                control_model=None,
                full_prompt=full_prompt,
                extra_pnginfo=extra_pnginfo,
                base_model_hint="z-image-turbo",
                negative_prompt_used="",
                vae_tiling=vae_tiling,
                vae_tiling_split=vae_tiling_split,
                low_ram=low_ram
            )

        return generated_images

class MfluxFiboPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # We explicitly disable dynamicPrompts here so JSON is safe
                "json_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "get_prompt"

    def get_prompt(self, json_prompt):
        return (json_prompt,)
