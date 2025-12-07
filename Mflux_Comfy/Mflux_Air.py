import os
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
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

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

class MfluxOptimizations:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_ram": ("BOOLEAN", {"default": False, "tooltip": "Enables aggressive memory saving during generation. Can reduce quality on some systems."}),
                "vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Splits image decoding to prevent VRAM crashes on large images. Only affects Flux models."}),
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
                "model_version": ([
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
    RETURN_NAMES = ("model_string",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "download_model"

    def download_model(self, model_version, force_redownload=False):
        model_path = download_hg_model(model_version, force_redownload=force_redownload)
        # Return the path as a string, so it can be connected to the 'model' input of QuickMfluxNode
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
                "base_model": (["dev", "schnell", "qwen", "fibo", "z-image-turbo"], {"default": "dev", "tooltip": "Architecture hint. Required if 'model' is a custom path. Aliias 'base_model' is ignored if 'model' is a Hugging Face ID."}),
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
        config_base_model = None # For ModelConfig.from_name

        if base_model == "qwen":
            target_class = QwenImage
            config_base_model = "qwen-image"
        elif base_model == "fibo":
            target_class = FIBO
            config_base_model = "fibo"
        elif base_model == "z-image-turbo":
            target_class = ZImageTurbo
            config_base_model = "z-image-turbo"
        else: # Default to Flux for "dev", "schnell", or other hints
            target_class = Flux1
            config_base_model = base_model # e.g. "dev", "schnell"

        if is_third_party_model(model) or "/" in str(model):
            # If it's a full path or HF ID, base_model hint is crucial for ModelConfig
            model_config = ModelConfig.from_name(model_name=model, base_model=config_base_model)
        else:
            # If it's an alias, base_model hint is still passed
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
        model_paths = []
        if os.path.exists(mflux_dir):
            try:
                model_paths = [f.name for f in os.scandir(mflux_dir) if f.is_dir()]
            except OSError:
                pass

        # Add common aliases for convenience
        available_aliases = ["dev", "schnell", "qwen", "fibo", "z-image-turbo"]
        # Filter out aliases that might conflict with directory names if a user named a folder "dev" etc.
        # This is a simple heuristic; more robust alias handling might be needed later.
        final_options = available_aliases.copy()
        for path_name in model_paths:
            if path_name not in available_aliases:
                 # Heuristic to add custom model dirs if they seem to follow a pattern
                if any(model_type in path_name.lower() for model_type in ["flux", "fibo", "qwen", "z-image"]):
                     if path_name not in final_options:
                        final_options.append(path_name)

        final_options.sort()


        return {
            "required": {
                "model_name": (final_options or ["None"], {"default": "schnell"}),
            },
            "optional": {
                "free_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",) # Changed to STRING to match new model input type
    RETURN_NAMES = ("model_string",) # Changed to model_string
    CATEGORY = "MFlux/Air"
    FUNCTION = "load"

    def load(self, model_name="", free_path=""):
        if free_path:
            if not os.path.exists(free_path):
                raise ValueError(f"Path does not exist: {free_path}")
            return (free_path,)

        if model_name and model_name != "None":
            # If it's a known alias, return the alias itself
            if model_name in ["dev", "schnell", "qwen", "fibo", "z-image-turbo"]:
                return (model_name,)
            # Otherwise, assume it's a path to a local directory
            full_path = get_full_model_path(mflux_dir, model_name)
            if os.path.exists(full_path):
                return (full_path,)
            else:
                # Fallback: return the name, let downstream handle it
                print(f"Warning: Local model path {full_path} not found. Passing model name '{model_name}' as string.")
                return (model_name,)


        return ("",) # Return empty string if nothing selected

class QuickMfluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Luxury food photograph"}),
                "model": ("STRING", {"default": "schnell", "tooltip": "1. ALIAS: Type 'dev', 'schnell', 'qwen', 'fibo', or 'z-image-turbo'. 'base_model' is ignored if alias is used.\n2. PATH: Type a path or connect the Downloader/Loader node. You MUST set 'base_model' to match the architecture (e.g., 'qwen', 'fibo')."}),
                "quantize": (["None", "3", "4", "5", "6", "8"], {"default": "8"}),
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
                "full_prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "generate"

    def generate(self, prompt, model, seed, width, height, steps, guidance, quantize="None", metadata=True, img2img=None, Loras=None, ControlNet=None, base_model="dev", negative_prompt="", optimizations=None, full_prompt=None, extra_pnginfo=None, size_preset="Custom", apply_size_preset=True, quality_preset="Balanced (25 steps)", apply_quality_preset=True, randomize_seed=True):

        final_width, final_height = width, height
        final_steps, final_guidance = steps, guidance
        final_seed = -1 if randomize_seed else seed

        low_ram = optimizations.get("low_ram", False) if optimizations else False
        vae_tiling = optimizations.get("vae_tiling", False) if optimizations else False
        vae_tiling_split = optimizations.get("vae_tiling_split", "horizontal") if optimizations else "horizontal"

        generated_images = generate_image(
            prompt, model, final_seed, final_width, final_height, final_steps, final_guidance, quantize, metadata,
            model_path=model, # Pass 'model' string as 'model_path'
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
            # if model_path: # This check needs to be against the actual resolved path
            #     detected = infer_quant_bits(model_path)
            #     if detected: quantize_effective = f"{detected}-bit"

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
                model_alias=model, # Store the alias/path string used for generation
                quantize=quantize,
                quantize_effective=quantize_effective,
                model_path=model, # Store the path string
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
                base_model_hint=base_model, # Store the base model hint
                negative_prompt_used=negative_prompt, # Store if negative prompt was used
                vae_tiling=vae_tiling, # Store vae tiling settings
                vae_tiling_split=vae_tiling_split
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
                "quantize": (["None", "4", "8"], {"default": "None", "tooltip": "Use '4' if loading the full model and want to save RAM. Leave 'None' for pre-quantized models."}),
                "Loras": ("MfluxLorasPipeline",),
                "img2img": ("MfluxImg2ImgPipeline",),
                "optimizations": ("MFLUX_OPTS", {"tooltip": "Connect MfluxOptimizations node here for Low RAM mode."}),
            },
            "hidden": {
                "full_prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "generate"

    def generate(self, prompt, model, seed, width, height, steps, metadata=True, quantize="None", Loras=None, img2img=None, optimizations=None, full_prompt=None, extra_pnginfo=None):

        guidance = 0.0 # Z-Image does not use guidance

        if "4bit" in model and quantize == "None":
            quantize = "4" # Default quantization for 4bit models

        low_ram = optimizations.get("low_ram", False) if optimizations else False
        # vae_tiling is not applicable to Z-Image, but we'll pass it through to generate_image for consistency if needed
        vae_tiling = optimizations.get("vae_tiling", False) if optimizations else False
        vae_tiling_split = optimizations.get("vae_tiling_split", "horizontal") if optimizations else "horizontal"


        generated_images = generate_image(
            prompt, model, seed, width, height, steps, guidance, quantize, metadata,
            model_path=model, # Pass 'model' string as 'model_path'
            img2img_pipeline=img2img,
            loras_pipeline=Loras,
            controlnet_pipeline=None, # Z-Image node does not have ControlNet
            base_model_hint="z-image-turbo", # Explicit hint for Z-Image
            negative_prompt="", # Z-Image does not support negative prompts
            low_ram=low_ram,
            vae_tiling=vae_tiling, # Will be ignored by ZImageTurbo generation logic
            vae_tiling_split=vae_tiling_split # Will be ignored
        )

        if metadata:
            image_path = img2img.image_path if img2img else None
            image_strength = img2img.image_strength if img2img else None
            lora_paths, lora_scales = get_lora_info(Loras)

            quantize_effective = quantize
            # if model_path: # This check needs to be against the actual resolved path
            #     detected = infer_quant_bits(model_path)
            #     if detected: quantize_effective = f"{detected}-bit"

            save_images_with_metadata(
                images=generated_images,
                prompt=prompt,
                model_alias=model,
                quantize=quantize,
                quantize_effective=quantize_effective,
                model_path=model,
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
                base_model_hint="z-image-turbo",
                negative_prompt_used="", # Z-Image does not use it
                vae_tiling=vae_tiling, # Store even if not used
                vae_tiling_split=vae_tiling_split
            )
        return generated_images
