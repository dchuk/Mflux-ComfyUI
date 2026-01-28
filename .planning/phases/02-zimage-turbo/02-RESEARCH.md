# Phase 2: Z-Image Turbo - Research

**Researched:** 2026-01-28
**Domain:** mflux Z-Image Turbo integration, ComfyUI node patterns, image generation workflows
**Confidence:** HIGH

## Summary

This phase focuses on fixing existing Z-Image Turbo nodes to work correctly with mflux 0.15.5 API. The research confirms that Z-Image Turbo is already partially implemented in the codebase (`MfluxZImageNode` in `Mflux_Air.py` and `MfluxZImageInpaint` in `Mflux_Pro.py`), but based on the CONTEXT.md decisions, we need to create a cleaner separation with dedicated loader/sampler nodes following the user's architectural decisions.

Key findings:
1. **Z-Image Turbo API is stable**: The `ZImageTurbo` class in mflux 0.15.5 has a well-defined `generate_image()` method with specific parameters
2. **No guidance support**: Z-Image Turbo explicitly uses `guidance=0.0` - this is hardcoded in the model
3. **img2img support exists**: The `generate_image()` method accepts `image_path` and `image_strength` parameters for img2img workflows
4. **Phase 1 utilities ready**: `tensor_utils.py` and `memory_utils.py` from Phase 1 provide the conversion and cleanup functions needed

**Primary recommendation:** Create three new node classes (`MfluxZImageLoader`, `MfluxZImageSampler`, `MfluxZImageImg2Img`) following ComfyUI patterns established in the codebase, with the loader handling model/quantization and the samplers handling generation.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mflux | 0.15.5 | MLX-native image generation | Already installed from Phase 1, provides ZImageTurbo class |
| mlx | 0.30.x | Apple ML framework | Required by mflux for GPU compute |
| torch | 2.x | Tensor operations | ComfyUI's core tensor library for IMAGE type |
| Pillow | 10.x | Image I/O | Bridge format between mflux and ComfyUI |
| folder_paths | (ComfyUI) | Model directory resolution | Standard ComfyUI model management |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.26+ | Array operations | Tensor conversion in utils |
| comfy.utils | (ComfyUI) | Progress bars | ProgressBar for step tracking |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Separate loader/sampler | Combined node like `MfluxZImageNode` | User decided: separate nodes for clarity |
| Custom pipeline class | Direct model reference | Pipeline class adds complexity; direct model output simpler |

**Installation:**
```bash
# Already installed from Phase 1
pip install mflux==0.15.5
```

## Architecture Patterns

### Recommended Node Structure

Based on CONTEXT.md decisions and existing codebase patterns:

```
Mflux_Comfy/
├── Mflux_Core.py       # Shared utilities (exists)
├── Mflux_Air.py        # Basic nodes (extend with Z-Image nodes)
└── utils/
    ├── tensor_utils.py # PIL <-> ComfyUI conversion (exists)
    └── memory_utils.py # MLX memory management (exists)
```

### Pattern 1: Loader Node Pattern
**What:** Node that loads model and returns a model reference for downstream use
**When to use:** When model loading should be separated from generation (our case)
**Example:**
```python
# Source: Existing MfluxModelsLoader pattern in Mflux_Air.py
class MfluxZImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Scan models folder, build dropdown
        local_models = _scan_zimage_models()
        return {
            "required": {
                "model": (local_models, {"tooltip": "Select Z-Image Turbo model"}),
                "quantize": (["4", "8", "None"], {"default": "4", "tooltip": "Quantization level: 4-bit saves memory, 8-bit higher quality, None for native weights"}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "mflux"
    FUNCTION = "load_model"
```

### Pattern 2: Sampler Node Pattern (txt2img)
**What:** Node that takes model and prompt, returns IMAGE
**When to use:** For text-to-image generation
**Example:**
```python
# Source: Existing MfluxZImageNode pattern
class MfluxZImageSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ZIMAGE_MODEL", {"tooltip": "Connect MfluxZImageLoader output"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic shot..."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50, "tooltip": "Number of inference steps"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "tooltip": "Image width"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "tooltip": "Image height"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "mflux"
    FUNCTION = "generate"
```

### Pattern 3: img2img Node Pattern
**What:** Separate node for image-to-image generation (per CONTEXT.md decision)
**When to use:** When user wants to transform existing image
**Example:**
```python
# Source: CONTEXT.md decision - separate nodes for txt2img and img2img
class MfluxZImageImg2Img:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ZIMAGE_MODEL", {"tooltip": "Connect MfluxZImageLoader output"}),
                "init_image": ("IMAGE", {"tooltip": "Starting image for transformation"}),
                "prompt": ("STRING", {"multiline": True}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength: 0.0 = original image, 1.0 = ignore image"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "mflux"
    FUNCTION = "generate"
```

### Anti-Patterns to Avoid
- **Combining txt2img and img2img in one node:** User decided separate nodes for clarity
- **Putting quantization on sampler:** User decided quantization belongs on loader only
- **Using `guidance` parameter:** Z-Image Turbo hardcodes guidance=0.0, don't expose it
- **Registering nodes on non-Apple Silicon:** Nodes should not appear at all (silent non-registration)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PIL to ComfyUI tensor | Custom conversion | `utils.tensor_utils.pil_to_comfy_tensor()` | Already implemented in Phase 1, handles edge cases |
| ComfyUI tensor to PIL | Custom conversion | `utils.tensor_utils.comfy_tensor_to_pil()` | Already implemented in Phase 1 |
| MLX memory cleanup | Manual cache clear | `utils.memory_utils.clear_mlx_memory()` | Already implemented with gc.collect() pattern |
| Model folder scanning | os.walk with custom logic | Existing `is_model_directory()` pattern | Already handles config.json detection |
| Model config resolution | Manual ModelConfig | `ModelConfig.z_image_turbo()` or `ModelConfig.from_name()` | mflux provides proper config factory |

**Key insight:** Phase 1 established the utility layer. Z-Image nodes should consume these utilities, not duplicate them.

## Common Pitfalls

### Pitfall 1: Forgetting to Clear MLX Cache
**What goes wrong:** OOM on second generation in same ComfyUI session
**Why it happens:** MLX caches computations; memory accumulates
**How to avoid:** Call `clear_mlx_memory()` after every `generate_image()` call
**Warning signs:** Memory usage grows with each generation; eventual crash

### Pitfall 2: Exposing Guidance Parameter
**What goes wrong:** User sets guidance > 0, gets unexpected results or errors
**Why it happens:** Z-Image Turbo model was trained without CFG guidance
**How to avoid:** Don't expose guidance in UI; hardcode to 0.0 internally
**Warning signs:** Generated images don't match prompt quality expectations

### Pitfall 3: Wrong Image Strength Semantics
**What goes wrong:** Confusion between "denoise" and "image_strength" naming
**Why it happens:** mflux uses `image_strength`, but ComfyUI convention is `denoise`
**How to avoid:** Per CONTEXT.md, label as `denoise` in UI but pass as `image_strength` to mflux
**Warning signs:** Users expect 1.0 to mean "full effect" but it means "ignore image completely"

### Pitfall 4: Skipping Temp File for img2img
**What goes wrong:** mflux expects file path, gets ComfyUI tensor
**Why it happens:** mflux `image_path` parameter needs actual file path
**How to avoid:** Use `_save_tensor_to_temp()` pattern from `Mflux_Pro.py` to write tensor to temp file
**Warning signs:** TypeError about expected str, got Tensor

### Pitfall 5: Model Cache Key Collision
**What goes wrong:** Wrong model used when switching quantization levels
**Why it happens:** `model_cache` in `Mflux_Core.py` uses tuple key; must include all relevant params
**How to avoid:** Ensure cache key includes model path AND quantization level
**Warning signs:** 4-bit model behavior when 8-bit selected (or vice versa)

### Pitfall 6: Non-Apple Silicon Registration
**What goes wrong:** Nodes appear but crash on Windows/Linux
**Why it happens:** mflux requires Apple Silicon; nodes shouldn't register elsewhere
**How to avoid:** Per CONTEXT.md, check platform before node registration; silent non-registration
**Warning signs:** Import errors on non-Mac systems

## Code Examples

Verified patterns from official sources and existing codebase:

### Z-Image Turbo Model Loading
```python
# Source: .reference/mflux/src/mflux/models/z_image/variants/turbo/z_image_turbo.py
from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
from mflux.models.common.config.model_config import ModelConfig

# Using factory method (preferred for known models)
model = ZImageTurbo(
    quantize=4,  # 4 for 4-bit, 8 for 8-bit, None for native
    model_path="/path/to/model/or/huggingface-id",
    model_config=ModelConfig.z_image_turbo(),
)

# Or with custom path resolution
model_config = ModelConfig.from_name("z-image-turbo", base_model="z-image-turbo")
model = ZImageTurbo(
    quantize=quantize_int,
    model_path=model_path,
    model_config=model_config,
)
```

### Z-Image Turbo Generation (txt2img)
```python
# Source: .reference/mflux/src/mflux/models/z_image/variants/turbo/z_image_turbo.py
# Note: generate_image returns GeneratedImage with .image property (PIL.Image)

result = model.generate_image(
    seed=42,
    prompt="A cinematic shot of a futuristic city",
    num_inference_steps=8,  # Default is 4, but 8 is common
    height=512,
    width=512,
    # Note: NO guidance parameter - Turbo uses 0.0 internally
)

# Extract PIL image
pil_image = result.image if hasattr(result, 'image') else result
```

### Z-Image Turbo Generation (img2img)
```python
# Source: .reference/mflux/src/mflux/models/z_image/variants/turbo/z_image_turbo.py
result = model.generate_image(
    seed=42,
    prompt="Transform to watercolor style",
    num_inference_steps=8,
    height=512,
    width=512,
    image_path="/path/to/input/image.png",  # Required for img2img
    image_strength=0.5,  # 0.0 = pure noise, 1.0 = pure image (inverted from typical denoise)
)
```

### Complete Sampler Node Pattern
```python
# Source: Composite from existing MfluxZImageNode and MfluxZImageInpaint patterns
from .utils.tensor_utils import pil_to_comfy_tensor, comfy_tensor_to_pil
from .utils.memory_utils import clear_mlx_memory

class MfluxZImageSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ZIMAGE_MODEL", {"tooltip": "Z-Image Turbo model from loader"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A cinematic shot..."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for generation"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50, "tooltip": "Inference steps"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "tooltip": "Output width"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "tooltip": "Output height"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "mflux"
    FUNCTION = "generate"

    def generate(self, model, prompt, seed, steps, width, height):
        try:
            result = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
            )

            pil_image = result.image if hasattr(result, 'image') else result
            image_tensor = pil_to_comfy_tensor(pil_image)
            return (image_tensor,)
        finally:
            clear_mlx_memory()
```

### img2img with Temp File Pattern
```python
# Source: Mflux_Pro.py _save_tensor_to_temp pattern
import os
import time
import uuid
from PIL import Image
import numpy as np
import folder_paths

def _save_tensor_to_temp(tensor, filename_prefix="z_img2img"):
    """Save ComfyUI IMAGE tensor to temp file for mflux."""
    in_dir = folder_paths.get_input_directory()
    fname = f"{filename_prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.png"
    out_path = os.path.join(in_dir, fname)

    array = tensor.cpu().numpy()
    if array.ndim == 4:
        array = array[0]  # Take first in batch
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(array, mode='RGB')
    img.save(out_path)
    return out_path

# In img2img node:
def generate(self, model, init_image, prompt, denoise, seed, steps):
    # Save ComfyUI tensor to temp file
    image_path = _save_tensor_to_temp(init_image)

    # Note: denoise in ComfyUI = image_strength in mflux
    # But semantics are inverted: denoise 0 = keep image, 1 = full noise
    # mflux image_strength: 0 = full noise, 1 = keep image
    # We map: mflux_strength = 1.0 - denoise
    image_strength = 1.0 - denoise

    result = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=steps,
        height=init_image.shape[1],  # [B,H,W,C]
        width=init_image.shape[2],
        image_path=image_path,
        image_strength=image_strength,
    )
    ...
```

### Node Registration with Platform Check
```python
# Source: CONTEXT.md decision - silent non-registration on non-Apple Silicon
import platform
import sys

# Check if Apple Silicon before registering nodes
def _is_apple_silicon():
    return sys.platform == "darwin" and platform.machine() == "arm64"

# In __init__.py
if _is_apple_silicon():
    from .Mflux_Comfy.Mflux_Air import (
        MfluxZImageLoader,
        MfluxZImageSampler,
        MfluxZImageImg2Img,
    )
    NODE_CLASS_MAPPINGS["MfluxZImageLoader"] = MfluxZImageLoader
    NODE_CLASS_MAPPINGS["MfluxZImageSampler"] = MfluxZImageSampler
    NODE_CLASS_MAPPINGS["MfluxZImageImg2Img"] = MfluxZImageImg2Img
    # ... display name mappings
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Combined loader/sampler node | Separate loader + sampler nodes | Phase 2 decision | Clearer UX, follows ComfyUI patterns |
| `guidance` parameter exposed | Guidance hardcoded to 0.0 | Always (Z-Image Turbo design) | Removes confusing parameter |
| All nodes on all platforms | Apple Silicon only registration | Phase 2 decision | Prevents crashes on unsupported systems |

**Deprecated/outdated:**
- `MfluxZImageNode`: Existing combined node - should be deprecated in favor of new separated nodes (but kept for backward compatibility per CONTEXT.md)

## Open Questions

Things that couldn't be fully resolved:

1. **Exact width/height step increments**
   - What we know: mflux Z-Image uses divisibility by 8 internally for latent sizing
   - What's unclear: Whether 64 step is better UX or if 8 step matches internals exactly
   - Recommendation: Use step=64 for UX consistency with other models; internally round to nearest 8 if needed

2. **Quantization dropdown values**
   - What we know: mflux supports 4-bit and 8-bit quantization, plus None for native
   - What's unclear: Whether 3, 5, 6-bit should be offered (they work but may not be optimal)
   - Recommendation: Offer "4", "8", "None" as per existing patterns; hide 3/5/6 as they're uncommon

3. **Model caching behavior with new loader**
   - What we know: Existing `model_cache` in Mflux_Core.py handles caching
   - What's unclear: Whether new loader should use same cache or separate cache
   - Recommendation: Use existing `model_cache` for consistency; ensure cache key includes all loader params

4. **Backward compatibility with MfluxZImageNode**
   - What we know: Existing workflows may use `MfluxZImageNode`
   - What's unclear: Whether to remove or deprecate
   - Recommendation: Keep `MfluxZImageNode` for compatibility; add new nodes alongside

## Sources

### Primary (HIGH confidence)
- `.reference/mflux/src/mflux/models/z_image/variants/turbo/z_image_turbo.py` - ZImageTurbo class API
- `.reference/mflux/src/mflux/models/z_image/z_image_initializer.py` - Model initialization pattern
- `.reference/mflux/src/mflux/models/common/config/model_config.py` - ModelConfig.z_image_turbo() factory
- `Mflux_Comfy/Mflux_Air.py` - Existing MfluxZImageNode and loader patterns
- `Mflux_Comfy/Mflux_Pro.py` - img2img pipeline and temp file patterns
- `Mflux_Comfy/utils/` - Phase 1 utilities (tensor_utils.py, memory_utils.py)

### Secondary (MEDIUM confidence)
- [ComfyUI Properties Documentation](https://docs.comfy.org/custom-nodes/backend/server_overview) - INPUT_TYPES, tooltips, CATEGORY patterns
- `.planning/phases/02-zimage-turbo/02-CONTEXT.md` - User decisions on node structure

### Tertiary (LOW confidence)
- Web searches for ComfyUI node patterns - verified against existing codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - verified from Phase 1, mflux reference code, existing codebase
- Architecture: HIGH - patterns taken from existing working nodes and CONTEXT.md decisions
- Pitfalls: HIGH - identified from reference code and existing implementation patterns

**Research date:** 2026-01-28
**Valid until:** 2026-02-28 (30 days - mflux API stable, ComfyUI patterns well-established)
