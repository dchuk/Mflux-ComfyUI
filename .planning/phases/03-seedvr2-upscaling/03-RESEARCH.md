# Phase 3: SeedVR2 Upscaling - Research

**Researched:** 2026-01-28
**Domain:** mflux SeedVR2 diffusion upscaler integration, ComfyUI node patterns, image upscaling workflows
**Confidence:** HIGH

## Summary

This phase adds SeedVR2 diffusion-based image upscaling to the mflux-comfyui package. SeedVR2 is a fundamentally different model from the controlnet-based upscaler (jasperai/Flux.1-dev-Controlnet-Upscaler) currently used by `MfluxUpscale` - it's a dedicated super-resolution diffusion model using a NaDiT transformer architecture. The implementation will follow the established loader/sampler pattern from Phase 2's Z-Image nodes.

Key findings:
1. **SeedVR2 API is straightforward**: The `SeedVR2` class has a simple `generate_image()` method accepting `image_path`, `resolution`, `softness`, and `seed`
2. **Single-step process**: Unlike iterative diffusion, SeedVR2 uses exactly 1 inference step internally (hardcoded in the model)
3. **Resolution accepts two modes**: Either a target pixel value for the shortest edge OR a `ScaleFactor` object (e.g., `2x`, `4x`)
4. **Softness parameter**: Range is 0.0 to 1.0, where 0.0 = sharpest (no pre-blur) and 1.0 = maximum softness (factor 8x pre-blur)
5. **Quantization supported**: Like other mflux models, supports 4-bit and 8-bit quantization
6. **No prompt required**: SeedVR2 uses pre-computed text embeddings internally, no user prompt needed

**Primary recommendation:** Create two nodes (`MfluxSeedVR2Loader`, `MfluxSeedVR2Upscaler`) following the Z-Image pattern. The loader handles model/quantization/cache settings, the upscaler handles IMAGE input and scale/softness parameters.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mflux | 0.15.5+ | MLX-native image generation | Contains SeedVR2 model implementation |
| mlx | 0.30.x | Apple ML framework | Required by mflux for GPU compute |
| torch | 2.x | Tensor operations | ComfyUI's core tensor library for IMAGE type |
| Pillow | 10.x | Image I/O | Required by SeedVR2Util for image preprocessing |
| folder_paths | (ComfyUI) | Directory resolution | Standard ComfyUI path management |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.26+ | Array operations | Tensor conversion in utils |
| comfy.utils | (ComfyUI) | Progress bars | Not applicable for SeedVR2 (single-step) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| MfluxUpscale (controlnet) | SeedVR2 | SeedVR2 is diffusion-native upscaler, not controlnet-based; different quality characteristics |
| Single combined node | Loader + Upscaler | Loader/sampler pattern per established Phase 2 pattern and user decisions |

**Installation:**
```bash
# Already installed from Phase 1
pip install mflux>=0.15.5
```

## Architecture Patterns

### Recommended Node Structure

Based on CONTEXT.md decisions and Z-Image patterns from Phase 2:

```
Mflux_Comfy/
├── Mflux_SeedVR2.py     # New file for SeedVR2 nodes
├── Mflux_Core.py        # Shared utilities (exists)
└── utils/
    ├── tensor_utils.py  # PIL <-> ComfyUI conversion (exists)
    └── memory_utils.py  # MLX memory management (exists)
```

### Pattern 1: SeedVR2 Loader Node
**What:** Load SeedVR2 model with quantization and cache options
**When to use:** Once per workflow, before upscaling
**Example:**
```python
# Source: Pattern adapted from MfluxZImageLoader in Mflux_ZImage.py
class MfluxSeedVR2Loader:
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
                    "tooltip": "Whether to clear MLX cache after each upscale. OFF keeps model in memory for faster subsequent runs."
                }),
            }
        }

    RETURN_TYPES = ("SEEDVR2_MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "mflux"
    FUNCTION = "load_model"
```

### Pattern 2: SeedVR2 Upscaler Node
**What:** Accept IMAGE input, return upscaled IMAGE output
**When to use:** After loading model, connected to any IMAGE source
**Example:**
```python
# Source: Pattern from CONTEXT.md decisions and SeedVR2 CLI
class MfluxSeedVR2Upscaler:
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
```

### Pattern 3: Resolution Calculation Display
**What:** Show calculated output dimensions to user
**When to use:** Return alongside upscaled image
**Example:**
```python
# Source: CONTEXT.md decision - display calculated dimensions
def _format_dimensions(input_w, input_h, output_w, output_h) -> str:
    """Format dimension string like '512x512 -> 2048x2048'"""
    return f"{input_w}x{input_h} -> {output_w}x{output_h}"

# In upscale method:
dims_string = _format_dimensions(input_w, input_h, true_w, true_h)
return (output_tensor, dims_string)
```

### Anti-Patterns to Avoid
- **Using prompt parameter:** SeedVR2 uses pre-computed text embeddings, no user prompt accepted
- **Exposing steps parameter:** SeedVR2 hardcodes num_inference_steps=1, don't expose
- **Combining loader and upscaler:** User decided on loader/sampler separation (consistent with Z-Image)
- **Restricting IMAGE input to Z-Image:** Accept any ComfyUI IMAGE source
- **Progress bar for single step:** SeedVR2 is one step, no meaningful progress to show

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PIL to ComfyUI tensor | Custom conversion | `utils.tensor_utils.pil_to_comfy_tensor()` | Already implemented in Phase 1 |
| ComfyUI tensor to PIL | Custom conversion | `utils.tensor_utils.comfy_tensor_to_pil()` | Already implemented in Phase 1 |
| Tensor to temp file | Custom saving | `_save_tensor_to_temp()` pattern from Mflux_Pro.py | Already handles batch dimension, normalization |
| MLX memory cleanup | Manual cache clear | `utils.memory_utils.clear_mlx_memory()` | Already implemented with gc.collect() |
| Scale factor parsing | Custom parsing | `mflux.utils.scale_factor.ScaleFactor` | mflux provides this utility class |
| Image preprocessing | Custom resize/blur | `SeedVR2Util.preprocess_image()` | mflux handles the complex softness-based preprocessing |
| Color correction | Custom LAB matching | `SeedVR2Util.apply_color_correction()` | mflux applies wavelet reconstruction and histogram matching |
| Platform detection | Custom check | `_is_apple_silicon()` from Phase 2 | Already in __init__.py |

**Key insight:** SeedVR2 handles all the complex image processing (preprocessing, color correction) internally. Our nodes just need to bridge ComfyUI tensors to file paths.

## Common Pitfalls

### Pitfall 1: Passing ComfyUI Tensor Directly to SeedVR2
**What goes wrong:** TypeError - SeedVR2 expects file path, not tensor
**Why it happens:** `SeedVR2.generate_image()` takes `image_path` parameter (str/Path)
**How to avoid:** Save ComfyUI IMAGE tensor to temp file using `_save_tensor_to_temp()` pattern
**Warning signs:** `TypeError: expected str, got Tensor`

### Pitfall 2: Confusing Softness Direction
**What goes wrong:** User expects "softness" to add detail, gets blur instead
**Why it happens:** Softness 0.0 = sharpest, 1.0 = maximum pre-blur (factor 8x downscale-upscale)
**How to avoid:** Clear tooltip: "0.0 = sharpest/maximum detail, 1.0 = maximum softness"
**Warning signs:** User complaints about "softness making images blurry"

### Pitfall 3: Not Cleaning Up Temp Files
**What goes wrong:** Temp directory fills with upscale input images
**Why it happens:** Each upscale saves input to temp file; files accumulate
**How to avoid:** Clean up temp file in finally block after upscale completes
**Warning signs:** Growing disk usage in ComfyUI input directory

### Pitfall 4: Forgetting Memory Clear on Cache-Off Mode
**What goes wrong:** OOM when user selects "clear_cache_after_use = False"
**Why it happens:** Model stays in memory; subsequent operations hit memory limits
**How to avoid:** Still clear cache after generation, just keep model loaded; separate "model cache" from "compute cache"
**Warning signs:** Memory grows with each upscale even with caching off

### Pitfall 5: Wrong Resolution Calculation
**What goes wrong:** Output dimensions don't match expected scale
**Why it happens:** SeedVR2 scales based on shortest edge, rounds to multiple of 16, adds padding
**How to avoid:** Use exact pattern from SeedVR2Util.preprocess_image() for dimension calculation
**Warning signs:** 512x768 at 2x produces 1024x1534 instead of 1024x1536

### Pitfall 6: Non-Apple Silicon Registration
**What goes wrong:** Nodes appear but crash on Windows/Linux
**Why it happens:** mflux/mlx require Apple Silicon
**How to avoid:** Platform gating via `_is_apple_silicon()` in __init__.py
**Warning signs:** Import errors on non-Mac systems

### Pitfall 7: Large Upscale Memory Exhaustion
**What goes wrong:** OOM on 4x upscale of large images
**Why it happens:** SeedVR2 processes full resolution in memory
**How to avoid:** No max limit (per CONTEXT.md), but user should understand hardware limits
**Warning signs:** Crash on 4x upscale of 2048px images

## Code Examples

Verified patterns from official sources and existing codebase:

### SeedVR2 Model Loading
```python
# Source: .reference/mflux/src/mflux/models/seedvr2/variants/upscale/seedvr2.py
from mflux.models.seedvr2 import SeedVR2
from mflux.models.common.config import ModelConfig

# Load model with quantization
model = SeedVR2(
    quantize=4,  # 4 for 4-bit, 8 for 8-bit, None for native fp16
    model_path=None,  # Uses default HuggingFace repo: numz/SeedVR2_comfyUI
    model_config=ModelConfig.seedvr2_3b(),
)
```

### SeedVR2 Image Upscaling with Scale Factor
```python
# Source: .reference/mflux/src/mflux/models/seedvr2/variants/upscale/seedvr2.py
from mflux.utils.scale_factor import ScaleFactor

# Using scale factor multiplier (1x, 2x, 4x)
result = model.generate_image(
    seed=42,
    image_path="/path/to/input.png",
    resolution=ScaleFactor(value=4),  # 4x upscale
    softness=0.0,  # Maximum sharpness
)

# Using target resolution for shortest edge
result = model.generate_image(
    seed=42,
    image_path="/path/to/input.png",
    resolution=2048,  # Target 2048px for shortest edge
    softness=0.3,  # Some softness
)
```

### SeedVR2 Softness Parameter Details
```python
# Source: .reference/mflux/src/mflux/models/seedvr2/variants/upscale/seedvr2_util.py (lines 31-41)
# Softness maps from 0.0-1.0 to factor 1.0-8.0
# factor = 1.0 + (softness * 7.0)
#
# At factor > 1.0:
#   1. Image is downscaled by factor
#   2. Then upscaled back to target resolution
#   3. This creates pre-blur effect before diffusion
#
# Examples:
#   softness=0.0 -> factor=1.0 -> no pre-blur
#   softness=0.5 -> factor=4.5 -> moderate pre-blur
#   softness=1.0 -> factor=8.0 -> maximum pre-blur
```

### Resolution Calculation Pattern
```python
# Source: .reference/mflux/src/mflux/models/seedvr2/variants/upscale/seedvr2_util.py (lines 20-29)
from mflux.utils.scale_factor import ScaleFactor

def calculate_output_dimensions(input_w, input_h, resolution):
    """Calculate true output dimensions matching SeedVR2Util.preprocess_image()"""
    if isinstance(resolution, ScaleFactor):
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
```

### Complete Loader Node Pattern
```python
# Source: Composite from MfluxZImageLoader and SeedVR2 patterns
from .utils.memory_utils import clear_mlx_memory

# Model cache for SeedVR2
_seedvr2_cache = {}

class MfluxSeedVR2Loader:
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
        # Conditional import to support platform gating
        try:
            from mflux.models.seedvr2 import SeedVR2
            from mflux.models.common.config import ModelConfig
        except ImportError as e:
            raise ImportError(
                "mflux is not installed or SeedVR2 not available. "
                "Please ensure you have 'mflux>=0.15.5' installed."
            ) from e

        q_val = None if quantize == "None" else int(quantize)
        cache_key = ("seedvr2", q_val)

        if cache_key not in _seedvr2_cache:
            _seedvr2_cache.clear()  # Clear old entries
            print(f"[MFlux-SeedVR2] Loading model (Quantize: {quantize})")

            instance = SeedVR2(
                quantize=q_val,
                model_path=None,
                model_config=ModelConfig.seedvr2_3b(),
            )
            _seedvr2_cache[cache_key] = instance
            print("[MFlux-SeedVR2] Model loaded successfully.")

        # Return model and cache setting as tuple (or wrapper object)
        model_data = {
            "model": _seedvr2_cache[cache_key],
            "clear_cache": clear_cache_after_use,
        }
        return (model_data,)
```

### Complete Upscaler Node Pattern
```python
# Source: Composite from CONTEXT.md decisions, SeedVR2 API, and Z-Image patterns
import os
import time
import uuid
import numpy as np
import torch
from PIL import Image
import folder_paths
from .utils.tensor_utils import pil_to_comfy_tensor
from .utils.memory_utils import clear_mlx_memory

class MfluxSeedVR2Upscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SEEDVR2_MODEL", {"tooltip": "SeedVR2 model from loader"}),
                "image": ("IMAGE", {"tooltip": "ComfyUI IMAGE to upscale"}),
                "scale_mode": (["Multiplier", "Longest Side"], {"default": "Multiplier"}),
                "multiplier": (["1x", "2x", "4x"], {"default": "4x"}),
                "longest_side": ("INT", {"default": 2048, "min": 64, "max": 8192}),
                "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                             "tooltip": "0.0 = sharpest, 1.0 = maximum softness"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "dimensions",)
    CATEGORY = "mflux"
    FUNCTION = "upscale"

    def upscale(self, model, image, scale_mode, multiplier, longest_side, softness, seed):
        from mflux.utils.scale_factor import ScaleFactor

        model_data = model
        seedvr2_model = model_data["model"]
        should_clear = model_data["clear_cache"]

        temp_path = None
        try:
            # Get input dimensions from tensor [B, H, W, C]
            input_h, input_w = image.shape[1], image.shape[2]

            # Save tensor to temp file (SeedVR2 requires file path)
            temp_path = self._save_tensor_to_temp(image)

            # Determine resolution parameter
            if scale_mode == "Multiplier":
                mult_int = int(multiplier.replace("x", ""))
                resolution = ScaleFactor(value=mult_int)
            else:
                resolution = longest_side

            # Calculate output dimensions for display
            output_w, output_h = self._calculate_dimensions(input_w, input_h, resolution)
            dims_string = f"{input_w}x{input_h} -> {output_w}x{output_h}"

            print(f"[MFlux-SeedVR2] Upscaling: {dims_string}, softness={softness}")

            # Generate upscaled image
            result = seedvr2_model.generate_image(
                seed=seed,
                image_path=temp_path,
                resolution=resolution,
                softness=softness,
            )

            # Extract PIL image and convert to ComfyUI tensor
            pil_image = result.image if hasattr(result, 'image') else result
            output_tensor = pil_to_comfy_tensor(pil_image)

            return (output_tensor, dims_string)
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            # Clear MLX cache based on user setting
            if should_clear:
                clear_mlx_memory()

    def _save_tensor_to_temp(self, tensor):
        """Save ComfyUI IMAGE tensor to temp file for SeedVR2."""
        in_dir = folder_paths.get_input_directory()
        fname = f"seedvr2_input_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.png"
        out_path = os.path.join(in_dir, fname)

        array = tensor.cpu().numpy()
        if array.ndim == 4:
            array = array[0]
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(array, mode='RGB')
        img.save(out_path)
        return out_path

    def _calculate_dimensions(self, input_w, input_h, resolution):
        """Calculate output dimensions matching SeedVR2Util pattern."""
        from mflux.utils.scale_factor import ScaleFactor

        if isinstance(resolution, ScaleFactor):
            target_res = resolution.get_scaled_value(min(input_w, input_h))
        else:
            target_res = resolution

        scale = target_res / min(input_w, input_h)
        true_w = int(input_w * scale)
        true_h = int(input_h * scale)
        true_w = (true_w // 2) * 2
        true_h = (true_h // 2) * 2

        return true_w, true_h
```

### Node Registration with Platform Check
```python
# Source: Existing __init__.py pattern
import platform
import sys

def _is_apple_silicon():
    return sys.platform == "darwin" and platform.machine() == "arm64"

# In __init__.py, alongside Z-Image nodes:
if _is_apple_silicon():
    try:
        from .Mflux_Comfy.Mflux_SeedVR2 import (
            MfluxSeedVR2Loader,
            MfluxSeedVR2Upscaler,
        )
        _SEEDVR2_NODES_AVAILABLE = True
    except ImportError:
        _SEEDVR2_NODES_AVAILABLE = False
else:
    _SEEDVR2_NODES_AVAILABLE = False

# Add to NODE_CLASS_MAPPINGS if available
if _SEEDVR2_NODES_AVAILABLE:
    NODE_CLASS_MAPPINGS["MfluxSeedVR2Loader"] = MfluxSeedVR2Loader
    NODE_CLASS_MAPPINGS["MfluxSeedVR2Upscaler"] = MfluxSeedVR2Upscaler
    NODE_DISPLAY_NAME_MAPPINGS["MfluxSeedVR2Loader"] = "MFlux SeedVR2 Loader"
    NODE_DISPLAY_NAME_MAPPINGS["MfluxSeedVR2Upscaler"] = "MFlux SeedVR2 Upscaler"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ControlNet-based upscaling (MfluxUpscale) | Diffusion-native SeedVR2 | Phase 3 addition | Different quality characteristics, dedicated upscale model |
| Combined upscale node | Loader + Upscaler separation | Phase 3 decision | Consistent with Z-Image pattern, load once run multiple |
| No progress feedback | Single-step model (N/A) | SeedVR2 design | No progress bar needed (1 inference step) |

**Not deprecated:**
- `MfluxUpscale`: Existing controlnet-based upscaler stays available. SeedVR2 is an alternative approach, not a replacement.

## Open Questions

Things that couldn't be fully resolved:

1. **Radio button vs dropdown for scale mode**
   - What we know: CONTEXT.md specifies "radio button selection"
   - What's unclear: ComfyUI doesn't have native radio buttons; combo box is standard
   - Recommendation: Use combo/dropdown with two options; visually similar UX

2. **Exact 1x scale behavior**
   - What we know: CONTEXT.md says "1x option included for enhancement without resize"
   - What's unclear: Whether 1x with softness=0 produces identical output
   - Recommendation: Include 1x in multiplier options; serves as "enhancement only" mode

3. **Cache toggle placement**
   - What we know: CONTEXT.md lists as "Claude's discretion"
   - What's unclear: Loader vs upscaler placement
   - Recommendation: Loader node - cache is a model-lifecycle concern, not per-upscale

4. **Very large upscales (8K+)**
   - What we know: No max limit per CONTEXT.md
   - What's unclear: Whether MLX/hardware can handle 8K output
   - Recommendation: Allow any size; document that user is responsible for hardware limits

5. **Quantization options**
   - What we know: mflux defaults list [3, 4, 5, 6, 8] for quantization
   - What's unclear: Which specifically work well with SeedVR2
   - Recommendation: Offer "4", "8", "None" - consistent with Z-Image, most common choices

## Sources

### Primary (HIGH confidence)
- `.reference/mflux/src/mflux/models/seedvr2/variants/upscale/seedvr2.py` - SeedVR2 class and generate_image() API
- `.reference/mflux/src/mflux/models/seedvr2/variants/upscale/seedvr2_util.py` - preprocess_image(), softness factor mapping, dimension calculation
- `.reference/mflux/src/mflux/models/seedvr2/cli/seedvr2_upscale.py` - CLI usage patterns and parameter defaults
- `.reference/mflux/src/mflux/models/seedvr2/seedvr2_initializer.py` - Model initialization and quantization
- `.reference/mflux/src/mflux/models/common/config/model_config.py` - ModelConfig.seedvr2_3b() factory, model name
- `.reference/mflux/src/mflux/utils/scale_factor.py` - ScaleFactor class for resolution handling
- `.reference/mflux/src/mflux/cli/parser/parsers.py` - CLI argument definitions, softness range (0.0-1.0)
- `Mflux_Comfy/Mflux_ZImage.py` - Z-Image loader/sampler pattern to follow
- `Mflux_Comfy/utils/tensor_utils.py` - Phase 1 tensor conversion utilities
- `Mflux_Comfy/utils/memory_utils.py` - Phase 1 memory management utilities

### Secondary (MEDIUM confidence)
- `.planning/phases/03-seedvr2-upscaling/03-CONTEXT.md` - User decisions on UX, parameters
- `.planning/phases/02-zimage-turbo/02-RESEARCH.md` - Phase 2 patterns to follow
- `Mflux_Comfy/Mflux_Pro.py` - Existing MfluxUpscale node for reference (different approach)

### Tertiary (LOW confidence)
- None - all findings verified against mflux source code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - verified from mflux source code and existing codebase
- Architecture: HIGH - follows established Z-Image pattern from Phase 2
- Pitfalls: HIGH - derived from API analysis and existing node implementations
- Code examples: HIGH - extracted from actual mflux implementation

**Research date:** 2026-01-28
**Valid until:** 2026-02-28 (30 days - SeedVR2 API stable in mflux 0.15.5+)
