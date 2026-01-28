# Feature Specifications

**Domain:** mflux ComfyUI custom nodes (Apple Silicon MLX)
**Researched:** 2026-01-27
**Confidence:** HIGH (verified against mflux 0.15.5 source code)

---

## Executive Summary

This document specifies the features needed for mflux-comfyui, mapping mflux 0.15.5 Python APIs to ComfyUI nodes. The three core features are: Z-Image Turbo text-to-image generation, Flux2 Klein text-to-image generation, and SeedVR2 diffusion-based upscaling. Each feature has a well-defined Python API with clear import paths, parameters, and return types.

---

## Feature 1: Z-Image Turbo Text-to-Image

**Priority:** HIGH (primary model)
**Confidence:** HIGH (verified from source)

### Description

Z-Image Turbo is a 6B parameter model optimized for fast, high-quality image generation. It uses a different architecture than Flux1/Flux2 and operates with no classifier-free guidance (guidance=0.0 hardcoded).

### mflux 0.15.5 API

**Import:**
```python
from mflux.models.z_image import ZImageTurbo
```

**Constructor:**
```python
model = ZImageTurbo(
    quantize: int | None = None,           # 3, 4, 5, 6, 8 or None for full precision
    model_path: str | None = None,         # Local path to pre-quantized model
    lora_paths: list[str] | None = None,   # LoRA adapter paths
    lora_scales: list[float] | None = None, # LoRA weights
)
```

**Generation:**
```python
image: PIL.Image.Image = model.generate_image(
    seed: int,                             # Required: random seed
    prompt: str,                           # Required: text prompt
    num_inference_steps: int = 4,          # Default: 4 (recommended: 4-12)
    height: int = 1024,                    # Must be multiple of 8
    width: int = 1024,                     # Must be multiple of 8
    image_path: Path | str | None = None,  # Optional: img2img input
    image_strength: float | None = None,   # Optional: img2img strength (0.0-1.0)
    scheduler: str = "linear",             # Default scheduler
)
```

**Return Type:** `PIL.Image.Image` - can be directly saved or converted to numpy

### ComfyUI Node Design

**Nodes Required:**
1. `MfluxZImageLoader` - Load/cache Z-Image Turbo model
2. `MfluxZImageGenerate` - Generate image from prompt

**Node Outputs:**
- Loader: `MFLUX_ZIMAGE_MODEL` (custom type to prevent cross-connection with PyTorch models)
- Generate: `IMAGE` (standard ComfyUI tensor `[B,H,W,C]`)

### Parameters for UI

| Parameter | Type | Default | Range | Notes |
|-----------|------|---------|-------|-------|
| `quantize` | dropdown | 4 | [None, 3, 4, 5, 6, 8] | 4-bit recommended for 32GB RAM |
| `seed` | INT | 0 | 0 to 2^32-1 | -1 for random |
| `prompt` | STRING | "" | multiline | Required, non-empty |
| `steps` | INT | 9 | 1-50 | 4-12 recommended |
| `width` | INT | 1024 | 512-2048, step=8 | Must be multiple of 8 |
| `height` | INT | 1024 | 512-2048, step=8 | Must be multiple of 8 |

### Memory Requirements

| Quantization | Model Size | Peak RAM Usage |
|--------------|------------|----------------|
| None (fp16) | ~12GB | ~18GB |
| 8-bit | ~6GB | ~10GB |
| 4-bit | ~3GB | ~6GB |

### Implementation Notes

- Model returns `PIL.Image.Image` directly, not `GeneratedImage` wrapper
- No guidance parameter (hardcoded to 0.0)
- Supports img2img via `image_path` and `image_strength` parameters
- Has callback system for progress reporting via `model.callbacks`

---

## Feature 2: Flux2 Klein Text-to-Image

**Priority:** HIGH (fastest model)
**Confidence:** HIGH (verified from source)

### Description

Flux2 Klein is available in 4B and 9B parameter variants. It uses a Qwen3 text encoder and has no CFG support (guidance must be 1.0). It's the fastest model in the mflux lineup.

### mflux 0.15.5 API

**Import:**
```python
from mflux.models.flux2 import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig
```

**Constructor:**
```python
model = Flux2Klein(
    quantize: int | None = None,
    model_path: str | None = None,
    lora_paths: list[str] | None = None,
    lora_scales: list[float] | None = None,
    model_config: ModelConfig | None = None,  # Determines 4B vs 9B
)

# For 4B model (default):
model = Flux2Klein()  # Uses ModelConfig.flux2_klein_4b()

# For 9B model:
model = Flux2Klein(model_config=ModelConfig.flux2_klein_9b())
```

**Available ModelConfig Options:**
- `ModelConfig.flux2_klein_4b()` - 4B parameter version (default)
- `ModelConfig.flux2_klein_9b()` - 9B parameter version

**Generation:**
```python
from mflux.utils.generated_image import GeneratedImage

result: GeneratedImage = model.generate_image(
    seed: int,                             # Required
    prompt: str,                           # Required
    num_inference_steps: int = 4,          # Default: 4 (recommended: 4-8)
    height: int = 1024,
    width: int = 1024,
    guidance: float = 1.0,                 # MUST be 1.0 (no CFG support)
    image_path: Path | str | None = None,  # Optional: img2img
    image_strength: float | None = None,
    scheduler: str = "flow_match_euler_discrete",
)

# Access the PIL image:
pil_image = result.image
```

**Return Type:** `GeneratedImage` - wrapper with `.image` property returning `PIL.Image.Image`

### ComfyUI Node Design

**Nodes Required:**
1. `MfluxFlux2Loader` - Load/cache Flux2 Klein model (with 4B/9B selection)
2. `MfluxFlux2Generate` - Generate image from prompt

### Parameters for UI

| Parameter | Type | Default | Range | Notes |
|-----------|------|---------|-------|-------|
| `variant` | dropdown | "4B" | ["4B", "9B"] | Model size selection |
| `quantize` | dropdown | 4 | [None, 3, 4, 5, 6, 8] | 4-bit for 32GB RAM |
| `seed` | INT | 0 | 0 to 2^32-1 | |
| `prompt` | STRING | "" | multiline | Required |
| `steps` | INT | 4 | 1-50 | 4-8 recommended |
| `width` | INT | 1024 | 512-2048, step=8 | |
| `height` | INT | 1024 | 512-2048, step=8 | |

### Memory Requirements

| Variant | Quantization | Model Size | Peak RAM Usage |
|---------|--------------|------------|----------------|
| 4B | 4-bit | ~2GB | ~5GB |
| 4B | 8-bit | ~4GB | ~8GB |
| 9B | 4-bit | ~4.5GB | ~8GB |
| 9B | 8-bit | ~9GB | ~14GB |

### Implementation Notes

- **CRITICAL:** `guidance` parameter MUST be 1.0, raises `ValueError` otherwise
- Returns `GeneratedImage` wrapper, access PIL image via `.image` property
- Uses Qwen3 text encoder (different from Z-Image/Flux1)
- Has callback system for progress reporting

---

## Feature 3: SeedVR2 Upscaling

**Priority:** HIGH (upscaling capability)
**Confidence:** HIGH (verified from source)

### Description

SeedVR2 is a 3B parameter diffusion-based upscaler. It takes an input image and produces a higher-resolution version. No text prompt is used - the model uses pre-computed text embeddings.

### mflux 0.15.5 API

**Import:**
```python
from mflux.models.seedvr2 import SeedVR2
from mflux.utils.scale_factor import ScaleFactor
```

**Constructor:**
```python
model = SeedVR2(
    quantize: int | None = None,
    model_path: str | None = None,
    # Note: NO LoRA support for SeedVR2
)
```

**Scale Factor:**
```python
# Can use ScaleFactor dataclass or integer
from mflux.utils.scale_factor import ScaleFactor

# As ScaleFactor object:
scale = ScaleFactor(value=2)  # 2x upscale
scale = ScaleFactor(value=1.5)  # 1.5x upscale

# Or parse from string:
scale = ScaleFactor.parse("2x")
scale = ScaleFactor.parse("1.5x")
```

**Generation:**
```python
from mflux.utils.generated_image import GeneratedImage

result: GeneratedImage = model.generate_image(
    seed: int,                                  # Required
    image_path: str | Path,                     # Required: input image
    resolution: int | ScaleFactor,              # Scale factor (e.g., 2 for 2x)
    softness: float = 0.0,                      # Sharpness control (0.0-1.0)
)

# Access the PIL image:
pil_image = result.image
```

**Return Type:** `GeneratedImage` - wrapper with `.image` property

### ComfyUI Node Design

**Nodes Required:**
1. `MfluxSeedVR2Loader` - Load/cache SeedVR2 model
2. `MfluxSeedVR2Upscale` - Upscale input image

**Node Inputs:**
- Upscale node takes `IMAGE` type (from other ComfyUI nodes) not file path
- Must convert ComfyUI tensor to PIL for mflux, then back to tensor

### Parameters for UI

| Parameter | Type | Default | Range | Notes |
|-----------|------|---------|-------|-------|
| `quantize` | dropdown | 4 | [None, 3, 4, 5, 6, 8] | Loader parameter |
| `seed` | INT | 0 | 0 to 2^32-1 | |
| `scale` | FLOAT | 2.0 | 1.0-4.0 | Upscale factor |
| `softness` | FLOAT | 0.0 | 0.0-1.0 | Higher = softer output |

### Memory Requirements

| Quantization | Model Size | Peak RAM (1024px input) | Peak RAM (2048px input) |
|--------------|------------|------------------------|------------------------|
| 4-bit | ~1.5GB | ~6GB | ~12GB |
| 8-bit | ~3GB | ~8GB | ~16GB |

### Implementation Notes

- **No prompt parameter** - uses pre-computed embeddings
- Input is file path, but ComfyUI uses tensors - need conversion layer
- `image_path` parameter is required (not optional like in generation models)
- Has VAE tiling for low-RAM mode (auto-activated)
- Applies color correction to match input image colors

---

## Feature 4: Progress Reporting

**Priority:** MEDIUM
**Confidence:** MEDIUM (callback system exists, ComfyUI integration needs testing)

### Description

Report generation progress to ComfyUI's progress bar during multi-step generation.

### mflux Callback System

All models have a `callbacks` attribute that manages progress reporting:

```python
# Models expose callbacks via:
model.callbacks.start(seed=seed, prompt=prompt, config=config)
# Returns context with: before_loop(), in_loop(t, latents), after_loop()
```

### ComfyUI Progress API

```python
from server import PromptServer

def report_progress(step: int, total: int):
    PromptServer.instance.send_sync(
        "progress",
        {"value": step, "max": total}
    )
```

### Implementation Approach

Two options:

**Option A: Custom Callback (Recommended)**
Create a callback class that implements `InLoopCallback` and reports to ComfyUI:

```python
from mflux.callbacks.callback import InLoopCallback

class ComfyUIProgressCallback:
    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        PromptServer.instance.send_sync(
            "progress",
            {"value": t + 1, "max": config.num_inference_steps}
        )
```

**Option B: Wrap Generation Loop**
Not recommended - would require modifying mflux models.

### Implementation Notes

- Need to register callback with model before generation
- Callback registration API needs investigation
- May need to check if `PromptServer.instance` is available in ComfyUI Desktop

---

## Feature 5: Metadata Saving

**Priority:** LOW (convenience feature)
**Confidence:** HIGH

### Description

Save generation parameters to output images for reproducibility.

### mflux Metadata Handling

mflux's `GeneratedImage.save()` method handles metadata:

```python
result = model.generate_image(...)
result.save(
    path="output.png",
    export_json_metadata=True,  # Creates .json sidecar file
)
```

The `GeneratedImage` class stores:
- seed, prompt, dimensions, steps
- quantization level
- LoRA paths and scales
- generation time

### ComfyUI Integration

ComfyUI has its own metadata system via `PngInfo`:

```python
from PIL.PngImagePlugin import PngInfo

metadata = PngInfo()
metadata.add_text("prompt", json.dumps(prompt_info))
metadata.add_text("workflow", json.dumps(workflow))
image.save(path, pnginfo=metadata)
```

### Implementation Approach

Combine both systems:
1. Let ComfyUI handle workflow embedding (automatic)
2. Add mflux-specific metadata (model, quantize, etc.) to PNG
3. Optionally create JSON sidecar with full generation parameters

---

## Feature 6: Memory Management

**Priority:** HIGH (critical for stability)
**Confidence:** HIGH

### Description

Properly release memory after generation to prevent OOM errors.

### MLX Memory Management

```python
import mlx.core as mx
import gc

# After generation:
mx.clear_cache()  # Release MLX memory pool
gc.collect()      # Python garbage collection
```

### Implementation Notes

- Call `mx.clear_cache()` after each generation
- Consider "Memory Saver" toggle that unloads model after each run
- Monitor with `mlx.core.get_active_memory()` during development
- Model caching must be carefully managed to avoid memory leaks

---

## Tensor Conversion Utilities

All nodes need to convert between mflux (PIL/MLX) and ComfyUI (PyTorch) formats.

### PIL to ComfyUI Tensor

```python
import numpy as np
import torch
from PIL import Image

def pil_to_comfyui(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor [B,H,W,C] in [0,1]."""
    np_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_array)
    return tensor.unsqueeze(0)  # Add batch dimension
```

### ComfyUI Tensor to PIL

```python
def comfyui_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor to PIL Image."""
    np_array = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_array)
```

### Batch Handling

```python
def comfyui_batch_to_pil_list(tensor: torch.Tensor) -> list[Image.Image]:
    """Convert batched ComfyUI tensor to list of PIL Images."""
    images = []
    for i in range(tensor.shape[0]):
        np_array = (tensor[i].cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(np_array))
    return images
```

---

## Node Summary

| Node | Category | Inputs | Outputs |
|------|----------|--------|---------|
| `MfluxZImageLoader` | mflux/loaders | quantize | MFLUX_ZIMAGE_MODEL |
| `MfluxZImageGenerate` | mflux/generation | model, prompt, seed, steps, w, h | IMAGE |
| `MfluxFlux2Loader` | mflux/loaders | variant, quantize | MFLUX_FLUX2_MODEL |
| `MfluxFlux2Generate` | mflux/generation | model, prompt, seed, steps, w, h | IMAGE |
| `MfluxSeedVR2Loader` | mflux/loaders | quantize | MFLUX_SEEDVR2_MODEL |
| `MfluxSeedVR2Upscale` | mflux/upscale | model, image, seed, scale, softness | IMAGE |

---

## Sources

- [filipstrand/mflux](https://github.com/filipstrand/mflux) - mflux 0.15.5 source code (HIGH confidence)
  - `src/mflux/models/z_image/variants/turbo/z_image_turbo.py`
  - `src/mflux/models/flux2/variants/txt2img/flux2_klein.py`
  - `src/mflux/models/seedvr2/variants/upscale/seedvr2.py`
  - `src/mflux/utils/scale_factor.py`
  - `src/mflux/callbacks/callback.py`
- [ComfyUI Datatypes](https://docs.comfy.org/custom-nodes/backend/datatypes) - IMAGE tensor format (HIGH confidence)
- [ComfyUI Progress API](https://docs.comfy.org/custom-nodes/backend) - PromptServer usage (MEDIUM confidence)
- [joonsoome/Mflux-ComfyUI](https://github.com/joonsoome/Mflux-ComfyUI) - Reference implementation patterns (MEDIUM confidence)
