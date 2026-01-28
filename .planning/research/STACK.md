# Technology Stack

**Project:** mflux-comfyui (ComfyUI Custom Nodes for mflux)
**Researched:** 2026-01-27
**Overall Confidence:** HIGH

## Executive Summary

This stack integrates mflux 0.15.5 (MLX-native image generation) with ComfyUI's custom node system on Apple Silicon. The architecture follows ComfyUI's V1 node patterns (stable, widely adopted) while preparing for V3 migration. Key considerations: MLX's unified memory model, mflux's model-specific class hierarchy, and ComfyUI's tensor format conventions.

---

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| **mflux** | `>=0.15.5` | Image generation engine | Latest stable release (Jan 26, 2026) with Z-Image Turbo, Flux2 Klein, SeedVR2 support. All models implemented from scratch in MLX. | HIGH |
| **MLX** | `>=0.30.0` | Array framework | Required by mflux. Unified memory model eliminates CPU/GPU data transfers on Apple Silicon. Version 0.30+ has critical performance improvements for diffusion models. | HIGH |
| **ComfyUI** | `>=1.0.0` | Node execution platform | Target platform. V1 node API is stable with well-documented patterns. | HIGH |
| **Python** | `>=3.10, <3.15` | Runtime | mflux requires 3.10+. ComfyUI officially supports 3.10-3.12, with 3.13/3.14 working. | HIGH |

### Supporting Libraries

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| **torch** | `>=2.0.0` | Tensor operations | ComfyUI uses PyTorch tensors for IMAGE type. Required for tensor format conversion between mflux (MLX arrays) and ComfyUI. | HIGH |
| **numpy** | `>=1.24.0` | Array bridging | Intermediate format when converting MLX arrays to PyTorch tensors. | HIGH |
| **Pillow** | `>=10.0.0` | Image I/O | Loading/saving images, format conversion. mflux returns PIL Images directly. | HIGH |
| **huggingface_hub** | `>=0.20.0` | Model downloads | Model weight downloads from Hugging Face. Dependency of mflux. | HIGH |

### Development Tools

| Tool | Version | Purpose | When to Use | Confidence |
|------|---------|---------|-------------|------------|
| **comfy-cli** | latest | Project scaffolding | `comfy node scaffold` for initial setup, `comfy node publish` for registry deployment. | HIGH |
| **uv** | `>=0.5.0` | Package management | Recommended by mflux docs for installation. Fast, modern pip replacement. | MEDIUM |
| **pytest** | `>=7.0.0` | Testing | Unit tests for node logic. PR #1 includes pytest integration. | HIGH |
| **ruff** | `>=0.1.0` | Linting/formatting | Fast Python linter. Replaces flake8/black/isort. | MEDIUM |

---

## mflux 0.15.5 API Patterns

### Model Class Hierarchy

mflux organizes models by family under `mflux.models.*`:

```
mflux/models/
  z_image/          # ZImageTurbo, ZImage
  flux/             # Flux1 (schnell, dev)
  flux2/            # Flux2Klein
  seedvr2/          # SeedVR2 (upscaling)
  fibo/             # Fibo
  qwen/             # QwenImage
  depth_pro/        # DepthPro
```

### Primary API Patterns

**Z-Image Turbo (recommended for default generation):**
```python
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8)
image = model.generate_image(
    prompt="A puffin standing on a cliff",
    seed=42,
    num_inference_steps=9,
    width=1280,
    height=500,
)
image.save("output.png")  # Returns PIL.Image
```

**Flux1 (dev/schnell variants):**
```python
from mflux.models.flux.variants.txt2img.flux import Flux1

flux = Flux1.from_name(
    model_name="schnell",  # or "dev", "krea-dev"
    quantize=8,
)
image = flux.generate_image(
    seed=42,
    prompt="Luxury food photograph",
    num_inference_steps=2,  # schnell: 2-4, dev: 20-25
    height=1024,
    width=1024,
)
```

**SeedVR2 Upscaling:**
```python
# CLI: mflux-upscale-seedvr2 --image-path input.png --resolution 2x
# Python API follows similar pattern - check CLI entry points
```

### Key Configuration Options

| Parameter | Values | Notes |
|-----------|--------|-------|
| `quantize` | `3, 4, 5, 6, 8` | Bits for quantization. 8-bit recommended for balance. LoRA requires quantize=8. |
| `num_inference_steps` | varies | schnell: 2-4, dev: 20-25, z-image-turbo: 6-12 |
| `width/height` | multiples of 8 | Image dimensions must be divisible by 8 |
| `seed` | int | For reproducibility |

---

## ComfyUI Node Registration Requirements

### Required Class Attributes

```python
class MfluxGenerateNode:
    """Generate images using mflux models."""

    # REQUIRED: Menu category
    CATEGORY = "mflux"

    # REQUIRED: Input specification
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["z-image-turbo", "schnell", "dev"],),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 50}),
                "quantize": ([4, 8],),
            },
            "optional": {
                "image": ("IMAGE",),  # For img2img
            }
        }

    # REQUIRED: Output types (tuple)
    RETURN_TYPES = ("IMAGE",)

    # OPTIONAL: Human-readable output names
    RETURN_NAMES = ("image",)

    # REQUIRED: Method name to execute
    FUNCTION = "generate"

    # OPTIONAL: Mark as output node (always executes)
    OUTPUT_NODE = False

    def generate(self, prompt, model, width, height, seed, steps, quantize, image=None):
        # Implementation here
        # Must return tuple matching RETURN_TYPES
        return (output_tensor,)
```

### Tensor Format Conversion

ComfyUI uses PyTorch tensors with shape `[B, H, W, C]` (batch, height, width, channels) with values in range `[0.0, 1.0]`. mflux returns PIL Images.

```python
import numpy as np
import torch
from PIL import Image

def pil_to_comfyui_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor."""
    np_array = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_array)
    # Add batch dimension: [H, W, C] -> [1, H, W, C]
    return tensor.unsqueeze(0)

def comfyui_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor to PIL Image."""
    # Remove batch dimension and convert
    np_array = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_array)
```

### Node Registration Pattern

```python
# __init__.py at package root
from .nodes.generate import MfluxGenerateNode
from .nodes.upscale import MfluxUpscaleNode

NODE_CLASS_MAPPINGS = {
    "MfluxGenerate": MfluxGenerateNode,
    "MfluxUpscale": MfluxUpscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MfluxGenerate": "Mflux Generate",
    "MfluxUpscale": "Mflux Upscale (SeedVR2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

---

## pyproject.toml Configuration

```toml
[project]
name = "comfyui-mflux"
version = "1.0.0"
description = "ComfyUI nodes for mflux image generation on Apple Silicon"
license = { file = "LICENSE" }
requires-python = ">=3.10"
dependencies = [
    "mflux>=0.15.5",
]
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[project.urls]
Repository = "https://github.com/YOUR_USERNAME/mflux-comfyui"

[tool.comfy]
PublisherId = "YOUR_PUBLISHER_ID"  # From Comfy Registry
DisplayName = "Mflux for ComfyUI"
Icon = "https://example.com/icon.png"
requires-comfyui = ">=1.0.0"
```

### requirements.txt

```
mflux>=0.15.5
# Note: mflux brings in MLX, numpy, Pillow, huggingface_hub as dependencies
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| **ML Framework** | MLX | PyTorch MPS | mflux is MLX-native. PyTorch MPS would require rewriting models. |
| **Image Gen** | mflux | diffusers | mflux is optimized for Apple Silicon, smaller memory footprint. |
| **Node API** | V1 (current) | V3 (preview) | V3 is work-in-progress. V1 is stable, widely adopted, documented. |
| **Package Mgmt** | uv | pip | uv is faster, recommended by mflux docs, but pip works fine. |
| **Quantization** | 8-bit | 4-bit | 8-bit required for LoRA support, better quality/speed balance. |

---

## What NOT to Use

| Technology | Why Not |
|------------|---------|
| **diffusers library** | mflux reimplements models in MLX. Mixing frameworks wastes memory. |
| **PyTorch MPS backend** | MLX unified memory is more efficient on Apple Silicon. Don't mix. |
| **ComfyUI V3 node API** | Still in preview. Will require migration later but not ready for production. |
| **Global model singletons** | ComfyUI may run nodes in parallel. Use node-local or properly managed instances. |
| **MLX < 0.27.0** | PR #1 shows UI warnings for older versions. Critical performance improvements in 0.30+. |

---

## Platform Constraints

| Constraint | Details |
|------------|---------|
| **macOS only** | MLX and mflux are Apple Silicon exclusive. No Windows/Linux support. |
| **macOS >= 14.0** | Required for MLX Metal support. |
| **Apple Silicon** | M1, M2, M3, M4, M5 series. Intel Macs not supported. |
| **Memory** | 32GB recommended for larger models (Qwen 20B). 16GB works for Z-Image/Flux1. |

---

## Model Storage Paths

| Model Type | Path | Notes |
|------------|------|-------|
| Quantized models | `ComfyUI/models/Mflux/` | Local cache for quantized weights |
| Full models | `~/Library/Caches/mflux/` | Hugging Face cache |
| LoRA adapters | `ComfyUI/models/loras/` | Standard ComfyUI LoRA location |

---

## Sources

- [mflux GitHub Repository](https://github.com/filipstrand/mflux) - HIGH confidence
- [mflux PyPI](https://pypi.org/project/mflux/) - HIGH confidence (version 0.15.5 verified)
- [ComfyUI Official Docs](https://docs.comfy.org/custom-nodes/walkthrough) - HIGH confidence
- [ComfyUI Registry Specifications](https://docs.comfy.org/registry/specifications) - HIGH confidence
- [MLX GitHub Repository](https://github.com/ml-explore/mlx) - HIGH confidence
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/install.html) - HIGH confidence
- [joonsoome/Mflux-ComfyUI](https://github.com/joonsoome/Mflux-ComfyUI) - MEDIUM confidence (reference implementation)
- [joonsoome/Mflux-ComfyUI PR #1](https://github.com/joonsoome/Mflux-ComfyUI/pull/1) - MEDIUM confidence (multi-model architecture)
- [ComfyUI V3 Specification](https://comfyui.org/en/comfyui-v3-dependency-resolution) - MEDIUM confidence (preview, subject to change)
