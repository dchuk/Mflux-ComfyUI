# Architecture Patterns

**Domain:** ComfyUI custom nodes for mflux image generation
**Researched:** 2026-01-27
**Confidence:** HIGH (verified against official documentation and source code)

## Executive Summary

The mflux-comfyui architecture must bridge two systems: ComfyUI's node-based execution model (PyTorch tensors in `[B,H,W,C]` format) and mflux's MLX-native model classes. The key architectural decision is **model-centric node organization** where each mflux model class (ZImageTurbo, Flux2, SeedVR2) gets dedicated wrapper nodes, with shared utilities extracted to a common layer.

## Recommended Architecture

```
mflux_comfyui/
├── __init__.py              # NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
├── nodes/
│   ├── __init__.py
│   ├── z_image.py           # ZImageTurbo nodes (txt2img)
│   ├── flux2.py             # Flux2 Klein nodes (txt2img, editing)
│   ├── seedvr2.py           # SeedVR2 upscaling node
│   └── loaders.py           # Model loader/downloader nodes
├── utils/
│   ├── __init__.py
│   ├── tensor_convert.py    # MLX <-> PyTorch tensor conversion
│   ├── progress.py          # ComfyUI progress bar integration
│   ├── metadata.py          # Image metadata handling
│   └── config.py            # Quantization, paths, model configs
├── requirements.txt         # mflux>=0.15.5, mlx>=0.27.0
└── pyproject.toml           # Package metadata
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `__init__.py` | Node registration, ComfyUI discovery | ComfyUI core |
| `nodes/*.py` | Wrap mflux model APIs as ComfyUI nodes | mflux models, utils |
| `utils/tensor_convert.py` | Convert between MLX arrays and PyTorch tensors | All node files |
| `utils/progress.py` | Report generation progress to ComfyUI UI | ComfyUI PromptServer |
| `utils/config.py` | Model paths, quantization levels, defaults | Model loaders |

### Data Flow

```
User Workflow (ComfyUI)
        │
        ▼
┌─────────────────┐
│  Model Loader   │ ─── Quantized model from HuggingFace/local cache
│    Node         │
└────────┬────────┘
         │ MODEL (mflux instance)
         ▼
┌─────────────────┐
│  Generation     │ ─── Prompt, seed, dimensions
│    Node         │
│ (txt2img/edit)  │
└────────┬────────┘
         │ generates MLX array
         ▼
┌─────────────────┐
│ tensor_convert  │ ─── MLX array → PyTorch [B,H,W,C]
└────────┬────────┘
         │ IMAGE (torch.Tensor)
         ▼
┌─────────────────┐
│  SeedVR2        │ ─── Optional upscaling
│  Upscale Node   │     (MLX → MLX → PyTorch)
└────────┬────────┘
         │ IMAGE (torch.Tensor)
         ▼
    ComfyUI Output
    (Preview/Save)
```

## mflux 0.15.5 Model Class Hierarchy

Based on source code analysis ([filipstrand/mflux](https://github.com/filipstrand/mflux)):

```
src/mflux/models/
├── common/              # Base classes, shared utilities
├── common_models/       # Reusable transformer blocks, VAEs
├── flux/               # FLUX.1 (legacy)
│   └── variants/
│       └── txt2img/flux.py  # Flux1 class
├── flux2/              # FLUX.2 Klein (4B/9B)
├── z_image/            # Z-Image Turbo (6B)
├── seedvr2/            # SeedVR2 upscaler (3B)
├── fibo/               # FIBO (8B) - out of scope
├── qwen/               # Qwen Image (20B) - out of scope
└── depth_pro/          # Depth estimation - out of scope
```

### Model Import Patterns

```python
# Z-Image Turbo (PRIMARY - best all-rounder)
from mflux.models.z_image import ZImageTurbo
model = ZImageTurbo(quantize=4)
image = model.generate_image(
    prompt="...",
    seed=42,
    num_inference_steps=9,
    width=1024,
    height=1024,
)

# FLUX.2 Klein (SECONDARY - fastest, editing capable)
from mflux.models.flux2 import Flux2
# Similar API pattern

# SeedVR2 (UPSCALING - no prompt required)
from mflux.models.seedvr2 import SeedVR2
# Takes image input, returns upscaled image
```

### Common API Pattern

All mflux models share:
- `__init__(quantize=None, model_path=None, lora_paths=None, lora_scales=None)`
- `generate_image(prompt, seed, num_inference_steps, height, width, ...)` returns `GeneratedImage`
- `GeneratedImage.save(path)` or `.image` for raw data

## ComfyUI Node Patterns

### Required Node Structure

```python
class MfluxZImageTurbo:
    CATEGORY = "mflux/generation"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MFLUX_MODEL",),  # Custom type for mflux instance
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 50}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)  # Standard ComfyUI IMAGE type
    FUNCTION = "generate"

    def generate(self, model, prompt, seed, steps, width, height):
        # Call mflux
        result = model.generate_image(
            prompt=prompt,
            seed=seed,
            num_inference_steps=steps,
            width=width,
            height=height,
        )
        # Convert MLX array → PyTorch tensor [B,H,W,C]
        image_tensor = mlx_to_torch(result.image)
        return (image_tensor,)
```

### Node Registration

```python
# __init__.py
from .nodes.z_image import MfluxZImageTurbo, MfluxZImageLoader
from .nodes.flux2 import MfluxFlux2, MfluxFlux2Loader
from .nodes.seedvr2 import MfluxSeedVR2Upscale, MfluxSeedVR2Loader

NODE_CLASS_MAPPINGS = {
    "MfluxZImageTurbo": MfluxZImageTurbo,
    "MfluxZImageLoader": MfluxZImageLoader,
    "MfluxFlux2": MfluxFlux2,
    "MfluxFlux2Loader": MfluxFlux2Loader,
    "MfluxSeedVR2Upscale": MfluxSeedVR2Upscale,
    "MfluxSeedVR2Loader": MfluxSeedVR2Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MfluxZImageTurbo": "Z-Image Turbo Generate",
    "MfluxZImageLoader": "Z-Image Turbo Model Loader",
    # ...
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

## Patterns to Follow

### Pattern 1: Separate Loader from Generator

**What:** Split model loading (heavy operation) from generation (uses loaded model)
**When:** Always - enables model reuse across multiple generations
**Why:** Prevents reloading 6GB+ models on every generation

```
[Model Loader] → MODEL → [Generator Node] → IMAGE
                            ↑
                          PROMPT
```

### Pattern 2: Custom Model Type

**What:** Define `MFLUX_MODEL` as custom type, not standard ComfyUI `MODEL`
**When:** Always - mflux models are MLX-native, not PyTorch
**Why:** Prevents accidental connection to incompatible PyTorch nodes

### Pattern 3: Explicit Tensor Conversion

**What:** Convert MLX arrays to PyTorch tensors at node boundaries only
**When:** When outputting IMAGE type for ComfyUI compatibility
**Why:** Keep MLX operations MLX-native for performance; convert only when required

```python
def mlx_to_torch(mlx_array):
    """Convert MLX array to ComfyUI IMAGE format [B,H,W,C]"""
    import numpy as np
    import torch

    # MLX → NumPy → PyTorch
    np_array = np.array(mlx_array)

    # Ensure [B,H,W,C] format
    if np_array.ndim == 3:  # [H,W,C]
        np_array = np_array[np.newaxis, ...]  # Add batch dim

    return torch.from_numpy(np_array).float()
```

### Pattern 4: Progress Callback Integration

**What:** Report generation progress to ComfyUI UI
**When:** During multi-step generation (9 steps for Z-Image, 12-25 for Flux2)
**Why:** User feedback, cancellation support

```python
from server import PromptServer

def progress_callback(step, total):
    PromptServer.instance.send_sync(
        "progress",
        {"value": step, "max": total}
    )
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Node

**What:** Single node that loads model AND generates image
**Why bad:** Reloads 6GB+ model on every workflow execution
**Instead:** Separate loader node, pass model reference

### Anti-Pattern 2: PyTorch-MLX Mixing in Hot Path

**What:** Converting between PyTorch and MLX during generation loop
**Why bad:** Memory pressure, performance degradation
**Instead:** Keep generation fully MLX; convert only at boundaries

### Anti-Pattern 3: Hardcoded Model Paths

**What:** Embedding absolute paths in node code
**Why bad:** Breaks cross-machine compatibility
**Instead:** Use HuggingFace Hub caching (`~/.cache/huggingface/hub`)

### Anti-Pattern 4: Ignoring Quantization

**What:** Always loading full-precision models
**Why bad:** Exceeds 32GB RAM on M1 Pro for larger models
**Instead:** Default to 4-bit quantization; make configurable

## Build Order (Dependencies)

Based on component dependencies, build in this order:

### Phase 1: Foundation
1. **`utils/tensor_convert.py`** - MLX ↔ PyTorch conversion (no dependencies)
2. **`utils/config.py`** - Model paths, quantization defaults
3. **`__init__.py` skeleton** - Empty mappings, package setup

### Phase 2: First Working Node
4. **`nodes/loaders.py`** - Model loader nodes (depends on config)
5. **`nodes/z_image.py`** - Z-Image Turbo generation (depends on tensor_convert, loaders)
6. **Register in `__init__.py`** - Add Z-Image nodes to mappings

*Milestone: Z-Image Turbo txt2img works end-to-end*

### Phase 3: Model Expansion
7. **`nodes/flux2.py`** - Flux2 Klein generation
8. **`nodes/seedvr2.py`** - SeedVR2 upscaling (different API - no prompt)

### Phase 4: Polish
9. **`utils/progress.py`** - Progress bar integration
10. **`utils/metadata.py`** - PNG metadata embedding

### Dependency Graph

```
tensor_convert ─────────────────────┐
       │                            │
       ▼                            ▼
    config ─────► loaders ─────► z_image ─────► __init__.py
                    │              │                  ▲
                    ▼              ▼                  │
                 flux2         seedvr2               │
                    │              │                  │
                    └──────────────┴──────────────────┘
```

## Scalability Considerations

| Concern | Current (M1 Pro 32GB) | Future (M4 Max 128GB) |
|---------|----------------------|----------------------|
| Model loading | 4-bit quantization required | Full precision possible |
| Concurrent models | 1 model at a time | 2-3 models possible |
| Image resolution | 1024x1024 comfortable | 2048x2048+ possible |
| Batch generation | Not recommended | Batch sizes up to 4 |

## Sources

- [ComfyUI Custom Node Walkthrough](https://docs.comfy.org/custom-nodes/walkthrough) - Official node structure documentation
- [ComfyUI Datatypes](https://docs.comfy.org/custom-nodes/backend/datatypes) - IMAGE tensor format `[B,H,W,C]`
- [filipstrand/mflux](https://github.com/filipstrand/mflux) - mflux source code and model organization
- [mflux PyPI](https://pypi.org/project/mflux/) - Version 0.15.5 API patterns
- [joonsoome/Mflux-ComfyUI PR #1](https://github.com/joonsoome/Mflux-ComfyUI/pull/1) - Existing node architecture patterns
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) - Reference for multi-node package organization
- [A Guide to ComfyUI Custom Nodes](https://www.bentoml.com/blog/a-guide-to-comfyui-custom-nodes) - Best practices
