# Project Research Summary

**Project:** mflux-comfyui (ComfyUI Custom Nodes for mflux)
**Domain:** ComfyUI custom node package / MLX image generation
**Researched:** 2026-01-27
**Confidence:** HIGH

## Executive Summary

This project wraps mflux 0.15.5 (Apple Silicon MLX-native image generation) as ComfyUI custom nodes. The domain is well-understood: mflux provides model classes (ZImageTurbo, Flux2Klein, SeedVR2) with straightforward `generate_image()` APIs that return PIL Images. ComfyUI expects PyTorch tensors in `[B,H,W,C]` format. The core challenge is bridging these two systems while managing model memory and respecting API constraints (e.g., Flux2Klein requires guidance=1.0, dimensions must be multiples of 8).

The recommended approach is **model-centric node organization**: separate loader nodes (heavy model instantiation) from generation nodes (lightweight execution). This prevents reloading 6GB+ models on every generation. Start with Z-Image Turbo as the reference implementation, then expand to Flux2 and SeedVR2. Memory management via `mlx.core.clear_cache()` is critical from day one.

Key risks center on API stability and memory management. mflux restructured imports between 0.10.0 and 0.15.0, and pre-quantized models from older versions are incompatible. These pitfalls are avoidable by pinning to 0.15.5+ and validating import paths before coding. The SeedVR2 upscaler Python API needs verification (documentation shows CLI usage only).

## Key Findings

### Recommended Stack

mflux 0.15.5 is MLX-native, meaning all model inference runs on Apple Silicon unified memory without PyTorch MPS. The stack is minimal: mflux brings in MLX, numpy, Pillow, and huggingface_hub as dependencies. ComfyUI requires PyTorch for tensor types, but PyTorch is not used for inference.

**Core technologies:**
- **mflux >=0.15.5**: Image generation engine - latest stable with Z-Image Turbo, Flux2 Klein, SeedVR2 support
- **MLX >=0.30.0**: Array framework - required by mflux, unified memory model on Apple Silicon
- **ComfyUI >=1.0.0**: Node execution platform - V1 node API is stable and well-documented
- **Python 3.10-3.12**: Runtime - mflux requires 3.10+, ComfyUI officially supports through 3.12

**Critical versions:**
- MLX < 0.27.0 causes UI warnings and performance issues
- mflux 0.13.0+ broke pre-quantized model compatibility
- 8-bit quantization required for LoRA support

### Expected Features

**Must have (table stakes):**
- Z-Image Turbo text-to-image generation (primary model, 6B params)
- Flux2 Klein text-to-image (fastest model, 4B/9B variants)
- SeedVR2 diffusion-based upscaling (3B params)
- Quantization support (4-bit, 8-bit) for memory management
- Proper tensor conversion (PIL to ComfyUI `[B,H,W,C]`)

**Should have (competitive):**
- Progress bar integration during multi-step generation
- Memory management (cache clearing, optional model unloading)
- Img2img support for Z-Image and Flux2

**Defer (v2+):**
- LoRA support (requires 8-bit quantization, API is complex)
- Fibo model (8B) and Qwen Image (20B) - large memory requirements
- Metadata embedding (convenience feature)
- ComfyUI V3 node API migration (still in preview)

### Architecture Approach

Model-centric organization with shared utilities. Each mflux model family (ZImage, Flux2, SeedVR2) gets dedicated loader and generation nodes. Custom `MFLUX_*_MODEL` types prevent accidental connection to incompatible PyTorch nodes. Tensor conversion happens only at node boundaries.

**Major components:**
1. **nodes/loaders.py** - Model loader nodes (instantiate mflux models with quantization)
2. **nodes/z_image.py, flux2.py, seedvr2.py** - Generation nodes per model family
3. **utils/tensor_convert.py** - PIL/MLX to PyTorch `[B,H,W,C]` conversion
4. **utils/config.py** - Model paths, quantization defaults, dimension constraints

### Critical Pitfalls

1. **Import path changes (0.10 to 0.15)** - mflux restructured imports completely. Use `from mflux.models.z_image import ZImageTurbo`, not legacy flat imports. Verify all imports against 0.15.5 source before coding.

2. **Pre-quantized model incompatibility** - Models quantized with mflux <0.13.0 fail to load. Document that users must re-quantize after upgrading. Add version detection with clear error messages.

3. **Tensor format mismatch** - ComfyUI expects `[B,H,W,C]` in `[0,1]` range. Always add batch dimension even for single images. Never squeeze batch dim.

4. **Memory not released** - MLX unified memory fills up. Call `mlx.core.clear_cache()` and `gc.collect()` after every generation. Implement from day one.

5. **Flux2 guidance must be 1.0** - Passing any other guidance value raises `ValueError`. Hardcode or hide this parameter for Flux2 nodes.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Foundation
**Rationale:** Core utilities must exist before any node can work. Tensor conversion is used by all generation nodes.
**Delivers:** Package structure, tensor conversion utility, config module, empty node registration
**Addresses:** Tensor format mismatch pitfall
**Avoids:** Building nodes without verified infrastructure

### Phase 2: Z-Image Turbo (First Working Node)
**Rationale:** Z-Image Turbo is the primary model with the simplest API (no guidance parameter, returns PIL directly). Validates entire architecture end-to-end.
**Delivers:** Z-Image loader node, Z-Image generation node, working txt2img pipeline
**Uses:** mflux ZImageTurbo class, tensor_convert utility
**Avoids:** Import path errors, memory leaks (implement cache clearing here)

### Phase 3: Flux2 Klein
**Rationale:** Builds on Phase 2 architecture. Different API constraints (guidance=1.0, returns GeneratedImage wrapper). Tests model-specific parameter handling.
**Delivers:** Flux2 loader (4B/9B variants), Flux2 generation node
**Implements:** Model-specific parameter validation

### Phase 4: SeedVR2 Upscaling
**Rationale:** Different workflow (takes image input, no prompt). Requires ComfyUI tensor to PIL conversion for input.
**Delivers:** SeedVR2 loader, upscale node
**Note:** Needs API verification - documentation shows CLI only

### Phase 5: Polish and Testing
**Rationale:** Core functionality complete. Focus on UX improvements and compatibility.
**Delivers:** Progress bar integration, ComfyUI Desktop testing, documentation

### Phase Ordering Rationale

- **Foundation first:** Tensor conversion and config are dependencies for all nodes
- **Z-Image before Flux2:** Simpler API validates architecture without model-specific edge cases
- **SeedVR2 last among features:** Different input/output pattern, may need API investigation
- **Memory management in Phase 2:** Implement cache clearing with first working node, not as afterthought

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (SeedVR2):** Verify Python API exists. All mflux docs show CLI usage (`mflux-upscale-seedvr2`). May need to call CLI or find undocumented class.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Well-documented ComfyUI node patterns and mflux import paths
- **Phase 2 (Z-Image):** API is verified from source code, straightforward implementation
- **Phase 3 (Flux2):** Similar to Z-Image, just different model config

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified against mflux 0.15.5 PyPI and source code |
| Features | HIGH | APIs verified from mflux source, ComfyUI docs authoritative |
| Architecture | HIGH | Based on official ComfyUI docs and reference implementations |
| Pitfalls | HIGH | Derived from mflux release notes and existing PR #1 |

**Overall confidence:** HIGH

### Gaps to Address

- **SeedVR2 Python API:** Verify `SeedVR2` class has `generate_image()` method or determine if CLI wrapper needed. Check `src/mflux/models/seedvr2/` in mflux source.

- **Progress callback integration:** mflux has callback system but exact registration API needs testing with ComfyUI's `PromptServer`.

- **ComfyUI Desktop compatibility:** Should test explicitly during Phase 5 - Desktop may lag behind development version.

## Sources

### Primary (HIGH confidence)
- [filipstrand/mflux](https://github.com/filipstrand/mflux) - mflux 0.15.5 source code
- [mflux PyPI](https://pypi.org/project/mflux/) - Version 0.15.5 verified
- [ComfyUI Custom Node Docs](https://docs.comfy.org/custom-nodes/walkthrough) - Official node patterns
- [ComfyUI Datatypes](https://docs.comfy.org/custom-nodes/backend/datatypes) - IMAGE tensor format
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/) - Memory management

### Secondary (MEDIUM confidence)
- [joonsoome/Mflux-ComfyUI](https://github.com/joonsoome/Mflux-ComfyUI) - Reference implementation
- [joonsoome/Mflux-ComfyUI PR #1](https://github.com/joonsoome/Mflux-ComfyUI/pull/1) - Multi-model architecture patterns
- [ComfyUI Registry Specs](https://docs.comfy.org/registry/specifications) - Publishing requirements

### Tertiary (LOW confidence)
- [ComfyUI V3 Spec](https://comfyui.org/en/comfyui-v3-dependency-resolution) - Preview, subject to change

---
*Research completed: 2026-01-27*
*Ready for roadmap: yes*
