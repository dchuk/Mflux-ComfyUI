# Mflux-ComfyUI Fork

## What This Is

A fork of [joonsoome/Mflux-ComfyUI](https://github.com/joonsoome/Mflux-ComfyUI) updated to work with the latest mflux (0.15.5), bringing modern MLX-accelerated image generation to ComfyUI Desktop on Mac. Enables text-to-image with Z-Image Turbo and Flux2 Klein models, plus SeedVR2 diffusion-based upscaling.

## Core Value

Mac users can generate images with the latest mflux models (Z-Image Turbo, Flux2 Klein) and upscale them with SeedVR2 — all through ComfyUI's node interface, without leaving the Apple Silicon ecosystem.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Fork and set up development environment with mflux 0.15.5
- [ ] Incorporate PR #1 changes (multi-model architecture, node refactoring)
- [ ] Update mflux dependency from 0.13.1 → 0.15.5
- [ ] Z-Image Turbo text-to-image generation works in ComfyUI
- [ ] Flux2 Klein (4B/9B) text-to-image generation works in ComfyUI
- [ ] SeedVR2 upscaling node for image upscaling
- [ ] All nodes work with ComfyUI Desktop Mac
- [ ] Models load and run on M1 Pro with 32GB RAM (4-bit quantization)
- [ ] Progress bar and metadata saving work correctly

### Out of Scope

- CUDA/Windows support — this is Mac-only (MLX)
- Flux1 Fill/Depth/Redux nodes — not a priority (can add later if needed)
- LoRA finetuning — using pre-trained models only
- Qwen/FIBO models — lower priority, can add after core models work

## Context

**Upstream repos:**
- [joonsoome/Mflux-ComfyUI](https://github.com/joonsoome/Mflux-ComfyUI) — Original ComfyUI nodes, stuck at mflux 0.10.0
- [filipstrand/mflux](https://github.com/filipstrand/mflux) — The mflux library, currently at 0.15.5
- [PR #1](https://github.com/joonsoome/Mflux-ComfyUI/pull/1) — Updates to 0.13.1, adds Z-Image Turbo (good starting point)

**Key mflux changes since 0.10.0:**
- 0.13.0: Z-Image Turbo, FIBO, Qwen architectures
- 0.14.0: SeedVR2 upscaler, global VAE tiling, low-RAM mode
- 0.15.0: Flux2 Klein (4B/9B) with multi-image editing

**Target environment:**
- ComfyUI Desktop Mac (Apple Silicon)
- M1 Pro, 32GB RAM
- 4-bit quantized models preferred for memory efficiency

## Constraints

- **Platform**: macOS with Apple Silicon only (MLX framework)
- **Compatibility**: Must work with ComfyUI Desktop's node system
- **Memory**: Target 4-bit quantized models for 32GB RAM headroom
- **Dependencies**: mflux 0.15.5, mlx >=0.27.0, huggingface-hub >=1.1.6

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start from PR #1 as base | Already has multi-model node architecture and Z-Image support | — Pending |
| Target mflux 0.15.5 (not just 0.13.1) | Need Flux2 Klein and SeedVR2 which require 0.14.0+ | — Pending |
| Mac-only focus | MLX is the differentiator; CUDA users have other options | — Pending |

---
*Last updated: 2026-01-27 after initialization*
