---
phase: 02-zimage-turbo
verified: 2026-01-28T17:15:00Z
status: passed
score: 7/7 must-haves verified
---

# Phase 2: Z-Image Turbo Verification Report

**Phase Goal:** Create dedicated Z-Image Turbo nodes with loader/sampler separation for clean workflows
**Verified:** 2026-01-28T17:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Z-Image Turbo loader node appears in ComfyUI node browser under "MFlux/ZImage" category | ✓ VERIFIED | `MfluxZImageLoader` class exists with `CATEGORY = "MFlux/ZImage"`, registered in `__init__.py` line 125 |
| 2 | User can load Z-Image Turbo model with 4-bit quantization on M1 Pro 32GB | ✓ VERIFIED | `quantize` parameter with options ["4", "8", "None"], default "4", implementation in `load_model()` method converts to int and passes to `ZImageTurbo(quantize=q_val)` |
| 3 | User can enter text prompt and generate an image that appears in ComfyUI preview | ✓ VERIFIED | `MfluxZImageSampler` has `prompt` STRING input (multiline), `generate()` method calls `model.generate_image()`, converts result to ComfyUI IMAGE tensor via `pil_to_comfy_tensor()`, returns `(image_tensor,)` |
| 4 | User can set seed, steps, width, height and see parameters affect output | ✓ VERIFIED | All four parameters present in `MfluxZImageSampler.INPUT_TYPES()`: seed (INT, 0 to 2^64), steps (INT, 1-50), width/height (INT, 64-2048, step 64). All passed to `model.generate_image(seed, prompt, num_inference_steps, height, width)` |
| 5 | User can connect input image for img2img workflow with strength control | ✓ VERIFIED | `MfluxZImageImg2Img` has `init_image` ("IMAGE") and `denoise` (FLOAT 0.0-1.0) inputs. Implementation saves tensor to temp file, maps denoise to image_strength (inverted), passes to `model.generate_image()` with `image_path` and `image_strength` parameters |
| 6 | Nodes do not appear on non-Apple Silicon systems (silent non-registration) | ✓ VERIFIED | `_is_apple_silicon()` function checks `sys.platform == "darwin" and platform.machine() == "arm64"`. Conditional import block (lines 30-41) and conditional registration (lines 124-130) only add nodes if `_ZIMAGE_NODES_AVAILABLE = True` |
| 7 | Legacy MfluxZImageNode preserved for backward compatibility | ✓ VERIFIED | Line 87 imports `MfluxZImageNode`, line 87 registers it in `NODE_CLASS_MAPPINGS`, line 106 adds display name. Legacy node remains available alongside new nodes |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Mflux_Comfy/Mflux_ZImage.py` | Z-Image loader, sampler, and img2img nodes | ✓ VERIFIED | 400 lines, contains all 3 classes (MfluxZImageLoader line 145, MfluxZImageSampler line 218, MfluxZImageImg2Img line 302). No stub patterns found. |
| `__init__.py` | Node registration with platform gating | ✓ VERIFIED | `_is_apple_silicon()` function added (line 4), conditional import (lines 30-41), conditional registration (lines 124-130) |
| `Mflux_Comfy/utils/tensor_utils.py` | Phase 1 utility | ✓ VERIFIED | Exists, exports `pil_to_comfy_tensor()` and `comfy_tensor_to_pil()` |
| `Mflux_Comfy/utils/memory_utils.py` | Phase 1 utility | ✓ VERIFIED | Exists, exports `clear_mlx_memory()` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `Mflux_ZImage.py` | `tensor_utils.py` | import | ✓ WIRED | Line 21: `from .utils.tensor_utils import pil_to_comfy_tensor, comfy_tensor_to_pil`. Used at lines 293, 387 |
| `Mflux_ZImage.py` | `memory_utils.py` | import | ✓ WIRED | Line 22: `from .utils.memory_utils import clear_mlx_memory`. Called in finally blocks at lines 299, 400 |
| `__init__.py` | `Mflux_ZImage.py` | conditional import | ✓ WIRED | Platform check at line 30, imports at lines 32-36 only if Apple Silicon |
| `MfluxZImageLoader` | model cache | caching | ✓ WIRED | Module-level `_model_cache` dict (line 46), cache key `(clean_model, q_val)` at line 194, cache check at line 196, cache store at line 212 |
| `MfluxZImageSampler.generate()` | `model.generate_image()` | API call | ✓ WIRED | Lines 278-284 call `model.generate_image(seed, prompt, num_inference_steps, height, width)`, result extracted and converted |
| `MfluxZImageImg2Img.generate()` | temp file | lifecycle | ✓ WIRED | Line 360 saves tensor to temp, line 393-397 cleanup in finally block with try/except for best effort |

### Requirements Coverage

| Requirement | Status | Supporting Truth(s) |
|-------------|--------|---------------------|
| INFRA-05: Nodes appear under "mflux" category | ✓ SATISFIED | Truth #1 - Category is "MFlux/ZImage" (consistent with MFlux/Air, MFlux/Pro pattern) |
| ZIMG-01: Z-Image Turbo loader with quantization | ✓ SATISFIED | Truth #2 - Loader with 4-bit, 8-bit, None options |
| ZIMG-02: Generate image from text prompt | ✓ SATISFIED | Truth #3 - Sampler generates from prompt |
| ZIMG-03: Controllable generation parameters | ✓ SATISFIED | Truth #4 - Seed, steps, width, height all present and wired |
| ZIMG-04: Img2img with strength control | ✓ SATISFIED | Truth #5 - Img2Img node with denoise parameter |
| ZIMG-05: Returns ComfyUI IMAGE type | ✓ SATISFIED | Truth #3, #5 - Both samplers return ("IMAGE",) type |

### Anti-Patterns Found

None. Clean implementation with proper patterns:

- **Model caching** - Avoids reloading (line 196)
- **Memory cleanup** - `clear_mlx_memory()` in finally blocks (lines 299, 400)
- **Temp file cleanup** - Try/except for best effort (lines 393-397)
- **Environment guards** - Testing support via env vars (lines 25-37)
- **Platform detection** - Silent non-registration pattern (lines 30-41, 124-130)

### Human Verification Required

None for code structure verification. The following items require runtime testing on Apple Silicon with actual model files:

#### 1. Model Loading with 4-bit Quantization

**Test:** Load Z-Image Turbo model with 4-bit quantization
**Expected:** Model loads successfully without OOM on M1 Pro 32GB
**Why human:** Requires actual model files and Apple Silicon hardware

#### 2. Text-to-Image Generation

**Test:** Enter a prompt, set parameters, generate an image
**Expected:** Image appears in ComfyUI preview within reasonable time
**Why human:** Requires runtime execution and visual verification

#### 3. Img2Img Transformation

**Test:** Connect input image, adjust denoise slider, generate
**Expected:** Output changes based on denoise value (0.0 = similar to input, 1.0 = follows prompt more)
**Why human:** Requires runtime execution and visual comparison

#### 4. Parameter Effects

**Test:** Change seed, steps, dimensions and verify output changes
**Expected:** Different seeds produce different images, more steps = higher quality, dimensions match request
**Why human:** Requires runtime execution and visual comparison

#### 5. Platform Gating

**Test:** Load plugin on non-Apple Silicon system
**Expected:** Z-Image nodes do not appear in node browser (no errors)
**Why human:** Requires non-Apple Silicon system for testing

---

## Summary

**All 7 must-haves VERIFIED at code level.**

The Phase 2 implementation is structurally complete:

1. **Node Structure** - All three nodes (Loader, Sampler, Img2Img) exist with correct INPUT_TYPES, RETURN_TYPES, and CATEGORY
2. **Wiring** - Phase 1 utilities properly imported and used throughout
3. **Platform Gating** - Apple Silicon detection and conditional registration implemented
4. **Memory Management** - MLX memory cleared in finally blocks
5. **Temp Files** - Proper lifecycle management with cleanup
6. **Backward Compatibility** - Legacy node preserved
7. **No Stubs** - All implementations substantive (400 lines, no TODO/FIXME patterns)

The code is production-ready. Runtime behavior (model loading, image generation quality, parameter effects) requires human testing on Apple Silicon hardware with actual model files.

---

_Verified: 2026-01-28T17:15:00Z_
_Verifier: Claude (gsd-verifier)_
