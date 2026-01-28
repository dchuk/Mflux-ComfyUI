---
phase: 03-seedvr2-upscaling
verified: 2026-01-28T17:30:00Z
status: passed
score: 6/6 must-haves verified
---

# Phase 3: SeedVR2 Upscaling Verification Report

**Phase Goal:** Users can upscale images using SeedVR2 diffusion upscaler
**Verified:** 2026-01-28T17:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SeedVR2 loader node appears in ComfyUI node browser under mflux category | ✓ VERIFIED | `__init__.py` registers nodes conditionally under "mflux" category (lines 157-161) |
| 2 | User can connect ComfyUI IMAGE output to upscale node input | ✓ VERIFIED | `MfluxSeedVR2Upscaler.INPUT_TYPES()` accepts `("IMAGE",)` type (line 185) |
| 3 | User can select scale factor (1x, 2x, 4x) via multiplier or longest side mode | ✓ VERIFIED | Scale modes implemented: multiplier choices ["1x", "2x", "4x"] and longest_side INT (lines 188-200) |
| 4 | User can adjust softness parameter (0.0-1.0) | ✓ VERIFIED | Softness parameter defined as FLOAT (0.0-1.0, step 0.1) with clear tooltip (lines 202-208) |
| 5 | Upscaled image outputs as ComfyUI IMAGE type | ✓ VERIFIED | Returns `("IMAGE", "STRING")` tuple with `pil_to_comfy_tensor()` conversion (lines 218, 273) |
| 6 | Output dimensions string shows input->output size calculation | ✓ VERIFIED | Dimensions string formatted as "{input_w}x{input_h} -> {output_w}x{output_h}" (line 254) |

**Score:** 6/6 truths verified (100%)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Mflux_Comfy/Mflux_SeedVR2.py` | MfluxSeedVR2Loader and MfluxSeedVR2Upscaler classes | ✓ VERIFIED | 287 lines, both classes exist with full implementations |
| `__init__.py` | Platform-gated SeedVR2 node registration | ✓ VERIFIED | `_SEEDVR2_NODES_AVAILABLE` flag with conditional imports and registration |

**Artifact Verification Details:**

**Mflux_Comfy/Mflux_SeedVR2.py:**
- **Level 1 (Exists):** ✓ EXISTS (287 lines)
- **Level 2 (Substantive):**
  - Line count: 287 (exceeds min 150) ✓
  - Stub patterns: None found ✓
  - Exports: MfluxSeedVR2Loader, MfluxSeedVR2Upscaler classes defined ✓
  - Status: ✓ SUBSTANTIVE
- **Level 3 (Wired):**
  - Imported in `__init__.py` lines 46-47, 100-101 ✓
  - Registered in NODE_CLASS_MAPPINGS when `_SEEDVR2_NODES_AVAILABLE` (lines 158-159) ✓
  - Status: ✓ WIRED

**__init__.py:**
- **Level 1 (Exists):** ✓ EXISTS
- **Level 2 (Substantive):**
  - Contains `_SEEDVR2_NODES_AVAILABLE` flag (lines 49, 51, 53, 103, 105, 107) ✓
  - Contains conditional import blocks (lines 43-53, 97-107) ✓
  - No stub patterns ✓
  - Status: ✓ SUBSTANTIVE
- **Level 3 (Wired):**
  - NODE_CLASS_MAPPINGS updated conditionally (lines 158-159) ✓
  - NODE_DISPLAY_NAME_MAPPINGS updated (lines 160-161) ✓
  - Status: ✓ WIRED

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| Mflux_SeedVR2.py | mflux.models.seedvr2.SeedVR2 | Conditional import | ✓ WIRED | Line 33: `from mflux.models.seedvr2 import SeedVR2` |
| __init__.py | Mflux_SeedVR2.py | Platform-gated import | ✓ WIRED | Lines 45-48: Conditional import when `_is_apple_silicon()` |
| MfluxSeedVR2Upscaler | pil_to_comfy_tensor | Output conversion | ✓ WIRED | Line 19: imported, Line 273: used to convert result |
| MfluxSeedVR2Upscaler | SeedVR2.generate_image() | Upscaling call | ✓ WIRED | Lines 259-264: Full call with seed, image_path, resolution, softness |
| Temp file cleanup | Upscaler finally block | Resource management | ✓ WIRED | Lines 278-287: temp file removed and MLX memory cleared in finally |

**Detailed Link Verification:**

**1. Component → mflux API (SeedVR2 model):**
```python
# Line 259-264: Full API call with all parameters
result = seedvr2_model.generate_image(
    seed=seed,
    image_path=temp_path,  # Uses temp file bridge pattern
    resolution=resolution,  # ScaleFactor or int
    softness=softness,
)
```
Status: ✓ WIRED — Complete call with response handling (lines 267-273)

**2. IMAGE Tensor → File Path Bridge:**
```python
# Line 243: Save tensor to temp file
temp_path = _save_tensor_to_temp(image)

# Lines 43-71: Implementation converts ComfyUI tensor to PIL Image, saves as PNG
def _save_tensor_to_temp(tensor: torch.Tensor, prefix: str = "seedvr2_input") -> str:
    # ... full implementation with numpy conversion and PIL save
```
Status: ✓ WIRED — Proper tensor-to-file bridge for file-path API

**3. Result → ComfyUI Tensor:**
```python
# Lines 267-270: Extract PIL image from result
if hasattr(result, 'image'):
    pil_image = result.image
else:
    pil_image = result

# Line 273: Convert to ComfyUI tensor
output_tensor = pil_to_comfy_tensor(pil_image)
```
Status: ✓ WIRED — Handles both result formats and converts to ComfyUI tensor

**4. Resource Cleanup:**
```python
# Lines 277-287: finally block ensures cleanup
finally:
    # Clean up temp file
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass
    
    # Clear MLX memory based on user setting
    if should_clear:
        clear_mlx_memory()
```
Status: ✓ WIRED — Proper cleanup prevents disk bloat and memory leaks

### Requirements Coverage

| Requirement | Status | Supporting Truths |
|-------------|--------|-------------------|
| UPSC-01: Load SeedVR2 model with quantization | ✓ SATISFIED | Truth 1: Loader node exists |
| UPSC-02: Upscale ComfyUI IMAGE input | ✓ SATISFIED | Truth 2: Accepts IMAGE input |
| UPSC-03: Specify scale factor (1x-4x) and seed | ✓ SATISFIED | Truth 3: Scale modes implemented |
| UPSC-04: Adjust softness parameter | ✓ SATISFIED | Truth 4: Softness control exists |
| UPSC-05: Output as ComfyUI IMAGE type | ✓ SATISFIED | Truth 5: Returns IMAGE type |

**All 5 Phase 3 requirements satisfied.**

### Anti-Patterns Found

None detected. Scanned for:
- TODO/FIXME/XXX/HACK comments: None found ✓
- Placeholder content: None found ✓
- Empty returns (return null/undefined/{}): None found ✓
- Console.log-only implementations: None found ✓
- Hardcoded test values: None found ✓

**Code Quality Assessment:**
- Proper error handling with try/finally blocks ✓
- Environment variable guards for imports ✓
- Model caching with clear cache strategy ✓
- Temp file cleanup in finally block ✓
- Memory management via user-controlled clear_cache flag ✓
- Helpful tooltips and documentation ✓

### Human Verification Required

#### 1. ComfyUI Node Browser Appearance

**Test:** Open ComfyUI, right-click canvas, search for "SeedVR2"
**Expected:** 
  - "MFlux SeedVR2 Loader" appears under "mflux" category
  - "MFlux SeedVR2 Upscaler" appears under "mflux" category
**Why human:** Requires ComfyUI runtime, can't verify programmatically

#### 2. Image Upscaling Quality

**Test:**
1. Load an image in ComfyUI
2. Connect to MfluxSeedVR2Upscaler
3. Set multiplier to 4x, softness to 0.0
4. Generate and inspect output quality
**Expected:**
  - Output image is 4x larger than input
  - Image appears sharp (softness 0.0)
  - No visual artifacts or corruption
**Why human:** Visual quality requires human judgment

#### 3. Softness Parameter Effect

**Test:**
1. Upscale same image with softness=0.0
2. Upscale same image with softness=1.0
3. Compare outputs side-by-side
**Expected:**
  - softness=0.0 produces sharper edges
  - softness=1.0 produces softer, less crisp output
  - Visual difference is noticeable
**Why human:** Perceptual quality comparison

#### 4. Scale Mode Accuracy

**Test:**
1. Upscale 512x512 image with Multiplier 2x
2. Upscale same image with Longest Side 1024
3. Check dimensions output string
**Expected:**
  - Multiplier 2x: "512x512 -> 1024x1024"
  - Longest Side 1024: "512x512 -> 1024x1024"
  - Both modes produce same result for this input
**Why human:** Need to verify actual output dimensions in ComfyUI

#### 5. Cache Toggle Behavior

**Test:**
1. Enable "Keep cached" on loader
2. Run upscale multiple times
3. Observe memory usage and speed
**Expected:**
  - First run loads model (slower)
  - Subsequent runs reuse cached model (faster)
  - Memory usage remains high between runs
  - Disabling cache clears memory after each run
**Why human:** Performance and memory observation requires runtime monitoring

#### 6. Platform Gating (Non-Apple Silicon)

**Test:** On Linux/Windows or Intel Mac, try to import nodes
**Expected:**
  - Nodes do NOT appear in node browser
  - No import errors or crashes
  - ComfyUI loads successfully
**Why human:** Requires non-Apple Silicon hardware for testing

---

## Verification Summary

**Phase Goal:** Users can upscale images using SeedVR2 diffusion upscaler

**Achievement Status:** ✓ GOAL ACHIEVED

**Automated Verification:**
- All 6 observable truths verified ✓
- All 2 required artifacts verified at all 3 levels ✓
- All 5 key links verified as wired ✓
- All 5 requirements satisfied ✓
- Zero anti-patterns detected ✓
- Syntax valid for both files ✓

**Code Commits:**
- f35dda6: Created Mflux_SeedVR2.py with full implementation (287 lines)
- 679750b: Updated __init__.py with platform-gated registration (+32 lines)

**Pattern Adherence:**
- Follows Z-Image loader/sampler separation pattern ✓
- Uses temp file bridge for file-path APIs ✓
- Platform gating matches existing nodes ✓
- CATEGORY="mflux" for unified node organization ✓

**What Actually Exists:**
- ✓ Two fully-implemented node classes with substantive logic
- ✓ SEEDVR2_MODEL custom type for type-safe connections
- ✓ Scale factor support (multiplier 1x/2x/4x and longest side mode)
- ✓ Softness parameter (0.0-1.0) with clear tooltip
- ✓ Dimensions output string showing transformation
- ✓ Temp file cleanup in finally block
- ✓ User-controlled MLX memory management
- ✓ Platform gating prevents non-Apple Silicon failures
- ✓ Conditional registration in NODE_CLASS_MAPPINGS

**What Does NOT Exist:**
- No stub implementations ✓
- No placeholder content ✓
- No TODO/FIXME comments ✓
- No empty return statements ✓
- No console.log-only handlers ✓

**Human Verification:**
6 items require human testing (node visibility, visual quality, parameter effects, performance)

---

_Verified: 2026-01-28T17:30:00Z_
_Verifier: Claude (gsd-verifier)_
