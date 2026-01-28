---
status: complete
phase: 02-zimage-turbo
source: [02-01-SUMMARY.md]
started: 2026-01-28T09:15:00Z
updated: 2026-01-28T09:20:00Z
---

## Current Test

[testing complete]

## Tests

### 1. MfluxZImageLoader Node Structure
expected: MfluxZImageLoader class exists with model dropdown and quantize inputs (4/8/None options), returns ZIMAGE_MODEL
result: pass
verified: Automated code inspection - Mflux_ZImage.py lines 145-215, INPUT_TYPES has model and quantize with ["4", "8", "None"], RETURN_TYPES = ("ZIMAGE_MODEL",)

### 2. MfluxZImageSampler Node Structure
expected: MfluxZImageSampler class exists with model/prompt/seed/steps/width/height inputs, returns IMAGE
result: pass
verified: Automated code inspection - Mflux_ZImage.py lines 218-299, all required inputs present with correct types and ranges, RETURN_TYPES = ("IMAGE",)

### 3. MfluxZImageImg2Img Node Structure
expected: MfluxZImageImg2Img class exists with model/init_image/denoise/prompt/seed/steps inputs, returns IMAGE
result: pass
verified: Automated code inspection - Mflux_ZImage.py lines 302-400, init_image type ("IMAGE",), denoise FLOAT 0.0-1.0, RETURN_TYPES = ("IMAGE",)

### 4. Platform Gating
expected: _is_apple_silicon() function exists, nodes conditionally imported and registered only on Apple Silicon
result: pass
verified: Automated code inspection - __init__.py lines 1-6 defines _is_apple_silicon(), lines 29-41 and 72-83 conditionally import, lines 124-130 conditionally register

### 5. Legacy Node Backward Compatibility
expected: Legacy MfluxZImageNode remains registered for existing workflows
result: pass
verified: Automated code inspection - __init__.py line 87 has "MfluxZImageNode": MfluxZImageNode in NODE_CLASS_MAPPINGS

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0

## Gaps

[none]

---

## Human Verification Notes

The following items require runtime testing with ComfyUI and actual model files (cannot be verified by code inspection alone):

1. **Model loading works** - Load Z-Image Turbo model with 4-bit quantization on M1 Pro 32GB
2. **Text-to-image generation** - Enter prompt, generate image, appears in ComfyUI preview
3. **Img2img transformation** - Connect input image, apply denoise control, verify output
4. **Parameter effects** - Verify seed/steps/dimensions affect output as expected

These are runtime tests that require the full ComfyUI environment with mflux installed.
