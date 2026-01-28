---
phase: 02-zimage-turbo
plan: 01
subsystem: nodes
tags: [z-image-turbo, comfyui, mflux, image-generation, loader-sampler]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: Phase 1 utilities (tensor_utils, memory_utils)
provides:
  - MfluxZImageLoader node for loading Z-Image Turbo model
  - MfluxZImageSampler node for text-to-image generation
  - MfluxZImageImg2Img node for image transformation with denoise control
  - ZIMAGE_MODEL custom type for loader/sampler connection
  - Platform-gated node registration (Apple Silicon only)
affects: [03-seedvr2-upscaling, 04-polish]

# Tech tracking
tech-stack:
  added: []
  patterns: [loader-sampler-separation, platform-gated-registration]

key-files:
  created:
    - Mflux_Comfy/Mflux_ZImage.py
  modified:
    - __init__.py

key-decisions:
  - "Loader/sampler pattern: Separate nodes for model loading and generation"
  - "Platform gating: _is_apple_silicon() for silent non-registration"
  - "ZIMAGE_MODEL type: Custom type for loader-to-sampler connection"
  - "Denoise mapping: User-friendly 0-1 scale (0=keep, 1=ignore) mapped to image_strength internally"

patterns-established:
  - "Loader/sampler separation: Load once, generate multiple with different parameters"
  - "Platform check: _is_apple_silicon() returns True on Apple Silicon, False otherwise"
  - "Conditional import: _ZIMAGE_NODES_AVAILABLE flag controls node registration"

# Metrics
duration: 4min
completed: 2026-01-28
---

# Phase 2 Plan 1: Z-Image Turbo Nodes Summary

**Dedicated Z-Image Turbo loader and sampler nodes following ComfyUI's loader/sampler pattern with platform-gated registration for Apple Silicon only**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-28T09:03:40Z
- **Completed:** 2026-01-28T09:07:10Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created MfluxZImageLoader for model loading with 4-bit, 8-bit, or no quantization
- Created MfluxZImageSampler for text-to-image generation with seed/steps/dimensions
- Created MfluxZImageImg2Img for image transformation with denoise control
- Added platform detection for Apple Silicon-only node registration
- Preserved legacy MfluxZImageNode for backward compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Z-Image nodes module** - `907c602` (feat)
2. **Task 2: Update node registration with platform gating** - `94713a3` (feat)

## Files Created/Modified
- `Mflux_Comfy/Mflux_ZImage.py` - 400-line module with MfluxZImageLoader, MfluxZImageSampler, MfluxZImageImg2Img classes
- `__init__.py` - Added _is_apple_silicon() and conditional Z-Image node registration

## Decisions Made
- **Loader/sampler pattern:** Follows existing MFlux/Air, MFlux/Pro naming convention with MFlux/ZImage category
- **Platform gating:** Nodes silently don't register on non-Apple Silicon (per CONTEXT.md requirement)
- **Backward compatibility:** Legacy MfluxZImageNode remains for existing workflows
- **Denoise mapping:** User sees 0.0-1.0 where 0=keep original, 1=ignore original (mapped internally to image_strength = 1 - denoise)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - execution proceeded smoothly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Z-Image Turbo nodes ready for testing on Apple Silicon
- Loader returns ZIMAGE_MODEL type for sampler connection
- All nodes in MFlux/ZImage category
- Ready for Phase 3 (SeedVR2 Upscaling) or Phase 4 (Polish)

---
*Phase: 02-zimage-turbo*
*Completed: 2026-01-28*
