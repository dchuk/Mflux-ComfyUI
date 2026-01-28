---
phase: 03-seedvr2-upscaling
plan: 01
subsystem: nodes
tags: [seedvr2, upscaling, diffusion, mlx, comfyui, super-resolution]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: tensor_utils, memory_utils, platform gating pattern
  - phase: 02-zimage-turbo
    provides: loader/sampler pattern, _is_apple_silicon() helper
provides:
  - MfluxSeedVR2Loader node (quantize, cache toggle)
  - MfluxSeedVR2Upscaler node (IMAGE input, scale modes, softness)
  - SEEDVR2_MODEL custom type for loader-to-upscaler connection
  - Platform-gated registration for Apple Silicon only
affects: [04-polish]

# Tech tracking
tech-stack:
  added: [mflux.models.seedvr2, mflux.utils.scale_factor]
  patterns: [loader-upscaler separation, temp file bridge for file-path APIs]

key-files:
  created: [Mflux_Comfy/Mflux_SeedVR2.py]
  modified: [__init__.py]

key-decisions:
  - "Temp file bridge for SeedVR2 API (requires file path, not tensor)"
  - "SEEDVR2_MODEL custom type for type-safe loader-upscaler connection"
  - "CATEGORY='mflux' for node browser organization (not subcategory)"
  - "Softness tooltip clarifies 0.0=sharpest, 1.0=maximum softness"

patterns-established:
  - "File-path API bridge: save tensor to temp, pass path, clean up in finally"
  - "Dimension calculation helper for resolution display string"

# Metrics
duration: 4min
completed: 2026-01-28
---

# Phase 3 Plan 1: SeedVR2 Upscaling Nodes Summary

**SeedVR2 diffusion upscaler nodes with multiplier/longest-side modes, softness control, and dimensions output string**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-28T09:10:00Z
- **Completed:** 2026-01-28T09:14:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- MfluxSeedVR2Loader node with quantize (4/8/None) and clear_cache toggle
- MfluxSeedVR2Upscaler accepting any ComfyUI IMAGE with scale modes and softness control
- Dimensions output string showing input->output transformation
- Platform gating ensures nodes only register on Apple Silicon

## Task Commits

Each task was committed atomically:

1. **Task 1: Create SeedVR2 nodes module** - `f35dda6` (feat)
2. **Task 2: Update node registration with platform gating** - `679750b` (feat)

## Files Created/Modified
- `Mflux_Comfy/Mflux_SeedVR2.py` - SeedVR2 loader and upscaler node classes (287 lines)
- `__init__.py` - Platform-gated node registration (+32 lines)

## Decisions Made
- **Temp file bridge pattern:** SeedVR2 API requires file path, not tensor - save to temp, process, clean up in finally block
- **SEEDVR2_MODEL custom type:** Type safety for loader-to-upscaler connection, prevents wrong model connections
- **Category "mflux":** All mflux nodes together per CONTEXT.md decision (not subcategory like "MFlux/SeedVR2")
- **Softness tooltip:** Explicit "0.0 = sharpest, 1.0 = maximum softness" to prevent user confusion

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- SeedVR2 upscaling nodes complete and registered
- Ready for Phase 4 (Polish) - error handling, documentation, and final testing
- All node types now implemented: generation (Z-Image), upscaling (SeedVR2)

---
*Phase: 03-seedvr2-upscaling*
*Completed: 2026-01-28*
