# Phase 3: SeedVR2 Upscaling - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Diffusion-based image upscaling using SeedVR2 through ComfyUI nodes. Users can upscale any ComfyUI IMAGE input using loader/upscaler node pattern. Accepts images from any source, not just Z-Image.

</domain>

<decisions>
## Implementation Decisions

### Scale factor options
- Two modes via radio button selection:
  1. **Multiplier dropdown**: 1x, 2x, 4x discrete options
  2. **Longest side (px)**: Text input for target pixel dimension, scales proportionally
- Default mode: Multiplier with 4x selected
- Default longest side value: 2048px
- No max pixel limit — trust user to know hardware limits
- 1x option included for enhancement without resize
- Display calculated output dimensions (e.g., "512x512 → 2048x2048")

### Softness control
- Label: "Softness" (not sharpness or detail)
- Range: Match mflux API native range (discover during research)
- Default: 0 (sharpest/maximum detail)
- Control type: Numeric input (consistent with ComfyUI conventions)

### Node structure
- Loader/upscaler separation (consistent with Z-Image pattern)
- Load once, upscale multiple images
- Quantization options: Only show what mflux supports for SeedVR2
- Category: "mflux" (same as Z-Image — all mflux nodes together)
- Input type: Any ComfyUI IMAGE (not restricted to Z-Image output)

### Processing feedback
- Progress indicator/bar if mflux supports progress values
- Implement using ComfyUI best practices/common patterns
- No timeout — let long upscales run to completion
- Memory caching: User-configurable toggle, default OFF (clear after each)

### Claude's Discretion
- Cache toggle placement (loader vs upscaler node)
- Exact progress bar implementation details
- Platform gating approach (follow Z-Image pattern)
- Error handling for invalid scale values

</decisions>

<specifics>
## Specific Ideas

- Radio button UX for scale mode selection mirrors common image editor patterns
- Calculated output dimensions help users plan workflow without mental math
- Flexible IMAGE input makes node useful in diverse workflows (not locked to Z-Image)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-seedvr2-upscaling*
*Context gathered: 2026-01-28*
