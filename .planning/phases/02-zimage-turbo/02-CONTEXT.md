# Phase 2: Z-Image Turbo - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix existing Z-Image Turbo nodes to work with mflux 0.15.5 API. Users can load the model, generate images from text prompts, and perform img2img transformations. Progress bars and metadata are Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Parameter organization
- Quantization options on loader node only (not sampler)
- Default image size: 512x512
- Default step count: 8 steps
- Default seed: fixed (42) for reproducibility

### Error presentation
- Follow standard ComfyUI node error patterns for all errors
- Model load failures, OOM, generation errors all use ComfyUI conventions
- Validation happens at runtime (not pre-queue)
- On non-Apple Silicon: nodes don't appear at all (silent non-registration)

### img2img workflow
- Separate nodes for txt2img and img2img (not combined)
- Image input labeled: `init_image`
- Strength parameter labeled: `denoise` (0.0-1.0)
- Default denoise: 0.5

### Node naming/category
- Category: `mflux` (top-level)
- Naming style: CamelCase (`MfluxZImageLoader`, `MfluxZImageSampler`, `MfluxZImageImg2Img`)
- Model source: ComfyUI convention - scan models folder, populate dropdown automatically
- All parameters have helpful tooltip descriptions

### Claude's Discretion
- Exact quantization dropdown options (based on mflux API)
- Width/height step increments and ranges
- Internal node class names vs display names
- Error message wording (following ComfyUI patterns)

</decisions>

<specifics>
## Specific Ideas

- User wants ComfyUI-native feel throughout - follow existing conventions rather than inventing new patterns
- Models loaded from proper ComfyUI model folder with automatic discovery
- Tooltips on every parameter for discoverability

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-zimage-turbo*
*Context gathered: 2026-01-28*
