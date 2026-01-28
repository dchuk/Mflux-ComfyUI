# Requirements: Mflux-ComfyUI

**Defined:** 2026-01-27
**Core Value:** Mac users can generate images with mflux models through ComfyUI's node interface

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Infrastructure

- [x] **INFRA-01**: Tensor conversion utility converts PIL Image to ComfyUI `[B,H,W,C]` tensor
- [x] **INFRA-02**: Tensor conversion utility converts ComfyUI tensor to PIL Image
- [x] **INFRA-03**: MLX memory is cleared after each generation to prevent OOM
- [ ] **INFRA-04**: Package installs correctly via ComfyUI custom node manager
- [x] **INFRA-05**: All nodes appear in ComfyUI node browser under "mflux" category

### Z-Image Turbo

- [x] **ZIMG-01**: User can load Z-Image Turbo model with quantization selection (4-bit, 8-bit, none)
- [x] **ZIMG-02**: User can generate image from text prompt using loaded model
- [x] **ZIMG-03**: User can specify seed, steps, width, height for generation
- [x] **ZIMG-04**: User can use input image for img2img generation with strength control
- [x] **ZIMG-05**: Generated image outputs as standard ComfyUI IMAGE type

### SeedVR2 Upscaling

- [ ] **UPSC-01**: User can load SeedVR2 upscaling model with quantization selection
- [ ] **UPSC-02**: User can upscale ComfyUI IMAGE input (not file path)
- [ ] **UPSC-03**: User can specify scale factor (1x-4x) and seed
- [ ] **UPSC-04**: User can adjust softness parameter for output sharpness
- [ ] **UPSC-05**: Upscaled image outputs as standard ComfyUI IMAGE type

### Polish

- [ ] **PLSH-01**: Progress bar updates during multi-step generation
- [ ] **PLSH-02**: Generation parameters saved to PNG metadata
- [ ] **PLSH-03**: JSON sidecar file with full generation params (optional)
- [ ] **PLSH-04**: Nodes work correctly in ComfyUI Desktop Mac

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Flux2 Klein

- **FLX2-01**: User can load Flux2 Klein model (4B or 9B variant)
- **FLX2-02**: User can generate image from text prompt
- **FLX2-03**: User can use img2img with Flux2

### Additional Models

- **MODL-01**: Flux1 schnell/dev support
- **MODL-02**: LoRA adapter loading for supported models
- **MODL-03**: Qwen/FIBO model support

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| CUDA/Windows support | MLX is Apple Silicon only |
| Flux1 Fill/Depth/Redux | Advanced variants, not core use case |
| LoRA finetuning | Using pre-trained models only |
| Multi-image editing | Flux2 feature, add in v2 if needed |
| Controlnet integration | Complex, defer to future |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| INFRA-04 | Phase 4 | Pending |
| INFRA-05 | Phase 2 | Complete |
| ZIMG-01 | Phase 2 | Complete |
| ZIMG-02 | Phase 2 | Complete |
| ZIMG-03 | Phase 2 | Complete |
| ZIMG-04 | Phase 2 | Complete |
| ZIMG-05 | Phase 2 | Complete |
| UPSC-01 | Phase 3 | Pending |
| UPSC-02 | Phase 3 | Pending |
| UPSC-03 | Phase 3 | Pending |
| UPSC-04 | Phase 3 | Pending |
| UPSC-05 | Phase 3 | Pending |
| PLSH-01 | Phase 4 | Pending |
| PLSH-02 | Phase 4 | Pending |
| PLSH-03 | Phase 4 | Pending |
| PLSH-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-27*
*Last updated: 2026-01-28 after Phase 2 completion*
