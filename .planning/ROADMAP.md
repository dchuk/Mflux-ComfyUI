# Roadmap: Mflux-ComfyUI

## Overview

This roadmap delivers a ComfyUI custom node package that wraps mflux 0.15.5 for Apple Silicon image generation. We progress from foundational utilities (tensor conversion, memory management) through Z-Image Turbo text-to-image, then SeedVR2 upscaling, and finally polish features (progress bars, metadata). Each phase delivers a coherent, testable capability.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Foundation** - Package structure, tensor conversion, memory management
- [x] **Phase 2: Z-Image Turbo** - First working text-to-image generation
- [x] **Phase 3: SeedVR2 Upscaling** - Diffusion-based image upscaling
- [ ] **Phase 4: Polish** - Progress bars, metadata, packaging

## Phase Details

### Phase 1: Foundation
**Goal**: Update mflux 0.13.1 -> 0.15.5 and verify core utilities work
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03
**Success Criteria** (what must be TRUE):
  1. mflux 0.15.5 installs and imports without errors
  2. PIL Image converts to ComfyUI tensor with correct shape `[1, H, W, 3]` and range `[0, 1]`
  3. ComfyUI tensor converts back to PIL Image without data loss
  4. MLX cache clears successfully after calling memory utility
  5. Package imports without errors in Python 3.10+
**Plans:** 1 plan

Plans:
- [x] 01-01-PLAN.md - Update mflux dependency and create tensor/memory utilities

### Phase 2: Z-Image Turbo
**Goal**: Create dedicated Z-Image Turbo nodes with loader/sampler separation for clean workflows
**Depends on**: Phase 1
**Requirements**: INFRA-05, ZIMG-01, ZIMG-02, ZIMG-03, ZIMG-04, ZIMG-05
**Success Criteria** (what must be TRUE):
  1. Z-Image Turbo loader node appears in ComfyUI node browser under "mflux" category
  2. User can load Z-Image Turbo model with 4-bit quantization on M1 Pro 32GB
  3. User can enter text prompt and generate an image that appears in ComfyUI preview
  4. User can set seed, steps, width, height and see parameters affect output
  5. User can connect input image for img2img workflow with strength control
**Plans:** 1 plan

Plans:
- [x] 02-01-PLAN.md - Create Z-Image loader, sampler, and img2img nodes with platform gating

### Phase 3: SeedVR2 Upscaling
**Goal**: Users can upscale images using SeedVR2 diffusion upscaler
**Depends on**: Phase 2
**Requirements**: UPSC-01, UPSC-02, UPSC-03, UPSC-04, UPSC-05
**Success Criteria** (what must be TRUE):
  1. SeedVR2 loader node appears in ComfyUI node browser under "mflux" category
  2. User can connect ComfyUI IMAGE output to upscale node input (not file path)
  3. User can select scale factor (1x-4x) and see proportional output size
  4. User can adjust softness parameter and see visual difference in output sharpness
  5. Upscaled image outputs as ComfyUI IMAGE type connectable to other nodes
**Plans:** 1 plan

Plans:
- [x] 03-01-PLAN.md - Create SeedVR2 loader and upscaler nodes with platform gating

### Phase 4: Polish
**Goal**: Production-ready UX with progress feedback and metadata preservation
**Depends on**: Phase 3
**Requirements**: INFRA-04, PLSH-01, PLSH-02, PLSH-03, PLSH-04
**Success Criteria** (what must be TRUE):
  1. Package installs correctly via ComfyUI custom node manager
  2. Progress bar updates visibly during multi-step generation (not frozen UI)
  3. PNG metadata contains generation parameters viewable in image viewer
  4. All nodes work correctly in ComfyUI Desktop Mac (not just dev server)
**Plans**: TBD

Plans:
- [ ] 04-01: Progress integration and metadata saving
- [ ] 04-02: Packaging and ComfyUI Desktop testing

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 1/1 | ✓ Complete | 2026-01-28 |
| 2. Z-Image Turbo | 1/1 | ✓ Complete | 2026-01-28 |
| 3. SeedVR2 Upscaling | 1/1 | ✓ Complete | 2026-01-28 |
| 4. Polish | 0/2 | Not started | - |

---
*Roadmap created: 2026-01-27*
*Last updated: 2026-01-28*
