# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-27)

**Core value:** Mac users can generate images with mflux models through ComfyUI's node interface
**Current focus:** Phase 2 - Z-Image Turbo - COMPLETE

## Current Position

Phase: 2 of 4 (Z-Image Turbo) - COMPLETE
Plan: 1 of 1 in current phase
Status: Phase complete
Last activity: 2026-01-28 - Completed 02-01-PLAN.md

Progress: [####......] 40%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 4.5 min
- Total execution time: 9 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 1/1 | 5min | 5min |
| 2. Z-Image Turbo | 1/1 | 4min | 4min |
| 3. SeedVR2 Upscaling | 0/1 | - | - |
| 4. Polish | 0/2 | - | - |

**Recent Trend:**
- Last 5 plans: 01-01 (5min), 02-01 (4min)
- Trend: (improving)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

| Decision | Rationale | Phase |
|----------|-----------|-------|
| Conditional MLX import | Support non-Apple systems gracefully | 01-01 |
| Clear MLX cache after every generation | Prevent memory accumulation between runs | 01-01 |
| Fallback Config class | Graceful degradation when mflux imports fail | 01-01 |
| Loader/sampler separation | Clean workflow pattern, load once generate multiple | 02-01 |
| Platform gating via _is_apple_silicon() | Silent non-registration on non-Apple Silicon | 02-01 |
| ZIMAGE_MODEL custom type | Type safety for loader-to-sampler connection | 02-01 |
| Denoise 0-1 mapping | User-friendly (0=keep, 1=ignore) internally mapped to image_strength | 02-01 |

### Pending Todos

None.

### Blockers/Concerns

- **Phase 3 (SeedVR2):** API verification needed - mflux docs show CLI only, Python API may need discovery during planning

## Session Continuity

Last session: 2026-01-28T09:07:10Z
Stopped at: Completed 02-01-PLAN.md (Phase 2 complete)
Resume file: None

---
*State initialized: 2026-01-27*
*Last updated: 2026-01-28*
