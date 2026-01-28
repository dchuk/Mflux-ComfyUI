# Domain Pitfalls

**Domain:** mflux ComfyUI custom nodes (Apple Silicon MLX)
**Researched:** 2026-01-27
**Confidence:** HIGH (verified against official releases, GitHub issues, and upstream PR #1)

---

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Import Path Changes Between mflux Versions

**What goes wrong:** Code imports from mflux 0.10.0 paths break silently or with cryptic errors when upgrading to 0.13+.

**Why it happens:** mflux 0.11.0 completely restructured the package from flat imports to a `models/` hierarchy. The 0.10.0 pattern `from mflux import Flux1, Config` became version-dependent. By 0.13.0, `Config` and `RuntimeConfig` were removed entirely, and parameters now pass directly to `generate_image()`.

**Consequences:**
- `ImportError` at ComfyUI startup
- Nodes fail to register, appear as red boxes
- Silent failures if old imports happen to exist but point to wrong classes

**Warning signs:**
- `ImportError: cannot import name 'Config' from 'mflux'`
- `ModuleNotFoundError: No module named 'mflux.flux'`
- Node class not appearing in ComfyUI node browser

**Prevention:**
1. Use the correct import paths for 0.15.x:
   - `from mflux.models.flux.variants.txt2img.flux import Flux1`
   - `from mflux.models.z_image import ZImageTurbo`
   - `from mflux.models.flux2 import Flux2`
   - `from mflux.models.seedvr2 import SeedVR2Upscaler` (verify exact path)
2. Remove `Config` class usage entirely - pass parameters directly
3. Create wrapper functions for model instantiation to isolate import changes

**Phase mapping:** Phase 1 (core infrastructure) - must be fixed before any node development

**Sources:**
- [mflux v0.11.0 Release Notes](https://github.com/filipstrand/mflux/releases) - Package restructure announcement
- [mflux v0.13.0 Release Notes](https://github.com/filipstrand/mflux/releases) - Config/RuntimeConfig removal

---

### Pitfall 2: Pre-Quantized Model Incompatibility

**What goes wrong:** Pre-quantized models saved with mflux <0.13.0 fail to load with newer versions.

**Why it happens:** mflux 0.13.0 made "internal changes which breaks compatibility with older pre-quantized models." The weight serialization format changed.

**Consequences:**
- Model loading crashes with weight shape mismatches
- Cryptic MLX errors about tensor dimensions
- Users with cached quantized models experience failures

**Warning signs:**
- Errors containing "weight shape mismatch" or "unexpected key"
- Loading works with full-precision models but fails with quantized
- Model loads on fresh install but fails on system with cached models

**Prevention:**
1. Document that users must re-quantize models using `mflux-save` after upgrading
2. Add version check at model load time with clear error message
3. Consider adding automatic model re-quantization migration or at minimum, clear instructions
4. In node UI, provide "Re-quantize Model" button or workflow

**Phase mapping:** Phase 2 (model loading nodes) - critical for first working prototype

**Sources:**
- [mflux v0.13.0 Release Notes](https://github.com/filipstrand/mflux/releases) - "breaks compatibility with older pre-quantized models"

---

### Pitfall 3: Tensor Format Mismatch (BHWC vs BCHW)

**What goes wrong:** Images from mflux (PIL/numpy) don't connect properly to ComfyUI's tensor pipeline.

**Why it happens:** ComfyUI standardizes on BHWC format (Batch, Height, Width, Channels) with values normalized to [0,1]. mflux returns PIL Images. Conversion errors are common, especially with batch dimension handling.

**Consequences:**
- Black/white/corrupted images
- `RuntimeError: shape mismatch` in downstream nodes
- Single images work but batches fail

**Warning signs:**
- Output image appears solid color or inverted
- Error messages about tensor dimensions
- Works in isolation but fails when connected to other nodes

**Prevention:**
1. Always convert mflux PIL output to ComfyUI tensor format explicitly:
   ```python
   import torch
   import numpy as np

   # PIL to ComfyUI IMAGE tensor
   img_np = np.array(pil_image).astype(np.float32) / 255.0
   img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add batch dim
   # Result: [1, H, W, 3] in [0,1] range
   ```
2. Never squeeze batch dimension even for single images
3. Add shape assertions at tensor boundaries
4. Test with batch sizes > 1 early in development

**Phase mapping:** Phase 1 (core infrastructure) - fundamental to all node output

**Sources:**
- [ComfyUI Tensor Documentation](https://docs.comfy.org/custom-nodes/backend/tensors) - Official format spec

---

### Pitfall 4: Memory Not Released After Generation

**What goes wrong:** VRAM/unified memory fills up after multiple generations, eventually crashing.

**Why it happens:** MLX uses unified memory with lazy evaluation. Python garbage collector doesn't automatically free MLX arrays. Model weights and intermediate tensors persist.

**Consequences:**
- OOM errors after several generations
- System slowdown as swap is used
- ComfyUI becomes unresponsive

**Warning signs:**
- Memory usage climbs with each generation (check Activity Monitor)
- First few generations work, later ones fail
- Restarting ComfyUI "fixes" the problem temporarily

**Prevention:**
1. Call `mlx.core.clear_cache()` after each generation
2. Implement explicit model unloading when switching models
3. Use `gc.collect()` after clearing MLX cache
4. Consider adding "Memory Saver" toggle that unloads model after each run
5. Monitor with `mlx.core.get_active_memory()` during development

**Phase mapping:** Phase 2 (generation nodes) - add memory management from the start

**Sources:**
- [MLX Unified Memory Documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [PR #1 Mflux-ComfyUI](https://github.com/joonsoome/Mflux-ComfyUI/pull/1) - Mentions cache clearing

---

## Moderate Pitfalls

Mistakes that cause delays or technical debt.

### Pitfall 5: Model-Specific Parameter Validation Missing

**What goes wrong:** Parameters valid for one model family crash another (e.g., guidance on schnell/z-image).

**Why it happens:** Different mflux model families have incompatible parameter sets:
- `schnell`: Ignores guidance, uses 2-4 steps
- `dev`: Requires guidance, uses 20-25 steps
- `Z-Image Turbo`: Hardcoded 0 guidance, 9 steps optimal
- `Flux2`: Different step counts, new parameters

**Consequences:**
- Generation fails with parameter errors
- Suboptimal results (e.g., too many steps on schnell)
- User confusion about which settings work

**Warning signs:**
- Error messages about unexpected keyword arguments
- Wildly different quality between model variants
- Community reports of "my settings don't work with model X"

**Prevention:**
1. Create model-specific node variants OR comprehensive parameter validation
2. Auto-detect model type and filter/clamp parameters
3. Show/hide UI parameters based on selected model
4. Document per-model recommended settings in node tooltips

**Phase mapping:** Phase 3 (model-specific nodes) - consider during node architecture

**Sources:**
- [mflux README](https://github.com/filipstrand/mflux/blob/main/README.md) - Model-specific defaults

---

### Pitfall 6: LoRA Path Resolution Failures

**What goes wrong:** LoRA loading fails with path errors or wrong adapter applied.

**Why it happens:** mflux 0.13.0 unified the LoRA API, removing `lora_names` and `lora_repo_id` parameters. The new `LoRALibrary.resolve_paths()` system works differently.

**Consequences:**
- LoRAs that worked in 0.10.0 don't load
- Silently loads wrong LoRA
- Path confusion between local files and HuggingFace repos

**Warning signs:**
- `FileNotFoundError` for LoRA paths
- Generation looks like base model despite LoRA being "loaded"
- Works with absolute paths, fails with relative

**Prevention:**
1. Use the unified `LoRALibrary.resolve_paths()` API
2. Support multiple path formats: local absolute, local relative to ComfyUI, HF repo ID
3. Validate LoRA compatibility with base model before loading
4. Display resolved path in node UI for debugging

**Phase mapping:** Phase 4 (LoRA support) - research API thoroughly before implementing

**Sources:**
- [mflux v0.13.0 Release Notes](https://github.com/filipstrand/mflux/releases) - LoRA API unification

---

### Pitfall 7: Dimension Constraints Not Enforced

**What goes wrong:** Users enter invalid dimensions, causing cryptic backend errors.

**Why it happens:** mflux requires width/height to be multiples of certain values (typically 8 or 16). ComfyUI doesn't automatically enforce this.

**Consequences:**
- Reshape errors in transformer layers
- "Tensor size mismatch" deep in generation code
- Hard-to-debug failures

**Warning signs:**
- Works at 1024x1024, fails at 1000x1000
- Error messages mentioning "reshape" or "view"

**Prevention:**
1. In `INPUT_TYPES`, use step=8 or step=16 for dimension inputs
2. Add validation that rounds to nearest valid size
3. Display warning if user enters non-compliant dimensions
4. Clamp dimensions in node execution before passing to mflux

**Phase mapping:** Phase 2 (generation nodes) - basic input validation

**Sources:**
- [Mflux-ComfyUI README](https://github.com/joonsoome/Mflux-ComfyUI) - "Width/Height must be multiples of 8"

---

### Pitfall 8: Empty Prompt Crashes

**What goes wrong:** Empty or whitespace-only prompts cause backend reshape errors.

**Why it happens:** Tokenizer produces empty sequences that break attention layer reshaping.

**Consequences:**
- Cryptic tensor shape errors
- Node appears broken when user hasn't entered prompt yet

**Warning signs:**
- Error contains "reshape" and mentions sequence length 0
- Works with any text, fails with empty string

**Prevention:**
1. Validate prompt is non-empty before generation
2. Fall back to safe default like `"."` (as PR #1 does)
3. Add `forceInput: true` to prevent empty prompt widget
4. Show clear error message if prompt validation fails

**Phase mapping:** Phase 2 (generation nodes) - basic input validation

**Sources:**
- [PR #1 Mflux-ComfyUI](https://github.com/joonsoome/Mflux-ComfyUI/pull/1) - Empty prompt fallback fix

---

### Pitfall 9: ComfyUI Version Incompatibility

**What goes wrong:** Nodes work in standalone ComfyUI but fail in ComfyUI Desktop.

**Why it happens:** ComfyUI Desktop may lag behind development versions. Frontend updates can break custom node UI. Python environment differences exist.

**Consequences:**
- Nodes don't appear in Desktop
- UI glitches or missing widgets
- Backend errors only in Desktop

**Warning signs:**
- Works on `comfyanonymous/ComfyUI` but not Desktop
- Frontend console errors
- UI elements render incorrectly

**Prevention:**
1. Test on ComfyUI Desktop explicitly (not just standalone)
2. Use stable/documented node APIs, avoid internal undocumented features
3. Check ComfyUI version at startup, warn if incompatible
4. Document minimum ComfyUI version requirement

**Phase mapping:** Phase 5 (testing/polish) - validate compatibility explicitly

**Sources:**
- [ComfyUI Desktop Installation Guide](https://comfyui-wiki.com/en/install/install-comfyui/comfyui-desktop-installation-guide) - Version lag warning

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 10: Missing Node Registration

**What goes wrong:** Node code is correct but doesn't appear in ComfyUI.

**Why it happens:** Forgot to add class to `NODE_CLASS_MAPPINGS` or `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`.

**Consequences:**
- Node invisible in node browser
- No error message - silent failure

**Warning signs:**
- Code is correct but node doesn't appear
- Other nodes from same package work

**Prevention:**
1. Add registration immediately when creating new node class
2. Use consistent naming pattern for all nodes
3. Add startup log message confirming node registration
4. Create checklist for adding new nodes

**Phase mapping:** All phases - procedural discipline

---

### Pitfall 11: Metadata Not Saving Correctly

**What goes wrong:** Generation parameters not saved to PNG/JSON metadata.

**Why it happens:** mflux 0.14.x added new metadata fields. ComfyUI has its own metadata system. Conflict between the two.

**Consequences:**
- Can't reproduce generations from saved images
- Workflow embedding fails
- Third-party tools can't read parameters

**Warning signs:**
- Missing fields when inspecting saved images
- Different metadata format than expected

**Prevention:**
1. Use mflux's built-in metadata saving where possible
2. Supplement with ComfyUI workflow embedding
3. Test metadata roundtrip: generate -> save -> load -> verify
4. Support both PNG embedded and sidecar JSON

**Phase mapping:** Phase 4 (polish) - secondary feature

**Sources:**
- [mflux v0.14.2 Release Notes](https://github.com/filipstrand/mflux/releases) - Enhanced IPTC/XMP metadata

---

### Pitfall 12: Quantization + LoRA Incompatibility

**What goes wrong:** LoRA doesn't work or produces artifacts with quantization < 8-bit.

**Why it happens:** LoRA arithmetic requires sufficient precision. Low-bit quantization causes numerical errors.

**Consequences:**
- LoRA effect invisible or corrupted
- Quality degradation specific to LoRA + quantize combo

**Warning signs:**
- LoRA works at quantize=8, fails at quantize=4
- Visual artifacts only when LoRA is enabled

**Prevention:**
1. Enforce quantize >= 8 when LoRA is enabled
2. Show warning in UI when user selects incompatible combo
3. Document limitation prominently

**Phase mapping:** Phase 4 (LoRA support) - known constraint

**Sources:**
- [Mflux-ComfyUI README](https://github.com/joonsoome/Mflux-ComfyUI) - "LoRA + quantize < 8 is not supported"

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Core Infrastructure | Import paths wrong | Verify all imports against 0.15.5 before coding |
| Model Loading | Quantized model incompatibility | Add version detection and clear re-quantize instructions |
| Generation Nodes | Tensor format mismatch | Create robust PIL-to-ComfyUI conversion utility |
| Generation Nodes | Memory leaks | Implement cache clearing from day one |
| Z-Image/Flux2 Nodes | Model-specific parameters | Create validation per model family |
| SeedVR2 Upscaling | API uncertainty | Verify Python API exists (vs CLI-only) |
| LoRA Support | Path resolution changes | Use LoRALibrary.resolve_paths() |
| Testing | Desktop incompatibility | Explicit Desktop testing phase |

---

## Domain-Specific Research Flags

These areas need deeper investigation before implementation:

1. **SeedVR2 Python API**: All documentation shows CLI usage (`mflux-upscale-seedvr2`). Need to verify if `SeedVR2Upscaler` class has a Python API for programmatic use, or if we need to shell out to CLI.

2. **Flux2 Python API**: New in 0.15.0 - verify exact import path and `generate_image()` signature for `Flux2` class.

3. **Low RAM Mode**: mflux 0.14.0 added auto-activating VAE tiling in low-RAM mode. Understand how to expose this in nodes.

4. **MLX Version Pinning**: mflux has historically pinned MLX versions to avoid compatibility issues. Verify current requirements don't conflict with other ComfyUI MLX nodes.

---

## Sources

- [mflux GitHub Releases](https://github.com/filipstrand/mflux/releases) - HIGH confidence
- [mflux README](https://github.com/filipstrand/mflux/blob/main/README.md) - HIGH confidence
- [Mflux-ComfyUI Original Repo](https://github.com/joonsoome/Mflux-ComfyUI) - HIGH confidence
- [PR #1 Mflux-ComfyUI](https://github.com/joonsoome/Mflux-ComfyUI/pull/1) - HIGH confidence
- [ComfyUI Custom Node Docs](https://docs.comfy.org/custom-nodes/backend/) - HIGH confidence
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/) - HIGH confidence
- [ComfyUI Desktop Guide](https://comfyui-wiki.com/en/install/install-comfyui/comfyui-desktop-installation-guide) - MEDIUM confidence
