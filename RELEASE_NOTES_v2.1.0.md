# Release Notes — v2.1.0

This release upgrades the backend to **mflux 0.13.1**, introducing a polymorphic loading architecture that supports **Z-Image Turbo**, **Qwen**, and **FIBO** models. It also introduces a dedicated **Optimizations node** for hardware management, a **Smart Model Loader** with visual status indicators, and improved stability.

## Highlights
- **Backend**: Upgraded to mflux 0.13.1 (requires macOS + Apple Silicon).
- **New Models**:
  - **Z-Image Turbo**: Support for the 6B parameter distilled model (optimized for speed).
  - **Qwen & FIBO**: Backend support for Qwen-Image and FIBO architectures.
- **New Nodes**:
  - **MFlux Z-Image Turbo**: Dedicated node with optimized defaults (9 steps, 0 guidance, 4-bit quantization).
  - **MFlux Optimizations**: A dedicated node to handle **Low RAM** (MemorySaver) and **VAE Tiling** settings, separating hardware constraints from generation parameters.
- **Smart Model Loader**: The `Mflux Models Loader` now recursively scans your local folders and checks your system cache. It uses visual indicators to show model availability:
  - 🟢 **Cached**: Found in HuggingFace system cache (ready to use).
  - 📁 **Local**: Found in `ComfyUI/models/Mflux`.
  - ☁️ **Alias**: Abstract shortcut (may trigger a download).
- **Unified Loading**: The `QuickMfluxNode` uses a single string input for models, seamlessly handling local paths, HuggingFace repo IDs, and aliases.
- **Robustness**: Added safety checks to prevent backend crashes when the prompt is empty.
- **ControlNet**: Fixed the preview behavior for the **Flux ControlNet Upscaler** (now correctly shows the original image instead of a Canny map).

## Breaking changes
- **Requirement**: Now requires `mflux==0.13.1` and `huggingface_hub>=0.26.0`.
- **UI Inputs**:
  - The `Local_model` input has been removed from the main node; use the unified `model` input instead.
  - `low_ram` and `vae_tiling` toggles have been moved to the new **MFlux Optimizations** node.
  - **Renamed Inputs**: `Mflux Models Downloader` input changed to `model`. `Mflux Models Loader` input changed from to `model`. Existing workflows using these nodes may need to be refreshed.
- **Z-Image**: Does not support Classifier-Free Guidance (guidance must be 0). Use the dedicated node to handle this automatically.

## Installation
- **ComfyUI-Manager**: Search “Mflux-ComfyUI” and install/update.
- **Manual**:
  1. cd /path/to/ComfyUI/custom_nodes
  2. git clone https://github.com/joonsoome/Mflux-ComfyUIMflux-ComfyUI.git
  3. Activate venv and install deps:
     - pip install --upgrade pip wheel setuptools
     - pip install 'mlx>=0.27.0' 'huggingface_hub>=0.26.0'
     - pip install 'mflux==0.13.1'
  4. Restart ComfyUI

## Usage notes
- **Z-Image Turbo**: Use the dedicated node. It defaults to the 4-bit quantized model (`filipstrand/Z-Image-Turbo-mflux-4bit`) and 9 steps. This will be downloaded the first time you use it and will be saved in: `User/.cache/huggingface/hub`. Press `Cmd + Shift + .` to unhide the .cache folder.
- **Optimizations**: To enable **Low RAM** mode or **VAE Tiling** (for large resolutions), add the `MFlux Optimizations` node and connect it to the `optimizations` input on the main node.
- **Qwen Models**: Use the new `negative_prompt` input on the QuickMfluxNode when using Qwen models.
- **Custom Paths**: When loading a custom model path or third-party repo, use the `base_model` dropdown as an **Architecture Hint** (e.g., select `qwen` if loading a Qwen model path).
- **Width/Height**: Should be multiples of 16.

## Known limitations
- ControlNet support is currently best‑effort (depends on backend build).
- Z-Image Turbo requires downloading ~6GB of weights (for 4-bit) on the first run.

## Thanks
- mflux by @filipstrand and contributors
- MFlux-ComfyUI 2.0.0 by @joonsoome
- MFlux-ComfyUI 1.2.5 by @raysers
- MFLUX-WEBUI by @CharafChnioune (Apache‑2.0 inspiration)