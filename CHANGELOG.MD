# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0]

### Added

*   **New Model Architectures (Backend):** Added support for non-Flux architectures via `mflux` 0.13.1.
    *   **Z-Image Turbo:** 6B parameter distilled model.
    *   **Qwen:** Qwen-Image architecture.
    *   **FIBO:** Fibo architecture.
*   **MFlux Z-Image Turbo Node:** A dedicated node for the Z-Image Turbo model.
    *   Includes hardcoded optimizations specific to this model: 9 steps, 0 guidance, and 4-bit quantization defaults.
*   **MFlux Optimizations Node:** A new node to decouple hardware constraints from generation parameters.
    *   **Low RAM:** Triggers the `mflux.callbacks.instances.memory_saver.MemorySaver` callback.
    *   **VAE Tiling:** Enables `enable_tiling` on the VAE decoder to prevent VRAM crashes on large resolutions.
    *   **VAE Tiling Split:** Options for `horizontal` or `vertical` splitting.
*   **QuickMfluxNode Inputs:**
    *   `negative_prompt`: Added string input (Required for Qwen models; ignored by Flux).
    *   `model_input`: Added input dock to allow direct chaining from Loader/Downloader nodes (overrides the text field).
    *   `optimizations`: Added input to accept the `MFLUX_OPTS` packet from the new Optimizations node.
*   **Metadata Fields:** The saved JSON metadata now includes:
    *   `negative_prompt_used`
    *   `vae_tiling`
    *   `vae_tiling_split`
    *   `base_model_hint`
    *   `model_alias`

### Changed

*   **Backend Architecture (Polymorphic Loading):**
    *   **Previous behavior:** The backend supported only the FLUX architecture via the `Flux1` class.
    *   **New behavior:** Implements a factory pattern (`load_or_create_model`) to dynamically instantiate distinct classes (`Flux1`, `FIBO`, `QwenImage`, `ZImageTurbo`, `Flux1Fill`, `Flux1Depth`, `Flux1Redux`) based on the `base_model` hint and model type.
*   **Mflux Models Loader:**
    *   **Smart Scanning:** Now recursively scans subdirectories in `models/Mflux` to find model roots.
    *   **Visual Indicators:** Added symbols to the dropdown to indicate source: 🟢 (Cached), 📁 (Local), ☁️ (Alias).
    *   **Labels:** Renamed input to `model` and output to `model_to_load`.
*   **Mflux Models Downloader:**
    *   **Labels:** Renamed input to `model` and output to `downloaded_model`.
*   **Dependencies:**
    *   Updated `mflux` from `0.10.0` to `0.13.1`.
    *   Updated `huggingface_hub` from `>=0.24` to `>=0.26.0`.
*   **QuickMfluxNode Model Input:**
    *   Changed `model` input from a **Dropdown** to a **String**.
    *   Now accepts: Known aliases (e.g., `schnell`, `dev`), local paths, or HuggingFace Repo IDs.
*   **Base Model Logic:**
    *   The `base_model` input is now strictly an **Architecture Hint**.
    *   Options: `dev`, `schnell`, `qwen`, `fibo`, `z-image-turbo`.
    *   Usage: Required only when loading custom paths or third-party repositories to tell the backend which class to instantiate.
*   **Mflux Models Downloader List:**
    *   Updated the internal list to point to repositories compatible with mflux 0.13.1. The full list of selectable models is now:
        *   `dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit`
        *   `dhairyashil/FLUX.1-dev-mflux-4bit`
        *   `filipstrand/FLUX.1-Krea-dev-mflux-4bit`
        *   `akx/FLUX.1-Kontext-dev-mflux-4bit`
        *   `filipstrand/Z-Image-Turbo-mflux-4bit`
        *   `briaai/Fibo-mlx-4bit`
        *   `briaai/Fibo-mlx-8bit`
        *   `filipstrand/Qwen-Image-mflux-6bit`
*   **Dimensions:** Recommended Width/Height steps changed to multiples of **16** (previously 8) to align with Z-Image requirements.

### Removed

*   **Legacy Inputs:** Removed `Local_model` input from `QuickMfluxNode` (functionality merged into the unified `model` string input).
*   **Inline Optimizations:** Removed `low_ram` and `vae_tiling` toggles directly from `QuickMfluxNode` (moved to the `MFlux Optimizations` node).

### Fixed

*   **Crash on Empty Prompt:** Added a safety check to `QuickMfluxNode` and `MfluxZImageNode`. If the prompt is empty, it falls back to `.` to prevent the backend error `[reshape] Cannot infer the shape of an empty array`.