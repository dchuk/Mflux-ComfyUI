<h1 align="center">Mflux-ComfyUI 2.1.0</h1>

<p align="center">
    <strong>ComfyUI nodes for mflux 0.13.1 (Apple Silicon/MLX)</strong><br/>
    <a href="README.md.kr">한국어</a> | <a href="README_zh.md">中文</a>
</p>

## Overview

- **Backend**: mflux 0.13.1 (requires macOS + Apple Silicon).
- **Graph compatibility**: Legacy inputs are migrated internally so your old graphs still work.
- **Unified Loading**: Seamlessly handles local paths, HuggingFace repo IDs, and predefined aliases (e.g., `dev`, `schnell`).

## What's New in mflux 0.13.1
This version brings significant backend enhancements:
- **Z-Image Turbo Support**: Support for the fast, distilled Z-Image variant optimized for speed (6B parameters).
- **FIBO & Qwen Support**: Backend support for FIBO and Qwen-Image architectures.
- **Smart Model Loader**: Visual indicators for cached vs. local models and recursive folder scanning.
- **Unified Architecture**: Improved resolution for models, LoRAs, and tokenizers.

## Key features

- **Core Generation**: Quick text2img and img2img in one node (`QuickMfluxNode`).
- **Z-Image Turbo**: Dedicated node for the new high-speed model (`MFlux Z-Image Turbo`).
- **Hardware Optimizations**: Dedicated node for **Low RAM** mode and **VAE Tiling** to prevent crashes on lower-memory Macs.
- **FLUX Tools Support**: Dedicated nodes for **Fill** (Inpainting), **Depth** (Structure guidance), and **Redux** (Image variation).
- **ControlNet**: Canny preview and best‑effort conditioning; includes support for the **Upscaler** ControlNet.
- **LoRA Support**: Unified LoRA pipeline (quantize must be 8 when applying LoRAs).
- **Quantization**: Rich options (None, 3, 4, 5, 6, 8-bit) for memory efficiency.
- **Metadata**: Saves full generation metadata (PNG + JSON) compatible with mflux CLI tools.

## Installation

### Using ComfyUI-Manager (Recommended)
- Search for “Mflux-ComfyUI” and install.

### Manual Installation
1. Navigate to your custom nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/joonsoome/Mflux-ComfyUI.git
   ```
3. Activate your ComfyUI virtual environment and install dependencies:
   ```bash
   # Example for standard venv
   source /path/to/ComfyUI/venv/bin/activate

   pip install --upgrade pip wheel setuptools
   pip install 'mlx>=0.27.0' 'huggingface_hub>=0.26.0'
   pip install 'mflux==0.13.1'
   ```
4. Restart ComfyUI.

**Note**: `mflux 0.13.1` requires `mlx >= 0.27.0`. If you are on an older version, please upgrade.

## Nodes

### MFlux/Air (Standard)
- **QuickMfluxNode**: The all-in-one node for standard FLUX txt2img, img2img, LoRA, and ControlNet.
- **MFlux Z-Image Turbo**: Dedicated node for Z-Image generation (optimized defaults: 9 steps, no guidance).
- **Mflux Optimizations**: Configure **Low RAM** (MemorySaver) and **VAE Tiling** settings here and connect to the main node.
- **Mflux Models Loader**: Smart selector for models. Recursively scans `models/Mflux` and checks your system cache.
  - 🟢 = Cached (Ready to use)
  - 📁 = Local (In your ComfyUI folder)
  - ☁️ = Alias (May trigger download)
- **Mflux Models Downloader**: Download quantized or full models directly from HuggingFace to your local folder.
- **Mflux Custom Models**: Compose and save custom quantized variants.

### MFlux/Pro (Advanced)
- **Mflux Fill**: FLUX.1-Fill support for inpainting and outpainting (requires mask).
- **Mflux Depth**: FLUX.1-Depth support for structure-guided generation.
- **Mflux Redux**: FLUX.1-Redux support for mixing image styles/structures.
- **Mflux Upscale**: Image upscaling using the Flux ControlNet Upscaler.
- **Mflux Img2Img / Loras / ControlNet**: Modular loaders for building custom pipelines.

## Usage Tips

- **Z-Image Turbo**: Use the dedicated node. It defaults to **9 steps** and **0 guidance** (required for this model).
- **Optimizations**: These settings (in the **Mflux Optimizations** node) trade speed for stability. If your generation works fine without them, keep them OFF for best performance.

  | Scenario | `low_ram` | `vae_tiling` | Why? |
  | :--- | :--- | :--- | :--- |
  | **Standard Use** (1024x1024, 4-bit model) | **OFF** | **OFF** | Fastest speed. Your Mac can handle it. |
  | **High Quality** (1024x1024, **8-bit or 16-bit** model) | **ON** | **OFF** | 16-bit models are ~24GB. You need Low RAM to fit them on most Macs. |
  | **High Res** (2048x2048 or Upscaling) | **OFF** | **ON** | Prevents the VAE from crashing at the end. |
  | **Potato Mode** (8GB RAM Mac, multitasking) | **ON** | **ON** | Maximum stability to prevent system freezes, at the cost of speed. |

- **LoRA Compatibility**: LoRAs currently require the base model to be loaded with `quantize=8` (or None).
- **Dimensions**: Width and Height should be multiples of 16 (automatically adjusted if needed).
- **Guidance**:
  - `dev` models respect guidance (default ~3.5).
  - `schnell` models ignore guidance (safe to leave as is).
- **Paths**:
  - Quantized models: `ComfyUI/models/Mflux`
  - LoRAs: `ComfyUI/models/loras` (create a `Mflux` subdirectory to keep them organized).
  - Automatically downloaded models from HuggingFace (like filipstrand/Z-Image-Turbo-mflux-4bit when using the Z-Image Turbo node for the first time): `User/.cache/huggingface/hub`, press `Cmd + Shift + .` to unhide the .cache folder.

## Workflows

Check the `workflows` folder for JSON examples:
- `Mflux text2img.json`
- `Mflux img2img.json`
- `Mflux ControlNet.json`
- `Mflux Fill/Redux/Depth` examples (if available)
The workflows for  Z-Image Turbo  are embedded in the png files in the `examples` folder:
- `Air_Z-Image-Turbo.png`
- `Air_Z-Image-Turbo_model_loader.png`
- `Air_Z-Image-Turbo_img2img_lora.png`

If nodes appear red in ComfyUI, use the Manager's "Install Missing Custom Nodes" feature.

## Acknowledgements

- **mflux** by [@filipstrand](https://github.com/filipstrand) and contributors.
- Original ComfyUI integration concepts by **raysers**.
- MFlux-ComfyUI 2.0.0 by **joonsoome**.
- Some code structure inspired by **MFLUX-WEBUI**.

## License

MIT
