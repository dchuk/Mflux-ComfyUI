Here is the updated `README_zh.md` file, translated to match the content of the current English `README.md`.

```markdown
<h1 align="center">Mflux-ComfyUI 2.1.0</h1>

<p align="center">
    <strong>适配 mflux 0.13.1 的 ComfyUI 节点 (Apple Silicon/MLX)</strong><br/>
    <a href="README.md.kr">한국어</a> | <a href="README.md">English</a>
</p>

## 概览

本分支将原有的节点升级以支持 **mflux 0.13.1**，同时保持了 ComfyUI 工作流的兼容性。它利用 mflux 0.13.x 的全新统一架构，不仅支持标准的 FLUX 生成，还支持 Fill（填充）、Depth（深度）、Redux（重组）和 Z-Image Turbo 等特殊变体。

- **后端**: mflux 0.13.1 (需要 macOS + Apple Silicon)。
- **图表兼容性**: 内部迁移了旧版输入，因此您的旧图表仍然可以工作。
- **统一加载**: 无缝处理本地路径、HuggingFace 仓库 ID 和预定义别名（例如 `dev`, `schnell`）。

## mflux 0.13.1 新特性
该版本带来了重大的后端增强：
- **Z-Image Turbo 支持**: 支持专为速度优化的快速蒸馏 Z-Image 变体（6B 参数）。
- **FIBO & Qwen 支持**: 针对 FIBO 和 Qwen-Image 架构的后端支持。
- **智能模型加载器**: 提供缓存模型与本地模型的视觉指示器，并支持递归文件夹扫描。
- **统一架构**: 改进了模型、LoRA 和 Tokenizer 的解析能力。

## 主要功能

- **核心生成**: 一个节点即可完成快速文生图 (text2img) 和 图生图 (img2img) (`QuickMfluxNode`)。
- **Z-Image Turbo**: 专为新的高速模型设计的独立节点 (`MFlux Z-Image Turbo`)。
- **硬件优化**: 专用节点用于 **Low RAM** (低内存) 模式和 **VAE Tiling** (分块解码)，防止在低内存 Mac 上崩溃。
- **FLUX 工具支持**: 专用于 **Fill** (内补绘制/Inpainting)、**Depth** (结构引导) 和 **Redux** (图像变体) 的节点。
- **ControlNet**: Canny 预览和尽力而为（best‑effort）的调节；包含对 **Upscaler** (放大) ControlNet 的支持。
- **LoRA 支持**: 统一的 LoRA 流程（应用 LoRA 时量化必须设为 8 或 None）。
- **量化**: 提供丰富的内存优化选项（None, 3, 4, 5, 6, 8-bit）。
- **元数据**: 保存完整的生成元数据 (PNG + JSON)，与 mflux CLI 工具兼容。

## 安装指南

### 使用 ComfyUI-Manager (推荐)
- 搜索 “Mflux-ComfyUI” 并安装。

### 手动安装
1. 进入您的 custom nodes 目录：
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```
2. 克隆仓库：
   ```bash
   git clone https://github.com/joonsoome/Mflux-ComfyUI.git
   ```
3. 激活您的 ComfyUI 虚拟环境并安装依赖：
   ```bash
   # 标准 venv 示例
   source /path/to/ComfyUI/venv/bin/activate

   pip install --upgrade pip wheel setuptools
   pip install 'mlx>=0.27.0' 'huggingface_hub>=0.26.0'
   pip install 'mflux==0.13.1'
   ```
4. 重启 ComfyUI。

**注意**: `mflux 0.13.1` 需要 `mlx >= 0.27.0`。如果您使用的是旧版本，请务必升级。

## 节点说明

### MFlux/Air (标准)
- **QuickMfluxNode**: 用于标准 FLUX 文生图、图生图、LoRA 和 ControlNet 的全能节点。
- **MFlux Z-Image Turbo**: Z-Image 生成专用节点（优化默认值：9 步，无引导）。
- **Mflux Optimizations**: 在此处配置 **Low RAM** (内存节能) 和 **VAE Tiling** 设置，并连接到主节点。
- **Mflux Models Loader**: 智能模型选择器。递归扫描 `models/Mflux` 并检查系统缓存。
  - 🟢 = 已缓存 (可直接使用)
  - 📁 = 本地 (位于 ComfyUI 文件夹中)
  - ☁️ = 别名 (可能会触发下载)
- **Mflux Models Downloader**: 直接从 HuggingFace 下载量化版或完整版模型到本地文件夹。
- **Mflux Custom Models**: 组合并保存自定义量化变体。

### MFlux/Pro (高级)
- **Mflux Fill**: FLUX.1-Fill 支持，用于内补绘制和外补绘制（需要遮罩）。
- **Mflux Depth**: FLUX.1-Depth 支持，用于结构引导生成。
- **Mflux Redux**: FLUX.1-Redux 支持，用于混合图像风格/结构。
- **Mflux Upscale**: 使用 Flux ControlNet Upscaler 进行图像放大。
- **Mflux Img2Img / Loras / ControlNet**: 用于构建自定义管道的模块化加载器。

## 使用提示

- **Z-Image Turbo**: 请使用专用节点。它默认设置为 **9 steps** 和 **0 guidance**（该模型必须使用 0 guidance）。
- **优化**: 如果遇到内存不足，请使用 **Mflux Optimizations** 节点。启用 `low_ram` 用于生成，启用 `vae_tiling` 用于解码大图像。
- **LoRA 兼容性**: 目前使用 LoRA 时，要求基础模型加载时 `quantize=8`（或者设为 None）。
- **尺寸**: 宽度和高度应为 16 的倍数（如有需要会自动调整）。
- **Guidance (引导系数)**:
  - `dev` 模型遵循 guidance 设置（默认约 3.5）。
  - `schnell` 模型忽略 guidance（保持默认即可）。
- **路径**:
  - 量化模型: `ComfyUI/models/Mflux`
  - LoRA: `ComfyUI/models/loras` (建议新建一个 `Mflux` 子文件夹以保持整洁)。
  - 从 HuggingFace 自动下载的模型（例如首次使用 Z-Image Turbo 节点时的 `filipstrand/Z-Image-Turbo-mflux-4bit`）：位于 `User/.cache/huggingface/hub`，按 `Cmd + Shift + .` 可显示隐藏的 .cache 文件夹。

## 工作流

请查看 `workflows` 文件夹中的 JSON 示例：
- `Mflux text2img.json`
- `Mflux img2img.json`
- `Mflux ControlNet.json`
- `Mflux Fill/Redux/Depth` 示例 (如果有)

Z-Image Turbo 的工作流嵌入在 `examples` 文件夹的 png 文件中：
- `Air_Z-Image-Turbo.png`
- `Air_Z-Image-Turbo_model_loader.png`
- `Air_Z-Image-Turbo_img2img_lora.png`

如果 ComfyUI 中节点显示为红色，请使用 Manager 的 “Install Missing Custom Nodes” 功能。

## 致谢

- **mflux**: 感谢 [@filipstrand](https://github.com/filipstrand) 及其贡献者。
- **raysers**: 最初的 ComfyUI 集成概念。
- MFlux-ComfyUI 2.0.0 by **joonsoome**.
- 部分代码结构参考了 **MFLUX-WEBUI**.

## 许可证

MIT
```