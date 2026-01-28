"""Tensor conversion utilities for ComfyUI integration.

ComfyUI IMAGE format: [B, H, W, C] float32 in range [0, 1]
PIL Image format: [H, W, C] uint8 in range [0, 255]
"""
import numpy as np
import torch
from PIL import Image


def pil_to_comfy_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor.

    Args:
        pil_image: PIL Image in any mode (will be converted to RGB)

    Returns:
        torch.Tensor with shape [1, H, W, 3] and dtype float32, values in [0, 1]
    """
    # Ensure RGB mode
    rgb_image = pil_image.convert("RGB")

    # Convert to numpy and normalize to [0, 1]
    image_np = np.array(rgb_image).astype(np.float32) / 255.0

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_np)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def comfy_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor to PIL Image.

    Args:
        tensor: torch.Tensor with shape [B, H, W, C] or [H, W, C],
                float32 in range [0, 1]

    Returns:
        PIL.Image.Image in RGB mode (first image if batch > 1)
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first in batch

    # Move to CPU, convert to numpy
    image_np = tensor.cpu().numpy()

    # Clamp and convert to uint8
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(image_np)
