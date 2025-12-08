from PIL import Image
import os
import pytest

def test_generate_config_fallback(monkeypatch, tmp_path):
    # Create a tiny image to act as input
    img = tmp_path / "in.png"
    Image.new("RGB", (16, 16)).save(img)

    import folder_paths
    # annotated filepath returns our tmp file
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda x: str(img))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_save_image_path", lambda filename_prefix, output_dir, h, w: (str(tmp_path), filename_prefix, 0, "", filename_prefix))

    import Mflux_Comfy.Mflux_Core as core

    # Mock flux generation
    class DummyFlux:
        def generate_image(self, **kwargs):
            # Return dummy image
            import numpy as np
            import torch
            arr = (np.ones((16, 16, 3), dtype=np.float32) * 255.0)
            t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
            return t

    # Patch load_or_create_model to return our dummy flux
    monkeypatch.setattr(core, "load_or_create_model", lambda *a, **k: DummyFlux())

    # Call generate_image with vae_tiling=True.
    # This verifies that the function accepts the argument and proceeds without error.
    result = core.generate_image(
        prompt="p", model_string="dev", seed=-1, width=16, height=16, steps=1,
        guidance=1.0, quantize="8", metadata=True, model_path="",
        img2img_pipeline=None, loras_pipeline=None, controlnet_pipeline=None, vae_tiling=True
    )

    # Should return a tuple with a tensor-like first element
    assert isinstance(result, tuple)
    assert len(result) == 1