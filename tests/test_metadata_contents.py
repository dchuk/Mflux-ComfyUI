from PIL import Image


def test_metadata_contains_mflux_and_lora_info(monkeypatch, tmp_path):
    # Create a small image and run save_images_with_metadata to write a JSON file
    src = tmp_path / "img_meta.png"
    Image.new("RGB", (16, 16)).save(src)

    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_save_image_path", lambda filename_prefix, output_dir, h, w: (str(tmp_path), filename_prefix, 0, "", filename_prefix))

    import Mflux_Comfy.Mflux_Core as core
    import numpy as _np
    import torch as _torch
    arr = (_np.ones((16, 16, 3), dtype=_np.uint8) * 128).astype(_np.uint8)
    # Use channels-last layout
    t = _torch.from_numpy(arr).unsqueeze(0).to(_torch.float32) / 255.0

    # Call the real saver with some lora info and extra parameters to validate in JSON
    # Using keyword arguments for robustness
    core.save_images_with_metadata(
        images=(t,),
        prompt="hello",
        model_alias="dev",
        quantize="8",
        quantize_effective="8",
        model_path="",
        seed=42,
        height=t.shape[1],
        width=t.shape[2],
        steps=5,
        guidance=2.0,
        image_path=str(src),
        image_strength=1.0,
        lora_paths=["loraA"],
        lora_scales=[0.5],
        control_image_path="ctrl.png",
        control_strength=0.6,
        control_model="cnet",
        extra_pnginfo={},
        base_model_hint="dev",
        low_ram=True,
        negative_prompt_used="bad quality",
    )

    mflux_dir = tmp_path / "MFlux"
    json_files = list(mflux_dir.glob("*.json"))
    assert len(json_files) == 1
    import json
    with open(json_files[0], 'r') as f:
        data = json.load(f)

    assert "mflux_version" in data and isinstance(data.get("mflux_version"), str)
    assert data.get("lora_paths") == ["loraA"]
    assert data.get("lora_scales") == [0.5]
    # Basic fields
    assert data.get("prompt") == "hello"
    assert data.get("model_alias") == "dev"
    assert data.get("quantize") == "8"
    assert data.get("quantize_effective") == "8"
    assert data.get("seed") == 42
    assert data.get("image_path") == str(src)
    assert data.get("image_strength") == 1.0
    # Flags passed through
    assert data.get("base_model_hint") == "dev"
    assert data.get("low_ram") is True
    # Guidance only present for dev model
    assert data.get("guidance") == 2.0
    # ControlNet fields
    assert data.get("control_image_path") == "ctrl.png"
    assert data.get("control_strength") == 0.6
    assert data.get("control_model") == "cnet"
    # New field
    assert data.get("negative_prompt_used") == "bad quality"