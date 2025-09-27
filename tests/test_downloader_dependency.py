import importlib

import pytest


def test_download_requires_huggingface(monkeypatch):
    module = importlib.import_module("Mflux_Comfy.Mflux_Air")
    monkeypatch.setattr(module, "snapshot_download", None, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        module.download_hg_model("flux.1-dev-mflux-4bit")

    assert "huggingface_hub is required" in str(excinfo.value)
