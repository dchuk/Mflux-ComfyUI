import importlib
import sys
import pytest
from unittest.mock import MagicMock

def reload_mflux_air():
    module_name = "Mflux_Comfy.Mflux_Air"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)

def test_download_requires_huggingface(monkeypatch):
    # Simulate huggingface_hub being missing
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    # Reload module to trigger the import check
    module = reload_mflux_air()

    # Ensure snapshot_download is None
    monkeypatch.setattr(module, "snapshot_download", None)

    # Expect a RuntimeError when trying to download
    with pytest.raises(RuntimeError) as excinfo:
        module.download_hg_model("flux.1-dev-mflux-4bit")

    assert "huggingface_hub is required" in str(excinfo.value)

def test_download_calls_snapshot(monkeypatch, tmp_path):
    # Mock snapshot_download
    mock_download = MagicMock()

    module = reload_mflux_air()
    monkeypatch.setattr(module, "snapshot_download", mock_download)
    monkeypatch.setattr(module, "mflux_dir", str(tmp_path))

    # Call download
    module.download_hg_model("flux.1-schnell-mflux-4bit")

    # Verify snapshot_download was called with correct args
    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args[1]
    assert call_kwargs["repo_id"] == "madroid/flux.1-schnell-mflux-4bit"
    assert call_kwargs["local_dir_use_symlinks"] is False