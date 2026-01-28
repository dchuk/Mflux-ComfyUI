import sys
import types
from unittest.mock import MagicMock

def test_controlnet_models_default_when_runtime_disabled(monkeypatch):
    import Mflux_Comfy.Mflux_Core as core

    # Force ModelConfig to be None
    monkeypatch.setattr(core, "ModelConfig", None)

    models = core.get_available_controlnet_models()

    assert models == [
        "InstantX/FLUX.1-dev-Controlnet-Canny",
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
    ]


def test_controlnet_loader_uses_discovery(monkeypatch):
    import Mflux_Comfy.Mflux_Core as core
    import Mflux_Comfy.Mflux_Pro as pro
    import importlib

    # 1. Ensure ModelConfig is NOT None so it tries to import
    monkeypatch.setattr(core, "ModelConfig", object())

    # 2. Mock the mflux.config.model_config module in sys.modules
    dummy_cfg1 = types.SimpleNamespace(controlnet_model="Repo/Alpha")
    dummy_cfg2 = types.SimpleNamespace(controlnet_model="Repo/Beta")

    dummy_module = types.SimpleNamespace(
        AVAILABLE_MODELS={
            "a": dummy_cfg1,
            "b": dummy_cfg2,
        }
    )
    monkeypatch.setitem(sys.modules, "mflux.config.model_config", dummy_module)

    # 3. Call the function
    models = core.get_available_controlnet_models()

    # 4. Verify
    assert "Repo/Alpha" in models
    assert "Repo/Beta" in models

    # 5. Verify Loader uses this list
    # We need to patch the function on core because Mflux_Pro imports it
    monkeypatch.setattr(core, "get_available_controlnet_models", lambda: ["Repo/Alpha", "Repo/Beta"])

    # Reload Mflux_Pro to pick up the patched function
    importlib.reload(pro)

    # Reload INPUT_TYPES to pick up the change
    input_types = pro.MfluxControlNetLoader.INPUT_TYPES()
    options, meta = input_types["required"]["model_selection"]

    assert options == ["Repo/Alpha", "Repo/Beta"]