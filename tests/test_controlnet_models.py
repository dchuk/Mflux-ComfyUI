import sys
import types


def test_controlnet_models_default_when_runtime_disabled(monkeypatch):
    import Mflux_Comfy.Mflux_Core as core

    monkeypatch.setattr(core, "_skip_mflux_import", True)
    monkeypatch.setattr(core, "ModelConfig", None)

    models = core.get_available_controlnet_models()

    assert models == [
        "InstantX/FLUX.1-dev-Controlnet-Canny",
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
    ]


def test_controlnet_models_discovered_from_available(monkeypatch):
    import Mflux_Comfy.Mflux_Core as core

    # Enable discovery path
    monkeypatch.setattr(core, "_skip_mflux_import", False)
    monkeypatch.setattr(core, "ModelConfig", object())

    dummy_cfg1 = types.SimpleNamespace(controlnet_model="Repo/First")
    dummy_cfg2 = types.SimpleNamespace(controlnet_model=None)
    dummy_cfg3 = types.SimpleNamespace(controlnet_model="Repo/Second")
    dummy_cfg4 = types.SimpleNamespace(controlnet_model="Repo/First")

    dummy_module = types.SimpleNamespace(
        AVAILABLE_MODELS={
            "a": dummy_cfg1,
            "b": dummy_cfg2,
            "c": dummy_cfg3,
            "d": dummy_cfg4,
        }
    )

    # Ensure import_module returns our dummy module
    from importlib import import_module as real_import_module

    def fake_import(name):
        if name == "mflux.config.model_config":
            return dummy_module
        return real_import_module(name)

    monkeypatch.setattr(core, "import_module", fake_import)
    monkeypatch.setitem(sys.modules, "mflux.config.model_config", dummy_module)

    models = core.get_available_controlnet_models()

    assert models == ["Repo/First", "Repo/Second"]


def test_controlnet_loader_uses_discovery(monkeypatch):
    import Mflux_Comfy.Mflux_Core as core
    import Mflux_Comfy.Mflux_Pro as pro

    monkeypatch.setattr(core, "get_available_controlnet_models", lambda: ["Repo/Alpha", "Repo/Beta"])

    input_types = pro.MfluxControlNetLoader.INPUT_TYPES()
    options, meta = input_types["required"]["model_selection"]

    assert options == ["Repo/Alpha", "Repo/Beta"]
    assert meta.get("default") == "Repo/Alpha"
