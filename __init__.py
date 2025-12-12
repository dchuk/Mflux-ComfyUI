try:
    # Normal package import when executed as a package by ComfyUI
    from .Mflux_Comfy import sam3_mps_fix
    from .Mflux_Comfy.Mflux_Air import (
        QuickMfluxNode,
        MfluxModelsLoader,
        MfluxModelsDownloader,
        MfluxCustomModels,
        MfluxZImageNode,
        MfluxOptimizations,
        MfluxFiboPrompt
    )
    from .Mflux_Comfy.Mflux_Pro import (
        MfluxImg2Img,
        MfluxLorasLoader,
        MfluxControlNetLoader,
        MfluxUpscale,
        MfluxFill,   # Added
        MfluxDepth,  # Added
        MfluxRedux,  # Added
        MfluxZImageInpaint # Added
    )
except Exception:
    # Fallback for environments where relative imports fail (e.g., direct execution during tests).
    import os
    import sys
    pkg_root = os.path.dirname(__file__)
    comfy_root = os.path.dirname(os.path.dirname(__file__))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    from Mflux_Comfy.Mflux_Air import (
        QuickMfluxNode,
        MfluxModelsLoader,
        MfluxModelsDownloader,
        MfluxCustomModels,
        MfluxZImageNode,
        MfluxOptimizations,
        MfluxFiboPrompt
    )
    from Mflux_Comfy.Mflux_Pro import (
        MfluxImg2Img,
        MfluxLorasLoader,
        MfluxControlNetLoader,
        MfluxUpscale,
        MfluxFill,   # Added
        MfluxDepth,  # Added
        MfluxRedux,  # Added
        MfluxZImageInpaint # Added
    )

NODE_CLASS_MAPPINGS = {
    "QuickMfluxNode": QuickMfluxNode,
    "MfluxZImageNode": MfluxZImageNode,
    "MfluxModelsLoader": MfluxModelsLoader,
    "MfluxModelsDownloader": MfluxModelsDownloader,
    "MfluxCustomModels": MfluxCustomModels,
    "MfluxImg2Img": MfluxImg2Img,
    "MfluxLorasLoader": MfluxLorasLoader,
    "MfluxControlNetLoader": MfluxControlNetLoader,
    "MfluxUpscale": MfluxUpscale,
    "MfluxOptimizations": MfluxOptimizations,
    # New Pro Nodes
    "MfluxFill": MfluxFill,
    "MfluxDepth": MfluxDepth,
    "MfluxRedux": MfluxRedux,
    "MfluxFiboPrompt": MfluxFiboPrompt,
    "MfluxZImageInpaint": MfluxZImageInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuickMfluxNode": "Quick MFlux Generation",
    "MfluxZImageNode": "MFlux Z-Image Turbo",
    "MfluxModelsLoader": "MFlux Models Loader",
    "MfluxModelsDownloader": "MFlux Models Downloader",
    "MfluxCustomModels": "MFlux Custom Models",
    "MfluxImg2Img": "Mflux Img2Img",
    "MfluxLorasLoader": "MFlux Loras Loader",
    "MfluxControlNetLoader": "MFlux ControlNet Loader",
    "MfluxUpscale": "MFlux Upscale",
    "MfluxOptimizations": "MFlux Optimizations",
    # New Pro Nodes
    "MfluxFill": "MFlux Fill (Inpainting)",
    "MfluxDepth": "MFlux Depth",
    "MfluxRedux": "MFlux Redux",
    "MfluxFiboPrompt": "MFlux Fibo JSON Prompt",
    "MfluxZImageInpaint": "MFlux Z-Image Inpaint (Turbo)",
}