import os


# Disable the native ControlNet import during tests to avoid triggering MLX
# initialisation on systems without Apple Metal support. The runtime will
# still attempt the real import when this env variable is unset.
os.environ.setdefault("MFLUX_COMFY_DISABLE_CONTROLNET_IMPORT", "1")
os.environ.setdefault("MFLUX_COMFY_DISABLE_MLX_IMPORT", "1")
os.environ.setdefault("MFLUX_COMFY_DISABLE_MFLUX_IMPORT", "1")
