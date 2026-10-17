"""Anima Tagger ComfyUI custom nodes — see :mod:`.nodes` and :mod:`.moss_layout`."""

from . import moss_layout, nodes

NODE_CLASS_MAPPINGS = {**nodes.NODE_CLASS_MAPPINGS, **moss_layout.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes.NODE_DISPLAY_NAME_MAPPINGS,
    **moss_layout.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
