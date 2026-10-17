"""Anima caption → MOSS-Image layout-JSON prompt, as a ComfyUI node.

A MOSS LoRA trained by the anima_lora trainer with ``caption_format=layout_json``
saw every caption through :func:`anime_tools.captions.layout_json.layout_json`;
this node runs the same function so a prompt typed in Anima format reaches the
text encoder as the identical string. The output goes into any STRING input
(``CLIPTextEncode`` with MOSS-Image's text encoder).

Torch-free. The danbooru tag KB (``danbooru_tags_classified.csv``) decides which
tags are characters / series / meta; it is looked up under ``HOME/models`` and
fetched there on first use when absent.
"""

from __future__ import annotations

import dataclasses
import logging
from functools import lru_cache

try:
    from anime_tools.captions.correction import (
        TAG_CSV_NAME,
        find_tag_csv,
        load_tag_knowledge_base,
    )
    from anime_tools.captions.layout_json import layout_json
    from anime_tools.downloads import by_id
except ImportError as e:  # pragma: no cover — install-time guidance
    raise ImportError(
        "the MOSS layout node needs `anime-tools` >= 0.7.13: "
        "pip install -U 'anime-tools @ git+https://github.com/sorryhyun/anime_tools'"
    ) from e

from .nodes import HOME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _kb():
    path = find_tag_csv(HOME)
    if path is None:
        asset = dataclasses.replace(by_id()["danbooru_tags"], dest=HOME / "models")
        logger.info("MOSS layout: fetching %s into %s", TAG_CSV_NAME, asset.dest)
        asset.fetch(log=logger.info)
        path = asset.dest / TAG_CSV_NAME
    logger.info("MOSS layout: tag KB %s", path)
    return load_tag_knowledge_base(path)


class AnimaCaptionToMossLayout:
    """Map an Anima-format caption onto MOSS-Image's layout-JSON prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Anima-format caption: tag bag, then optional "
                            "'On the left, …' position clauses and a "
                            "'Japanese text reads as \"…\"' clause."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "convert"
    CATEGORY = "anima"
    DESCRIPTION = (
        "Anima caption -> MOSS-Image layout JSON, byte-identical to what the "
        "anima_lora MOSS trainer encodes with caption_format=layout_json. "
        "Rating / meta, characters, series and @artists get their own summary "
        "keys, the remaining tags become summary.caption, each position clause "
        "a subject item, each quoted text line a text item. Use it in front of "
        "the text encoder for a LoRA trained on layout-JSON captions."
    )

    def convert(self, caption: str):
        text = layout_json(caption.strip(), _kb())
        logger.info("AnimaCaptionToMossLayout: %s", text)
        return (text,)


NODE_CLASS_MAPPINGS = {"AnimaCaptionToMossLayout": AnimaCaptionToMossLayout}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaCaptionToMossLayout": "Anima Caption → MOSS Layout JSON"
}
