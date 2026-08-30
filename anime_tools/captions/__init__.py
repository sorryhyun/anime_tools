"""Caption grammar + polishing (torch-free).

``position_clauses`` is the single caption grammar — never ``split(",")`` a
caption by hand (``docs/contract.md`` §3). ``shuffle`` carries the training-
time shuffle/no-artist grammar the trainer re-exports.
"""

from anime_tools.captions.position_clauses import (  # noqa: F401
    ParsedCaption,
    compose_caption,
    parse_caption,
)

__all__ = ["ParsedCaption", "compose_caption", "parse_caption"]
