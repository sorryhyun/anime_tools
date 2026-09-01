"""Tokenizer loading for caption-length / erasure-pool passes (curation side).

Takes tokenizer **directories** only, never a ``*.safetensors`` encoder path;
callers resolve a model to its tokenizer dir first. Torch-free apart from
``transformers``.
"""

from __future__ import annotations

from pathlib import Path


def _require_dir(path: str | Path, what: str) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(
            f"{what} must be a tokenizer *directory*, got {p}. Pass the bundled "
            "config dir (the trainer's `make` targets resolve it; a raw "
            "`.safetensors` text-encoder path is not accepted here)."
        )
    return p


def load_qwen3_tokenizer_from_dir(tokenizer_dir: str | Path):
    """``AutoTokenizer`` from a directory; pads with EOS when no pad token is set."""
    from transformers import AutoTokenizer

    p = _require_dir(tokenizer_dir, "--qwen3")
    tokenizer = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_t5_tokenizer_from_dir(tokenizer_dir: str | Path):
    """``T5TokenizerFast`` from a directory holding ``spiece.model`` +
    ``tokenizer.json`` (a bundled config dir) or a saved-pretrained layout."""
    from transformers import T5TokenizerFast

    p = _require_dir(tokenizer_dir, "--t5_tokenizer_path")
    spiece = p / "spiece.model"
    if spiece.exists():
        return T5TokenizerFast(
            vocab_file=str(spiece), tokenizer_file=str(p / "tokenizer.json")
        )
    return T5TokenizerFast.from_pretrained(str(p), local_files_only=True)
