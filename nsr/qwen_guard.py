"""Guardrails for the intended Qwen VL model family."""

from __future__ import annotations

import argparse
import re


def validate_qwen_vl_model_name(model: str, max_b: float = 30.0) -> None:
    name = model.lower()
    if "qwen" not in name:
        raise ValueError(f"expected a Qwen model, got: {model}")
    if "qwen3.5" not in name and "qwen3-vl" not in name:
        raise ValueError(
            f"expected Qwen3.5 or Qwen3-VL family for this project, got: {model}"
        )
    match = re.search(r"(\d+(?:\.\d+)?)b", name)
    if match and float(match.group(1)) > max_b:
        raise ValueError(f"model appears larger than {max_b:g}B: {model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--max-b", type=float, default=30.0)
    args = parser.parse_args()
    validate_qwen_vl_model_name(args.model, args.max_b)


if __name__ == "__main__":
    main()
