"""Write NSR special tokens to a text file accepted by ms-swift."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nsr.config import NSRConfig
from nsr.constants import special_token_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsr-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = NSRConfig.from_file(args.nsr_config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(special_token_names(cfg.num_emb_tokens)) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
