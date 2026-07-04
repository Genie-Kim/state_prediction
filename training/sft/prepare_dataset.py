"""Convert trajectory JSONL into ms-swift SFT JSONL with NSR metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nsr.config import NSRConfig
from nsr.schema import build_messages, iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Trajectory JSONL.")
    parser.add_argument("--output", required=True, help="ms-swift JSONL output.")
    parser.add_argument("--nsr-config", required=True)
    parser.add_argument(
        "--no-anchor",
        action="store_true",
        help="Override config and omit the anchor image for anchor ablation.",
    )
    args = parser.parse_args()

    cfg = NSRConfig.from_file(args.nsr_config)
    include_anchor = False if args.no_anchor else cfg.include_anchor
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for sample in iter_jsonl(args.input):
            row = build_messages(sample, cfg, include_anchor=include_anchor)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} rows to {output}")


if __name__ == "__main__":
    main()
