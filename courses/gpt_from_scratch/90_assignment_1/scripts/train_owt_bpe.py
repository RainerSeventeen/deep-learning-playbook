"""Train and serialize the OWT 32K byte-level BPE vocabulary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cs336_basics.tokenizer import train_bpe_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 OWT training corpus")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for vocab_32k.json and merges_32k.json",
    )
    # The remote container is capped at 64 GiB. Two workers and 32 chunks keep
    # the decoded text and temporary Counters below the container memory limit.
    default_processes = min(2, os.cpu_count() or 1)
    parser.add_argument("--num-process", type=int, default=default_processes)
    parser.add_argument("--num-chunks", type=int, default=32)
    args = parser.parse_args()

    vocab, merges = train_bpe_tokenizer(
        input_path=args.input,
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
        desired_num_chunks=args.num_chunks,
        num_process=args.num_process,
        show_progress=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "vocab_32k.json").open("w", encoding="utf-8") as file:
        json.dump({token_id: token.hex() for token_id, token in vocab.items()}, file)
    with (args.output_dir / "merges_32k.json").open("w", encoding="utf-8") as file:
        json.dump([(left.hex(), right.hex()) for left, right in merges], file)
    print(f"wrote {len(vocab):,} vocabulary entries and {len(merges):,} merges to {args.output_dir}")


if __name__ == "__main__":
    main()
