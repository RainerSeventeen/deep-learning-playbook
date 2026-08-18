"""Encode a text corpus with a trained BPE tokenizer into a NumPy token array.
本文件由 AI 撰写

Example:
    python -m cs336_basics.prepare_dataset \
        --input /root/autodl-tmp/data/owt_train.txt \
        --vocab /root/autodl-tmp/data/vocab/vocab_gpt2.json \
        --merges /root/autodl-tmp/data/vocab/merges_gpt2.json \
        --output /root/autodl-tmp/data/owt_train_tokens.npy
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .tokenizer import Tokenizer


_WORKER_TOKENIZER: Tokenizer | None = None


def _init_worker(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    """Create one tokenizer per encoder process."""
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer(vocab, merges, special_tokens)


def load_tokenizer(
    vocab_path: str | Path,
    merges_path: str | Path,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Load the hex-encoded vocabulary and merge files written by tokenizer.py."""
    with Path(vocab_path).open(encoding="utf-8") as file:
        serialized_vocab = json.load(file)
    with Path(merges_path).open(encoding="utf-8") as file:
        serialized_merges = json.load(file)

    vocab = {int(token_id): bytes.fromhex(token) for token_id, token in serialized_vocab.items()}
    merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in serialized_merges]
    return Tokenizer(vocab, merges, special_tokens)


def iter_token_ids(tokenizer: Tokenizer, input_path: str | Path) -> Iterator[int]:
    """Yield token IDs without loading the corpus or its IDs into memory at once."""
    with Path(input_path).open(encoding="utf-8") as file:
        yield from tokenizer.encode_iterable(file)


def find_line_boundaries(input_path: str | Path, chunk_size: int) -> list[tuple[int, int]]:
    """Split a UTF-8 text file into chunks without splitting a line.

    ``Tokenizer.encode_iterable`` tokenizes one line at a time.  Keeping this
    boundary makes parallel encoding produce the same IDs as the serial path.
    Newlines are single-byte UTF-8 characters, so every returned offset is safe
    to decode independently.
    """
    file_size = Path(input_path).stat().st_size
    if file_size == 0:
        return []

    boundaries = [0]
    with Path(input_path).open("rb") as file:
        target = chunk_size
        while target < file_size:
            file.seek(target)
            while True:
                data = file.read(1024 * 1024)
                if not data:
                    break
                newline = data.find(b"\n")
                if newline >= 0:
                    boundaries.append(file.tell() - len(data) + newline + 1)
                    break
            target += chunk_size
    boundaries.append(file_size)
    return list(zip(boundaries, boundaries[1:]))


def _iter_chunk_token_ids(input_path: str | Path, chunk: tuple[int, int]) -> Iterator[int]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("encoder worker was not initialized")
    start, end = chunk
    with Path(input_path).open("rb") as file:
        file.seek(start)
        while file.tell() < end:
            line = file.readline()
            if not line:
                break
            yield from _WORKER_TOKENIZER.encode(line.decode("utf-8"))


def _count_chunk(task: tuple[str, tuple[int, int], int]) -> tuple[int, int]:
    input_path, chunk, index = task
    return index, sum(1 for _ in _iter_chunk_token_ids(input_path, chunk))


def _write_chunk(task: tuple[str, str, tuple[int, int], int, int]) -> int:
    input_path, output_path, chunk, offset, index = task
    tokens = np.lib.format.open_memmap(output_path, mode="r+")
    for position, token_id in enumerate(_iter_chunk_token_ids(input_path, chunk), start=offset):
        tokens[position] = token_id
    tokens.flush()
    del tokens
    return index


def encode_to_npy(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    num_workers: int = 1,
    chunk_size_mib: int = 64,
) -> int:
    """Write corpus token IDs as a one-dimensional ``uint16`` NumPy array.

    The corpus is scanned twice: once to determine the exact token count and once
    to fill an on-disk memory map.  Both scans can use independent processes;
    chunks end at newlines so the resulting IDs retain serial ordering.
    """
    if max(tokenizer.vocab, default=-1) > np.iinfo(np.uint16).max:
        raise ValueError("uint16 output requires every token ID to be at most 65535")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if chunk_size_mib <= 0:
        raise ValueError("chunk_size_mib must be positive")

    source = Path(input_path)
    chunks = find_line_boundaries(source, chunk_size_mib * 1024**2)
    file_size = source.stat().st_size
    tasks = [(str(source), chunk, index) for index, chunk in enumerate(chunks)]
    pool_args = (tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)

    with Pool(processes=min(num_workers, len(chunks) or 1), initializer=_init_worker, initargs=pool_args) as pool:
        counts = [0] * len(chunks)
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Counting tokens") as progress:
            for index, count in pool.imap_unordered(_count_chunk, tasks):
                counts[index] = count
                start, end = chunks[index]
                progress.update(end - start)

        token_count = sum(counts)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tokens = np.lib.format.open_memmap(destination, mode="w+", dtype=np.uint16, shape=(token_count,))
        tokens.flush()
        del tokens

        offsets = np.cumsum([0, *counts[:-1]], dtype=np.int64)
        write_tasks = [
            (str(source), str(destination), chunk, int(offsets[index]), index)
            for index, chunk in enumerate(chunks)
        ]
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Writing tokens") as progress:
            for index in pool.imap_unordered(_write_chunk, write_tasks):
                start, end = chunks[index]
                progress.update(end - start)
    return token_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text corpus into uint16 BPE token IDs.")
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 text corpus")
    parser.add_argument("--vocab", type=Path, required=True, help="hex-encoded tokenizer vocabulary JSON")
    parser.add_argument("--merges", type=Path, required=True, help="hex-encoded tokenizer merges JSON")
    parser.add_argument("--output", type=Path, required=True, help="destination .npy file")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="encoder processes (default: all logical CPU cores)",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=64,
        help="input chunk size per task in MiB (default: 64)",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="special token preserved as one ID; repeat this option as needed",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    tokenizer = load_tokenizer(args.vocab, args.merges, args.special_token)
    token_count = encode_to_npy(
        tokenizer,
        args.input,
        args.output,
        num_workers=args.num_workers,
        chunk_size_mib=args.chunk_size_mib,
    )
    elapsed = time.perf_counter() - start_time
    print(f"wrote {token_count:,} uint16 token IDs to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
