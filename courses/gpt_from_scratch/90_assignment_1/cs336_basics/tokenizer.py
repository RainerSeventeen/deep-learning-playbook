from __future__ import annotations

import os
from collections import Counter
from multiprocessing import Pool
from os import PathLike
from typing import BinaryIO

import regex as re

PRE_TOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Split a file at occurrences of a special token.

    Every returned boundary is at a UTF-8-safe location because special tokens
    are themselves UTF-8 byte strings. Fewer boundaries can be returned when
    multiple guesses resolve to the same occurrence.
    """
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")
    if not split_special_token:
        raise ValueError("split_special_token must not be empty")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = max(4096, len(split_special_token))
    overlap = len(split_special_token) - 1

    for boundary_index in range(1, len(chunk_boundaries) - 1):
        search_position = chunk_boundaries[boundary_index]

        while search_position < file_size:
            file.seek(search_position)
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = search_position + found_at
                break

            if len(mini_chunk) < mini_chunk_size:
                search_position = file_size
            else:
                search_position += len(mini_chunk) - overlap
        else:
            search_position = file_size

        if search_position >= file_size:
            chunk_boundaries[boundary_index] = file_size

    return sorted(set(chunk_boundaries))


def pretokenize_chunk(
    chunk: tuple[int, int],
    special_token_pattern: str | None,
    input_path: str | PathLike[str],
) -> Counter[str]:
    """Count pre-tokens in one byte range of the training corpus."""
    start, end = chunk
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")

    # 按照 special token 进一步切分 segment
    segments = re.split(special_token_pattern, text) if special_token_pattern else [text]
    counts: Counter[str] = Counter()
    # 更新单词计数
    for segment in segments:
        counts.update(match.group() for match in re.finditer(PRE_TOKEN_PAT, segment))
    return counts


def pre_tokenize(
    input_path: str | PathLike[str],
    desired_num_chunks: int,
    num_process: int,
    special_tokens: list[str],
) -> Counter[str]:
    """Pre-tokenize a corpus and return the frequency of each pre-token."""
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")
    if num_process <= 0:
        raise ValueError("num_process must be positive")

    # 优先匹配更长的, 随后按照字典序
    sorted_special_tokens = sorted(set(special_tokens), key=lambda token: (-len(token), token))
    special_token_pattern = (
        "|".join(re.escape(token) for token in sorted_special_tokens) if sorted_special_tokens else None
    )

    with open(input_path, "rb") as file:
        if sorted_special_tokens:
            boundary_token = sorted_special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(file, desired_num_chunks, boundary_token)
        else:
            file.seek(0, os.SEEK_END)
            boundaries = [0, file.tell()]

    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    if not chunks:
        return Counter()

    worker_count = min(num_process, len(chunks))
    args = [(chunk, special_token_pattern, input_path) for chunk in chunks]
    if worker_count == 1:
        results = [pretokenize_chunk(*args[0])]
    else:
        with Pool(processes=worker_count) as pool:
            results = pool.starmap(pretokenize_chunk, args)

    total: Counter[str] = Counter()
    for result in results:
        total.update(result)
    return total


def _merge_pair_in_tokens(
    tokens: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    """Replace every non-overlapping occurrence of pair from left to right."""
    # 将 tokens 按照 merged_token 执行合并, 返回合并后的 byte 序列
    result: list[bytes] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            result.append(merged_token)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return tuple(result)


def bpe_merge(
    pre_token_counts: Counter[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte-level BPE merges from pre-token frequencies."""
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must not contain duplicates")

    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")

    # 初始 256 个词, 构建字节词表
    vocab_values = [bytes([byte]) for byte in range(256)]
    vocab_values.extend(token.encode("utf-8") for token in special_tokens)

    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for word, count in pre_token_counts.items():
        byte_tokens = tuple(bytes([byte]) for byte in word.encode("utf-8"))
        word_counts[byte_tokens] += count # 初始 byte 组合与计数

    merges: list[tuple[bytes, bytes]] = []
    number_of_merges = vocab_size - minimum_vocab_size

    for _ in range(number_of_merges):
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for tokens, count in word_counts.items():
            for left, right in zip(tokens, tokens[1:]):
                pair_counts[(left, right)] += count

        if not pair_counts:
            break

        pair = max(pair_counts, key=lambda candidate: (pair_counts[candidate], candidate))
        merged_token = pair[0] + pair[1]    # 需要合并的 token 对
        merges.append(pair)
        vocab_values.append(merged_token) # 增加一个 bpe 合并的词进去

        updated_word_counts: Counter[tuple[bytes, ...]] = Counter() # 扩充后词表, 单位是 pretoken 的 byte 元组
        for tokens, count in word_counts.items():
            if any(left == pair[0] and right == pair[1] for left, right in zip(tokens, tokens[1:])):
                tokens = _merge_pair_in_tokens(tokens, pair, merged_token)
            updated_word_counts[tokens] += count
        word_counts = updated_word_counts

    vocab = dict(enumerate(vocab_values))
    
    # vocal 指定了词表的元素, merges 确定了词表的顺序 (越靠前频率越高, encode 优先级越高, 更先合并)
    return vocab, merges


def train_bpe_tokenizer(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    desired_num_chunks: int = 4,
    num_process: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer vocabulary and ordered merge list."""
    pre_token_counts = pre_tokenize(
        input_path,
        desired_num_chunks=desired_num_chunks,
        num_process=num_process,
        special_tokens=special_tokens,
    )
    return bpe_merge(pre_token_counts, vocab_size, special_tokens)


if __name__ == "__main__":
    train_bpe_tokenizer(
        input_path="./data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=2000,
        desired_num_chunks=4,
        num_process=4,
        special_tokens=["<|endoftext|>"],
    )
