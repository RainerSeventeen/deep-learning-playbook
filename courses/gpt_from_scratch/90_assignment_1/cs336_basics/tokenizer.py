from __future__ import annotations

import json
import heapq
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from os import PathLike
from typing import BinaryIO

import regex as re

PRE_TOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        if any(not token for token in self.special_tokens):
            raise ValueError("special tokens must not be empty")

        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_token_ids = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }

        special_pattern = _special_token_pattern(self.special_tokens)
        self.special_token_pattern = (
            re.compile(f"({special_pattern})") if special_pattern is not None else None
        )
        self._bpe_cache: dict[str, tuple[bytes, ...]] = {}

    def _apply_bpe(self, pre_token: str) -> tuple[bytes, ...]:
        cached = self._bpe_cache.get(pre_token)
        if cached is not None:
            return cached

        tokens = tuple(bytes([byte]) for byte in pre_token.encode("utf-8"))
        while len(tokens) > 1:
            ranked_pairs = (
                (self.merge_ranks[pair], pair)
                for pair in zip(tokens, tokens[1:]) 
                if pair in self.merge_ranks
            )
            # 排名最高且在 pre_token 出现的那个组合
            best = min(ranked_pairs, default=None)
            if best is None:
                break

            pair = best[1]
            tokens = _merge_pair_in_tokens(tokens, pair, pair[0] + pair[1])

        self._bpe_cache[pre_token] = tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        
        parts = (
            self.special_token_pattern.split(text)
            if self.special_token_pattern is not None
            else [text]
        )

        token_ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_token_ids:
                # special token 需要映射
                token_ids.append(self.special_token_ids[part])
                continue

            for pre_token in pre_tokenize_text(part):
                token_ids.extend(self.token_to_id[token] for token in self._apply_bpe(pre_token))
        return token_ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into a UTF-8 string."""
        return b"".join(self.vocab[token_id] for token_id in ids).decode(
            "utf-8", errors="replace"
        )

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode chunks of text from an iterable."""
        for text in iterable:
            yield from self.encode(text)


def _special_token_pattern(special_tokens: list[str]) -> str | None:
    """对 special token 排序并构造正则 pattern"""
    sorted_special_tokens = sorted(set(special_tokens), key=lambda token: (-len(token), token))
    if not sorted_special_tokens:
        return None
    return "|".join(re.escape(token) for token in sorted_special_tokens)


def _pre_tokenize_with_pattern(text: str, special_token_pattern: str | None) -> list[str]:
    segments = re.split(special_token_pattern, text) if special_token_pattern else [text]
    return [
        match.group()
        for segment in segments
        for match in re.finditer(PRE_TOKEN_PAT, segment)
    ]


def pre_tokenize_text(
    text: str,
    special_tokens: list[str] | None = None,
) -> list[str]:
    """操作内存的 pre_tokenize 接口"""
    special_token_pattern = _special_token_pattern(special_tokens or [])
    return _pre_tokenize_with_pattern(text, special_token_pattern)


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
    """面向训练进程池的接口函数, 对 chunk 进行 pretokenize, 并统计总数"""
    start, end = chunk
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")

    return Counter(_pre_tokenize_with_pattern(text, special_token_pattern))


def _pretokenize_task(args: tuple[tuple[int, int], str | None, str | PathLike[str]]) -> Counter[str]:
    return pretokenize_chunk(*args)


def pre_tokenize(
    input_path: str | PathLike[str],
    desired_num_chunks: int,
    num_process: int,
    special_tokens: list[str],
    show_progress: bool = False,
) -> Counter[str]:
    """Pre-tokenize a corpus and return the frequency of each pre-token."""
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")
    if num_process <= 0:
        raise ValueError("num_process must be positive")

    # 优先匹配更长的, 随后按照字典序
    sorted_special_tokens = sorted(set(special_tokens), key=lambda token: (-len(token), token))
    special_token_pattern = _special_token_pattern(special_tokens)

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
    total: Counter[str] = Counter()
    if worker_count == 1:
        total.update(pretokenize_chunk(*args[0]))
    else:
        # Recreate workers between chunks so their large temporary Counters are
        # returned to the OS instead of accumulating across a long corpus.
        with Pool(processes=worker_count, maxtasksperchild=1) as pool:
            iterator = pool.imap(_pretokenize_task, args)
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(
                    iterator,
                    total=len(args),
                    desc="Pre-tokenizing OWT",
                    unit="chunk",
                )
            for result in iterator:
                # Merge each worker result immediately instead of retaining one
                # Counter per input chunk in the parent process.
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


def _count_token_pairs(tokens: tuple[bytes, ...], count: int) -> Counter[tuple[bytes, bytes]]:
    """Count adjacent pairs in one word type, weighted by its corpus frequency."""
    return Counter({pair: occurrences * count for pair, occurrences in Counter(zip(tokens, tokens[1:])).items()})


class _ReversePair:
    """Heap key that makes bytes pairs resolve in descending lexicographic order."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]) -> None:
        self.pair = pair

    def __lt__(self, other: _ReversePair) -> bool:
        return self.pair > other.pair


def _push_pair(pair_heap: list[tuple[int, _ReversePair]], pair: tuple[bytes, bytes], count: int) -> None:
    if count > 0:
        heapq.heappush(pair_heap, (-count, _ReversePair(pair)))


def bpe_merge(
    pre_token_counts: Counter[str],
    vocab_size: int,
    special_tokens: list[str],
    show_progress: bool = False,
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

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for tokens, count in word_counts.items():
        token_pairs = _count_token_pairs(tokens, count)
        pair_counts.update(token_pairs)
        for pair in token_pairs:
            pair_to_words[pair].add(tokens)

    pair_heap: list[tuple[int, _ReversePair]] = []
    for pair, count in pair_counts.items():
        _push_pair(pair_heap, pair, count)

    merges: list[tuple[bytes, bytes]] = []
    number_of_merges = vocab_size - minimum_vocab_size

    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(total=number_of_merges, desc="BPE merges", unit="merge")

    for _ in range(number_of_merges):
        while pair_heap and pair_counts.get(pair_heap[0][1].pair, 0) != -pair_heap[0][0]:
            heapq.heappop(pair_heap)
        if not pair_heap:
            break

        pair = heapq.heappop(pair_heap)[1].pair
        affected_words = tuple(pair_to_words[pair])
        merged_token = pair[0] + pair[1]    # 需要合并的 token 对
        merges.append(pair)
        vocab_values.append(merged_token) # 增加一个 bpe 合并的词进去

        # Only words containing the selected pair can change the next iteration's
        # pair frequencies. Updating those contributions avoids a full recount.
        pair_deltas: Counter[tuple[bytes, bytes]] = Counter()
        for tokens in affected_words:
            count = word_counts.pop(tokens)
            old_pairs = _count_token_pairs(tokens, count)
            new_tokens = _merge_pair_in_tokens(tokens, pair, merged_token)
            new_pairs = _count_token_pairs(new_tokens, count)
            pair_deltas.subtract(old_pairs)
            pair_deltas.update(new_pairs)
            for old_pair in old_pairs:
                pair_to_words[old_pair].discard(tokens)
            for new_pair in new_pairs:
                pair_to_words[new_pair].add(new_tokens)
            word_counts[new_tokens] += count

        for candidate, delta in pair_deltas.items():
            new_count = pair_counts[candidate] + delta
            if new_count > 0:
                pair_counts[candidate] = new_count
                _push_pair(pair_heap, candidate, new_count)
            else:
                pair_counts.pop(candidate, None)
                pair_to_words.pop(candidate, None)

        if len(pair_heap) > max(1024, 4 * len(pair_counts)):
            pair_heap = []
            for candidate, count in pair_counts.items():
                _push_pair(pair_heap, candidate, count)

        if progress is not None:
            progress.update()

    if progress is not None:
        progress.close()

        # Counter.subtract retains zero-count entries, which would otherwise
        # allow an unavailable pair to be selected when all remaining counts are zero.
        pair_counts = Counter({candidate: count for candidate, count in pair_counts.items() if count > 0})

    vocab = dict(enumerate(vocab_values))

    # vocal 指定了词表的元素, merges 确定了词表的顺序 (越靠前频率越高, encode 优先级越高, 更先合并)
    return vocab, merges


def train_bpe_tokenizer(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    desired_num_chunks: int = 4,
    num_process: int = 4,
    show_progress: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer vocabulary and ordered merge list."""
    pre_token_counts = pre_tokenize(
        input_path,
        desired_num_chunks=desired_num_chunks,
        num_process=num_process,
        special_tokens=special_tokens,
        show_progress=show_progress,
    )
    vocab, merges = bpe_merge(pre_token_counts, vocab_size, special_tokens, show_progress=show_progress)

    return vocab, merges


if __name__ == "__main__":
    """
    使用指令
    sudo py-spy record \
    --subprocesses \
    --rate 100 \
    -o data/benchmark/tokenizer-profile.svg \
    -- .venv/bin/python cs336_basics/tokenizer.py
    """
    # 1. 在测试集上测试数据, 并记录性能信息
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    input_size = os.path.getsize(input_path)
    start_time = time.perf_counter()

    vocab, merges = train_bpe_tokenizer(
        input_path=input_path,
        vocab_size=10000,
        desired_num_chunks=4,
        num_process=4,
        special_tokens=["<|endoftext|>"],
    )

    # 统计时间参数
    elapsed_time = time.perf_counter() - start_time
    print("\nTokenizer training benchmark")
    print(f"  input:            {input_path}")
    print(f"  input size:       {input_size / 1024**2:.2f} MiB")
    print(f"  elapsed time:     {elapsed_time:.2f} s")
    print(f"  throughput:       {input_size / 1024**2 / elapsed_time:.2f} MiB/s")
    print(f"  vocabulary size:  {len(vocab):,}")
    print(f"  merges learned:   {len(merges):,}")

    # 2. dump 数据并保存下来
    os.makedirs("./data/benchmark", exist_ok=True)
    with open("./data/benchmark/vocab_TinyStories.json", mode="w", encoding="utf-8") as f:
        json.dump({token_id: token.hex() for token_id, token in vocab.items()}, f)
    with open("./data/benchmark/merges_TinyStories.json", mode="w", encoding="utf-8") as f:
        json.dump([(left.hex(), right.hex()) for left, right in merges], f)
