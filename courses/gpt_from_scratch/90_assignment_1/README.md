# CS336 Spring 2025 Assignment 1: Basics

## Source materials

This directory is based on Stanford CS336's official
[assignment1-basics repository](https://github.com/stanford-cs336/assignment1-basics).
The assignment handout and test snapshots are intentionally not tracked here;
retrieve them from that upstream repository when needed. To run the complete
official test suite, also copy its `tests/fixtures/` directory; this repository
only keeps the small text fixtures used for focused local checks.

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

### Forward inference

Use the resolved `config.yml` and `checkpoint.pt` from one training run to
reconstruct the model, encode a prompt with the same BPE files, and obtain the
logits for every prompt token:

```sh
uv run -m cs336_basics.inference \
  --config runs/<timestamp>/config.yml \
  --checkpoint runs/<timestamp>/checkpoint.pt \
  --vocab data/benchmark/vocab_TinyStories.json \
  --merges data/benchmark/merges_TinyStories.json \
  --special-token '<|endoftext|>' \
  --prompt 'Once upon a time'
```

`--checkpoint`, `--vocab`, and `--merges` can instead be set in the optional
`inference` section of the YAML config. The command reports prompt token IDs,
the logits shape, and the argmax next-token ID. Prompts longer than
`model.context_length` are rejected.

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
