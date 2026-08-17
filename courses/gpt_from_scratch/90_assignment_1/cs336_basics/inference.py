"""Load a trained language model and run one forward pass for a text prompt.

Example:
    uv run -m cs336_basics.inference \
        --config runs/20260817-195357/config.yml \
        --checkpoint runs/20260817-195357/checkpoint.pt \
        --vocab data/benchmark/vocab_TinyStories.json \
        --merges data/benchmark/merges_TinyStories.json \
        --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .model import TransformerLM
from .prepare_dataset import load_tokenizer
from .train import create_model


def load_model_for_inference(
    model_config: Mapping[str, object], checkpoint_path: str | Path, device: str
) -> tuple[TransformerLM, int]:
    """Recreate a model from config and restore its trained parameters."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    model = create_model(model_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("checkpoint must contain a 'model' state dictionary")

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint.get("iteration", 0))


@torch.inference_mode()
def forward_prompt(
    model: TransformerLM, token_ids: list[int], context_length: int, device: str
) -> torch.Tensor:
    """Return logits with shape ``(1, prompt_length, vocab_size)`` for a prompt."""
    if not token_ids:
        raise ValueError("prompt must encode to at least one token")
    if len(token_ids) > context_length:
        raise ValueError(
            f"prompt has {len(token_ids)} tokens, exceeding context_length={context_length}"
        )

    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    return model(tokens)


def _config_or_argument(
    argument: str | None, inference_config: Mapping[str, object], name: str
) -> str:
    value = argument if argument is not None else inference_config.get(name)
    if value is None:
        raise ValueError(f"missing {name}; pass --{name.replace('_', '-')} or set inference.{name}")
    return str(value)


def run_inference(
    config: Mapping[str, Any],
    prompt: str,
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    merges_path: str | Path,
    device: str | None = None,
    special_tokens: list[str] | None = None,
) -> tuple[list[int], torch.Tensor, int]:
    """Tokenize a prompt, load a checkpoint, and return token IDs, logits, and iteration."""
    model_config = config["model"]
    inference_config = config.get("inference", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("config section 'model' must be a mapping")
    if not isinstance(inference_config, Mapping):
        raise ValueError("config section 'inference' must be a mapping")

    selected_device = device or str(inference_config.get("device", "cpu"))
    tokenizer = load_tokenizer(vocab_path, merges_path, special_tokens)
    token_ids = tokenizer.encode(prompt)
    model, iteration = load_model_for_inference(model_config, checkpoint_path, selected_device)
    logits = forward_prompt(model, token_ids, int(model_config["context_length"]), selected_device)
    return token_ids, logits, iteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one language-model forward pass for a prompt")
    parser.add_argument("--config", type=Path, required=True, help="training or resolved run YAML config")
    parser.add_argument("--prompt", required=True, help="text prompt to encode and forward")
    parser.add_argument("--checkpoint", type=Path, help="trained checkpoint path")
    parser.add_argument("--vocab", type=Path, help="hex-encoded BPE vocabulary JSON")
    parser.add_argument("--merges", type=Path, help="hex-encoded BPE merges JSON")
    parser.add_argument("--device", help="PyTorch device, overriding inference.device")
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="special token recognized by the tokenizer; repeat as needed",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    inference_config = config.get("inference", {})
    if not isinstance(inference_config, Mapping):
        raise ValueError("config section 'inference' must be a mapping")
    checkpoint = _config_or_argument(args.checkpoint, inference_config, "checkpoint")
    vocab = _config_or_argument(args.vocab, inference_config, "vocab")
    merges = _config_or_argument(args.merges, inference_config, "merges")
    special_tokens = args.special_token
    if special_tokens is None:
        configured_tokens = inference_config.get("special_tokens", [])
        if not isinstance(configured_tokens, list) or not all(
            isinstance(token, str) for token in configured_tokens
        ):
            raise ValueError("inference.special_tokens must be a list of strings")
        special_tokens = configured_tokens

    token_ids, logits, iteration = run_inference(
        config,
        args.prompt,
        checkpoint,
        vocab,
        merges,
        device=args.device,
        special_tokens=special_tokens,
    )
    next_token_id = int(logits[0, -1].argmax().item())
    print(f"checkpoint iteration: {iteration}")
    print(f"prompt token ids: {token_ids}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"argmax next token id: {next_token_id}")


if __name__ == "__main__":
    main()
