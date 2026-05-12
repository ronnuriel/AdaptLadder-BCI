from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


LOGIT_TO_PHONEME = [
    "BLANK",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
    " | ",
]


def add_official_model_training_to_path(repo_root: Path) -> Path:
    model_training = repo_root / "third_party" / "nejm-brain-to-text" / "model_training"
    if not model_training.exists():
        raise FileNotFoundError(
            f"Official baseline code not found at {model_training}. "
            "Run `git submodule update --init --recursive`."
        )
    if str(model_training) not in sys.path:
        sys.path.insert(0, str(model_training))
    return model_training


def load_model_args(model_path: Path):
    args_path = model_path / "checkpoint" / "args.yaml"
    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(args_path)
    except ModuleNotFoundError:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading checkpoint args requires `omegaconf` or `pyyaml`. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc
        with args_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)


def resolve_device(gpu_number: int) -> torch.device:
    if torch.cuda.is_available() and gpu_number >= 0:
        if gpu_number >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu_number} requested, but only {torch.cuda.device_count()} CUDA devices are visible.")
        return torch.device(f"cuda:{gpu_number}")
    return torch.device("cpu")


def load_official_gru_decoder(repo_root: Path, model_path: Path, device: torch.device):
    add_official_model_training_to_path(repo_root)
    from rnn_model import GRUDecoder

    model_args = load_model_args(model_path)
    model = GRUDecoder(
        neural_dim=model_args["model"]["n_input_features"],
        n_units=model_args["model"]["n_units"],
        n_days=len(model_args["dataset"]["sessions"]),
        n_classes=model_args["dataset"]["n_classes"],
        rnn_dropout=model_args["model"]["rnn_dropout"],
        input_dropout=model_args["model"]["input_network"]["input_layer_dropout"],
        n_layers=model_args["model"]["n_layers"],
        patch_size=model_args["model"]["patch_size"],
        patch_stride=model_args["model"]["patch_stride"],
    )

    checkpoint = torch.load(model_path / "checkpoint" / "best_checkpoint", map_location=device, weights_only=False)
    state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        clean_key = key.replace("module.", "").replace("_orig_mod.", "")
        state_dict[clean_key] = value
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_args


def greedy_ctc_decode(logits: np.ndarray, blank_id: int = 0) -> list[int]:
    """Official-style greedy decode: argmax, remove blanks, then remove repeats."""
    frame_ids = np.argmax(logits, axis=-1).astype(int).tolist()
    no_blanks = [idx for idx in frame_ids if idx != blank_id]
    return [idx for i, idx in enumerate(no_blanks) if i == 0 or idx != no_blanks[i - 1]]


def trim_target_sequence(seq_class_ids: np.ndarray, seq_len: int) -> list[int]:
    return [int(x) for x in seq_class_ids[: int(seq_len)] if int(x) != 0]


def edit_distance(reference: Sequence[int], hypothesis: Sequence[int]) -> int:
    prev = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        curr = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ref_item == hyp_item else 1),
                )
            )
        prev = curr
    return prev[-1]


def phoneme_error_rate(reference: Sequence[int], hypothesis: Sequence[int]) -> tuple[int, int, float]:
    n_ref = len(reference)
    distance = edit_distance(reference, hypothesis)
    return distance, n_ref, float(distance / n_ref) if n_ref else math.nan


def logits_quality_metrics(logits: np.ndarray, blank_id: int = 0) -> dict[str, float]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
    pred = np.argmax(probs, axis=-1)
    return {
        "blank_rate": float(np.mean(pred == blank_id)),
        "mean_confidence": float(np.mean(np.max(probs, axis=-1))),
        "entropy": float(np.mean(entropy)),
    }


def phoneme_ids_to_string(ids: Sequence[int]) -> str:
    return " ".join(LOGIT_TO_PHONEME[int(idx)] for idx in ids)
