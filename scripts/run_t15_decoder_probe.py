from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decoder_eval import (
    add_official_model_training_to_path,
    greedy_ctc_decode,
    logits_quality_metrics,
    load_official_gru_decoder,
    phoneme_error_rate,
    phoneme_ids_to_string,
    resolve_device,
    trim_target_sequence,
)


def _as_plain_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _count_trials(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return len(handle.keys())


def _set_use_amp(model_args, enabled: bool) -> None:
    try:
        model_args["use_amp"] = enabled
    except TypeError:
        model_args.use_amp = enabled


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a decoder-facing T15 PER/logit probe without the language model.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--mode", choices=["native-day", "cross-day"], default="native-day")
    parser.add_argument("--source-session", default=None)
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--output-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/t15_decoder_probe_session_summary.csv"))
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-trials-per-session", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    device = resolve_device(args.gpu_number)
    model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)
    _set_use_amp(model_args, enabled=(device.type == "cuda" and bool(model_args["use_amp"])))
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep

    b2txt_csv_df = pd.read_csv(args.csv_path)
    sessions = list(model_args["dataset"]["sessions"])

    if args.mode == "cross-day":
        if args.source_session is None:
            raise ValueError("--source-session is required in cross-day mode")
        if args.source_session not in sessions:
            raise ValueError(f"Unknown source session {args.source_session!r}")
        source_input_layer = sessions.index(args.source_session)
    else:
        source_input_layer = None

    eval_sessions = []
    total_trials = 0
    for session in sessions:
        eval_file = args.data_dir / session / f"data_{args.eval_type}.hdf5"
        if not eval_file.exists():
            continue
        eval_sessions.append((session, eval_file))
        total_trials += _count_trials(eval_file)
        if args.max_sessions is not None and len(eval_sessions) >= args.max_sessions:
            break

    if args.max_trials_per_session is not None:
        total_trials = sum(min(_count_trials(path), args.max_trials_per_session) for _, path in eval_sessions)

    rows = []
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with tqdm(total=total_trials, desc=f"{args.mode} {args.eval_type} PER", unit="trial") as progress:
        for session, eval_file in eval_sessions:
            data = load_h5py_file(str(eval_file), b2txt_csv_df)
            native_input_layer = sessions.index(session)
            input_layer = native_input_layer if args.mode == "native-day" else source_input_layer
            assert input_layer is not None

            n_trials = len(data["neural_features"])
            if args.max_trials_per_session is not None:
                n_trials = min(n_trials, args.max_trials_per_session)

            for trial_idx in range(n_trials):
                neural_input = np.expand_dims(data["neural_features"][trial_idx], axis=0)
                neural_input = torch.tensor(neural_input, device=device, dtype=dtype)

                logits = runSingleDecodingStep(
                    neural_input,
                    input_layer,
                    model,
                    model_args,
                    device,
                )[0]

                pred_ids = greedy_ctc_decode(logits)
                if args.eval_type == "val":
                    true_ids = trim_target_sequence(data["seq_class_ids"][trial_idx], data["seq_len"][trial_idx])
                    edit_dist, n_phonemes, per = phoneme_error_rate(true_ids, pred_ids)
                    true_phonemes = phoneme_ids_to_string(true_ids)
                else:
                    true_ids = []
                    edit_dist, n_phonemes, per = np.nan, np.nan, np.nan
                    true_phonemes = ""

                quality = logits_quality_metrics(logits)
                rows.append(
                    {
                        "session": session,
                        "block": int(data["block_num"][trial_idx]),
                        "trial": int(data["trial_num"][trial_idx]),
                        "mode": args.mode,
                        "input_layer_session": sessions[input_layer],
                        "sentence_label": _as_plain_string(data["sentence_label"][trial_idx]),
                        "true_phonemes": true_phonemes,
                        "pred_phonemes": phoneme_ids_to_string(pred_ids),
                        "edit_distance": edit_dist,
                        "num_phonemes": n_phonemes,
                        "PER": per,
                        **quality,
                    }
                )
                progress.update(1)

    df = pd.DataFrame(rows)
    args.output_trials.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_trials, index=False)

    summary = (
        df.groupby(["session", "mode", "input_layer_session"], as_index=False)
        .agg(
            n_trials=("PER", "size"),
            mean_PER=("PER", "mean"),
            median_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
    )
    summary.to_csv(args.output_summary, index=False)

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Model has {len(model_args['dataset']['sessions'])} day-specific input layers.")
    print(f"Wrote trial-level results to {args.output_trials}")
    print(f"Wrote session summary to {args.output_summary}")
    if args.eval_type == "val" and not df.empty:
        aggregate_per = df["edit_distance"].sum() / df["num_phonemes"].sum()
        print(f"Aggregate phoneme-weighted PER: {aggregate_per:.4f}")
        print(f"Aggregate trial-mean PER: {df['PER'].mean():.4f}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
