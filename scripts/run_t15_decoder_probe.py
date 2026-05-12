from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decoder_eval import load_official_gru_decoder, resolve_device


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
    args = parser.parse_args()

    device = resolve_device(args.gpu_number)
    _model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Model has {len(model_args['dataset']['sessions'])} day-specific input layers.")
    print("Next step: wire HDF5 val iteration, greedy PER, blank rate, confidence, and CSV outputs.")


if __name__ == "__main__":
    main()
