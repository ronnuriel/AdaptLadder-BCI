import h5py
import pandas as pd
from datetime import datetime
import torch


def create_post_implant_day_map(csv_path, sessions_list):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    min_day = df["Post-implant day"].min()
    max_day = df["Post-implant day"].max()

    day_map = {}
    for session in sessions_list:
        try:
            date_part = session.split(".", 1)[1]
            session_date = pd.to_datetime(date_part.replace(".", "-"))
            row = df[df["Date"] == session_date]
            if not row.empty:
                raw_day = row.iloc[0]["Post-implant day"]
                norm_day = (raw_day - min_day) / (max_day - min_day)
                day_map[session] = float(norm_day)
            else:
                day_map[session] = 0.5
        except Exception:
            day_map[session] = 0.5
    return day_map


def load_h5py_file(file_path, b2txt_csv_df):
    data = {
        "neural_features": [],
        "n_time_steps": [],
        "seq_class_ids": [],
        "seq_len": [],
        "transcriptions": [],
        "session": [],
        "block_num": [],
        "trial_num": [],
        "corpus_ids": [],
    }

    all_corpuses = b2txt_csv_df["Corpus"].unique()
    corpus_map = {name: i for i, name in enumerate(all_corpuses)}

    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            g = f[key]

            neural_features = g["input_features"][:]
            n_time_steps = g.attrs["n_time_steps"]
            seq_class_ids = g["seq_class_ids"][:] if "seq_class_ids" in g else None
            seq_len = g.attrs["seq_len"] if "seq_len" in g.attrs else None
            transcription = g["transcription"][:] if "transcription" in g else None
            session = g.attrs["session"]
            block_num = g.attrs["block_num"]
            trial_num = g.attrs["trial_num"]

            date_part = session.split(".", 1)[1]
            dt = datetime.strptime(date_part, "%Y.%m.%d")
            formatted_date = dt.strftime("%Y-%m-%d")

            corpus_list = b2txt_csv_df[
                (b2txt_csv_df["Date"] == formatted_date)
                & (b2txt_csv_df["Block number"] == block_num)
            ]["Corpus"].values

            corpus_id = corpus_map[corpus_list[0]] if len(corpus_list) > 0 else 0

            data["neural_features"].append(neural_features)
            data["n_time_steps"].append(n_time_steps)
            data["seq_class_ids"].append(seq_class_ids)
            data["seq_len"].append(seq_len)
            data["transcriptions"].append(transcription)
            data["session"].append(session)
            data["block_num"].append(block_num)
            data["trial_num"].append(trial_num)
            data["corpus_ids"].append([corpus_id])

    data["corpus_ids"] = torch.tensor(data["corpus_ids"], dtype=torch.long)
    return data


def load_dataset_split(data_dir, sessions, split_name, b2txt_csv_df):
    dataset_dict = {}
    total_trials = 0

    print(f"\n--- Loading '{split_name}' data ---")
    for session in sessions:
        session_path = data_dir / session
        if not session_path.is_dir():
            continue

        file_path = session_path / f"data_{split_name}.hdf5"
        if file_path.exists():
            data = load_h5py_file(str(file_path), b2txt_csv_df)
            dataset_dict[session] = data
            total_trials += len(data["neural_features"])
            print(f"  {session}: {len(data['neural_features'])} trials")
        else:
            print(f"  Missing: {file_path}")

    print(f"Total '{split_name}' trials loaded: {total_trials}")
    return dataset_dict, total_trials
