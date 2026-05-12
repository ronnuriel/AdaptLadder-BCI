import torch
from torch.utils.data import Dataset


class BrainToTextDataset(Dataset):
    def __init__(self, data_dict, sessions_list, post_implant_map=None, add_post_implant_day=False):
        self.samples = []
        self.add_post_implant_day = add_post_implant_day

        for session_name, session_data in data_dict.items():
            if session_name not in sessions_list:
                continue

            day_idx = sessions_list.index(session_name)
            implant_day_val = 0.5
            if self.add_post_implant_day and post_implant_map is not None:
                implant_day_val = post_implant_map.get(session_name, 0.5)

            for i in range(len(session_data["neural_features"])):
                self.samples.append({
                    "neural": session_data["neural_features"][i],
                    "phonemes": session_data["seq_class_ids"][i],
                    "phoneme_len": session_data["seq_len"][i],
                    "day_idx": day_idx,
                    "corpus_id": session_data["corpus_ids"][i],
                    "implant_day_val": implant_day_val,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        neural_features = torch.from_numpy(item["neural"]).float()

        if self.add_post_implant_day:
            time_col = torch.full(
                (neural_features.shape[0], 1),
                item["implant_day_val"],
                dtype=neural_features.dtype,
            )
            neural_features = torch.cat([neural_features, time_col], dim=1)

        return {
            "neural": neural_features,
            "phonemes": item["phonemes"],
            "phoneme_len": item["phoneme_len"],
            "day_idx": item["day_idx"],
            "corpus_id": item["corpus_id"],
        }


def validation_collate_fn(batch):
    day_indices = torch.tensor([item["day_idx"] for item in batch], dtype=torch.long)
    corpus_ids = torch.cat([item["corpus_id"] for item in batch])

    neural_features = [item["neural"] for item in batch]
    n_time_steps = torch.tensor([f.shape[0] for f in neural_features], dtype=torch.long)
    padded_neural = torch.nn.utils.rnn.pad_sequence(
        neural_features, batch_first=True, padding_value=0.0
    )

    target_phonemes = [torch.from_numpy(item["phonemes"]) for item in batch]
    target_lengths = torch.tensor([item["phoneme_len"] for item in batch], dtype=torch.long)
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        target_phonemes, batch_first=True, padding_value=0
    )

    return {
        "neural_features": padded_neural,
        "n_time_steps": n_time_steps,
        "day_indicies": day_indices,
        "corpus_ids": corpus_ids,
        "seq_class_ids": padded_targets,
        "phone_seq_lens": target_lengths,
    }
