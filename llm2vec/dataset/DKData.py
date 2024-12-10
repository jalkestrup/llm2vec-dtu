import json
import random
import os
from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

class DKData(Dataset):
    def __init__(
        self,
        dataset_name: str = "DK-data",
        split: str = "validation",
        file_path: str = "cache/dk-data",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading DK data from {file_path}...")

        all_samples = []
        data_map = {}
        id_ = 0

        # List all jsonl files in the directory
        jsonl_files = [f for f in os.listdir(file_path) if f.endswith('.jsonl')]

        for jsonl_file in jsonl_files:
            dataset_name = os.path.splitext(jsonl_file)[0]
            logger.info(f"Loading dataset {dataset_name}...")

            with open(os.path.join(file_path, jsonl_file), "r") as f:
                dataset_samples = [json.loads(d) for d in f.readlines()]

            for sample in dataset_samples:
                query = self.separator + sample["query"]
                pos = self.separator + sample.get("positive", "")
                neg = self.separator + sample.get("negative", "")

                data_map.setdefault(dataset_name, []).append(id_)

                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name=dataset_name,
                    )
                )
                id_ += 1

        if self.shuffle_individual_datasets:
            for samples in data_map.values():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        logger.info(
            f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i: i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            final_idx_order.extend(batch)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "E5Data does not have a validation split."