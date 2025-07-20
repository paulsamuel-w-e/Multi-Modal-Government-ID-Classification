from datasets import Dataset, DatasetDict
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
import torch

class DatasetLoader:
    def __init__(self, parquet_dir="./Data/split", model_name="microsoft/layoutlmv3-base"):
        """
        Initialize the DatasetLoader with a directory and model name.
        """
        self.parquet_dir = Path(parquet_dir)
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        self.label2id = None
        self.id2label = None

    def _check_files(self, *paths):
        """
        Check if all specified parquet files exist.
        """
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Parquet file not found at {path}")

    def _normalize_bbox(self, bbox, width, height):
        """
        Normalize bounding boxes to [0, 1023] based on image dimensions.
        """
        normalized = []
        for box in bbox:
            x1 = min(max(box[0], 0), width) / width * 1023
            y1 = min(max(box[1], 0), height) / height * 1023
            x2 = min(max(box[2], 0), width) / width * 1023
            y2 = min(max(box[3], 0), height) / height * 1023
            normalized.append([int(x1), int(y1), int(x2), int(y2)])
        return normalized

    def _preprocess(self, example):
        """
        Preprocess a single example for LayoutLMv3.
        """
        image = Image.open(example["image_path"]).convert("RGB")
        img_width, img_height = image.size

        normalized_bbox = self._normalize_bbox(example["bbox"], img_width, img_height)

        encoding = self.processor(
            images=image,
            text=example["words"],
            boxes=normalized_bbox,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # Flatten tensors for Trainer compatibility
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = self.label2id[example["label"]]
        return encoding

    def load_my_data(self):
        """
        Loads train/val/test Parquet files and returns a processed DatasetDict.
        """
        train_path = self.parquet_dir / "train_dataset.parquet"
        val_path = self.parquet_dir / "val_dataset.parquet"
        test_path = self.parquet_dir / "test_dataset.parquet"

        self._check_files(train_path, val_path, test_path)

        train_dataset = Dataset.from_parquet(str(train_path))
        val_dataset = Dataset.from_parquet(str(val_path))
        test_dataset = Dataset.from_parquet(str(test_path))

        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

        print("Column names:", dataset["train"].column_names)

        # Label encoding based on training data
        self.label2id = {label: i for i, label in enumerate(sorted(set(dataset["train"]["label"])))}
        self.id2label = {i: label for label, i in self.label2id.items()}

        # Test preprocessing on one sample
        sample = dataset["train"][0]
        print("Sample keys:", sample.keys())
        self._preprocess(sample)  # Validate preprocessing

        # Preprocess entire dataset
        dataset = dataset.map(self._preprocess, remove_columns=dataset["train"].column_names)

        return dataset, self.label2id, self.id2label

    def load_test_data(self, label2id=None):
        """
        Loads and preprocesses only the test dataset from test_dataset.parquet.
        If label2id is provided, uses it; otherwise, infers from test data.
        """
        test_path = self.parquet_dir / "test_dataset.parquet"
        self._check_files(test_path)

        test_dataset = Dataset.from_parquet(str(test_path))

        print("Test dataset column names:", test_dataset.column_names)

        # Use provided label2id or infer from test data
        if label2id is not None:
            self.label2id = label2id
        else:
            self.label2id = {label: i for i, label in enumerate(sorted(set(test_dataset["label"])))}
        self.id2label = {i: label for label, i in self.label2id.items()}

        # Test preprocessing on one sample
        sample = test_dataset[0]
        print("Test sample keys:", sample.keys())
        self._preprocess(sample)  # Validate preprocessing

        # Preprocess test dataset
        test_dataset = test_dataset.map(self._preprocess, remove_columns=test_dataset.column_names)

        return test_dataset, self.label2id, self.id2label