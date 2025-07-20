from datasets import Dataset, DatasetDict
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
import torch
import json

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
            text=example["full_text"],
            boxes=normalized_bbox,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = self.label2id[example["label"]]
        return encoding

    def load_my_data(self, raw=False):
        """
        Loads train/val/test Parquet files and returns a DatasetDict.
        If raw=True, returns unprocessed dataset.
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

        self.label2id = {label: i for i, label in enumerate(sorted(set(dataset["train"]["label"])))}
        self.id2label = {i: label for label, i in self.label2id.items()}

        sample = dataset["train"][0]
        print("Sample keys:", sample.keys())
        if not raw:
            self._preprocess(sample)

        if raw:
            return dataset, self.label2id, self.id2label

        dataset = dataset.map(self._preprocess, remove_columns=dataset["train"].column_names)

        return dataset, self.label2id, self.id2label

    def load_test_data(self, label2id=None, raw=False):
        """
        Loads and preprocesses only the test dataset from test_dataset.parquet.
        If label2id is provided, uses it; otherwise, infers from test data.
        If raw=True, returns unprocessed dataset.
        """
        test_path = self.parquet_dir / "test_dataset.parquet"
        self._check_files(test_path)

        test_dataset = Dataset.from_parquet(str(test_path))

        print("Test dataset column names:", test_dataset.column_names)

        if label2id is not None:
            self.label2id = label2id
        else:
            self.label2id = {label: i for i, label in enumerate(sorted(set(test_dataset["label"])))}
        self.id2label = {i: label for label, i in self.label2id.items()}

        sample = test_dataset[0]
        print("Test sample keys:", sample.keys())
        if not raw:
            self._preprocess(sample)

        if raw:
            return test_dataset, self.label2id, self.id2label

        test_dataset = test_dataset.map(self._preprocess, remove_columns=test_dataset.column_names)

        return test_dataset, self.label2id, self.id2label

    def export_to_json(self, output_dir="./"):
        """
        Exports raw train/val/test datasets to separate JSON files with id, text, label, and image_path.
        Returns data as dict of splits and label mappings.
        """
        dataset, label2id, id2label = self.load_my_data(raw=True)
        data = {"train": [], "validation": [], "test": []}
        for split in ["train", "validation", "test"]:
            for item in dataset[split]:
                data[split].append({
                    "id": item["id"],
                    "text": item["full_text"],
                    "label": item["label"],
                    "image_path": item["image_path"]
                })
            output_path = Path(output_dir) / f"{split}_dataset.json"
            with open(output_path, 'w') as f:
                json.dump(data[split], f, indent=2)
            print(f"✅ Exported {split} dataset to {output_path}")
        return data, label2id, id2label