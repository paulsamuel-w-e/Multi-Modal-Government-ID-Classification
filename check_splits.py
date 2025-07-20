from datasets import Dataset
from collections import Counter
from pathlib import Path

def check_class_distribution(parquet_dir="./Data/split"):
    parquet_dir = Path(parquet_dir)

    # Define paths to the splits
    train_path = parquet_dir / "train_dataset.parquet"
    val_path = parquet_dir / "val_dataset.parquet"
    test_path = parquet_dir / "test_dataset.parquet"

    # Load the datasets
    train_dataset = Dataset.from_parquet(str(train_path))
    val_dataset = Dataset.from_parquet(str(val_path))
    test_dataset = Dataset.from_parquet(str(test_path))

    # Get unique labels
    train_labels = train_dataset["label"]
    val_labels = val_dataset["label"]
    test_labels = test_dataset["label"]

    # Print stats
    print("\n📊 Class distribution:")
    print(f"Train split:     {len(set(train_labels))} classes → {dict(Counter(train_labels))}")
    print(f"Validation split:{len(set(val_labels))} classes → {dict(Counter(val_labels))}")
    print(f"Test split:      {len(set(test_labels))} classes → {dict(Counter(test_labels))}")

    # Check if all labels are present in all splits
    all_labels = set(train_labels + val_labels + test_labels)
    print(f"\nTotal unique labels across all splits: {len(all_labels)} → {sorted(all_labels)}")

if __name__ == "__main__":
    check_class_distribution("./Data/split")
