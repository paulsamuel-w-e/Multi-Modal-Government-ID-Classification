from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import argparse

def get_datasets(file_path: str, val_size: float, test_size: float):
    df = pd.read_parquet(file_path)

    # First stratified split: train vs (val + test)
    strat_split1 = StratifiedShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=42)
    for train_idx, temp_idx in strat_split1.split(df, df["label"]):
        df_train = df.iloc[train_idx]
        df_temp = df.iloc[temp_idx]

    # Second stratified split: val vs test
    relative_test_size = test_size / (val_size + test_size)
    strat_split2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=42)
    for val_idx, test_idx in strat_split2.split(df_temp, df_temp["label"]):
        df_val = df_temp.iloc[val_idx]
        df_test = df_temp.iloc[test_idx]

    # Convert pandas to Hugging Face datasets
    train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
    val_ds = Dataset.from_pandas(df_val.reset_index(drop=True))
    test_ds = Dataset.from_pandas(df_test.reset_index(drop=True))

    return train_ds, val_ds, test_ds

def main():
    parser = argparse.ArgumentParser(description="Load and stratified split Parquet dataset")
    parser.add_argument("--file_path", type=str, default="./Data/extracts/documents.parquet", help="Path to the Parquet file")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size (fraction)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size (fraction)")
    parser.add_argument("--output_path", type=str, default="./Data/split", help="Path to save split datasets")

    args = parser.parse_args()

    try:
        file = Path(args.file_path)
        if not file.exists():
            raise FileNotFoundError(f"File does not exist: {file}")
        if file.suffix != '.parquet':
            raise ValueError("Provided file is not a Parquet file.")

        train_ds, val_ds, test_ds = get_datasets(str(file), args.val_size, args.test_size)

        print(f"Train dataset size: {len(train_ds)}")
        print(f"Validation dataset size: {len(val_ds)}")
        print(f"Test dataset size: {len(test_ds)}")

        output_dir = Path(args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_ds.to_parquet(str(output_dir / "train_dataset.parquet"))
        val_ds.to_parquet(str(output_dir / "val_dataset.parquet"))
        test_ds.to_parquet(str(output_dir / "test_dataset.parquet"))

        print("Datasets saved as Parquet files.")

        return {"train": output_dir / "train_dataset.parquet",
                "val": output_dir / "val_dataset.parquet",
                "test": output_dir / "test_dataset.parquet"}

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

main() if __name__ == "__main__" else None