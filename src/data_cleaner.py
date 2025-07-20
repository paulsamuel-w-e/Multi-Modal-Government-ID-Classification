# ./src/data_cleaner.py

import gc
import regex as re
from pathlib import Path
import simplejson
import tqdm
import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Load a single JSON file
def load_text_data(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return simplejson.load(f)


# Remove items from a list by index (used for bbox, confidence, words, etc.)
def remove_text_data(json_list: list, index_list: list):
    for idx in sorted(index_list, reverse=True):  # Critical: reverse order
        if 0 <= idx < len(json_list):
            del json_list[idx]
    return json_list


# Save cleaned list of dicts to a JSON file
def write_text_data(file_path: Path, data: list):
    with open(file_path, 'w', encoding='utf-8') as f:
        simplejson.dump(data, f, ensure_ascii=False, indent=4)


# Convert polygon bbox (list of points) to a rectangle [x0, y0, x1, y1]
def polygon_to_rect(polygon):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


# Reorder dict keys for consistency
def reorder_keys(d):
    desired_order = ["id", "label", "image_path", "full_text", "words", "bbox", "confidence"]
    return {k: d[k] for k in desired_order if k in d}


# Clean individual OCR JSON file using regex and bounding box simplification
def perform_regex(text_path: Path):
    doc = load_text_data(text_path)

    cleaned_texts = []
    removed_indices = []

    for idx, text in enumerate(doc['full_text']):
        # Apply regex cleanup
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
        text = re.sub(r'[^\w\s!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', '', text)  # Remove weird unicode
        text = re.sub(r'(?<!\w)[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~](?!\w)', '', text)  # Remove isolated punctuations
        text = text.strip()

        if text:
            cleaned_texts.append(text)
        else:
            removed_indices.append(idx)

    # Critical: reverse sort to safely delete from lists
    sorted_desc = sorted(removed_indices, reverse=True)

    cleaned_conf = remove_text_data(doc['confidence'], sorted_desc)
    cleaned_bbox = remove_text_data(doc['bbox'], sorted_desc)
    cleaned_words = remove_text_data(doc['full_text'], sorted_desc)  # if needed, not reused later

    # Final cleaned structure
    doc['full_text'] = " ".join(cleaned_texts).strip()
    doc['words'] = cleaned_texts
    doc['bbox'] = [polygon_to_rect(poly) for poly in cleaned_bbox]
    doc['confidence'] = cleaned_conf

    return reorder_keys(doc)


# Main pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="./Data/extracts/ocr_texts",
        help="Path to your OCR extracted texts directory"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    dataset = []

    loop = tqdm.tqdm(input_path.iterdir(), desc="Processing directories")

    for dir_path in loop:
        if dir_path.is_dir():
            for file_path in dir_path.iterdir():
                if not file_path.name.endswith('.json'):
                    print(f"Skipping non-json file: {file_path.name}")
                    continue
                try:
                    doc = perform_regex(file_path)
                    dataset.append(doc)
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
        gc.collect()

    print(f"Number of objects processed: {len(dataset)}")

    # Save cleaned JSON
    cleaned_json_path = Path("./Data/extracts/cleaned_dataset.json")
    write_text_data(cleaned_json_path, dataset)

    # Convert to Parquet
    df = pd.DataFrame(dataset)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, "./Data/extracts/documents.parquet")

    print("Data cleaning and conversion to Parquet completed successfully.")
    gc.collect()