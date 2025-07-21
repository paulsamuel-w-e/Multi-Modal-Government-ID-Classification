# src/predict_single_image.py

import os
import glob
import torch
import simplejson as json
from PIL import Image
from transformers import AutoProcessor, LayoutLMv3ForSequenceClassification

def load_cleaned_ocr(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_layoutlmv3(image_path, json_path, model_dir="./saved_model"):
    # --- Load processor and model ---
    processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Load cleaned OCR ---
    doc = load_cleaned_ocr(json_path)
    image = Image.open(image_path).convert("RGB")

    # --- Prepare input ---
    encoding = processor(
        image,
        doc["words"],
        boxes=doc["bbox"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # --- Predict ---
    with torch.no_grad():
        outputs = model(**encoding)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()

    # --- Get label name ---
    id2label = model.config.id2label
    label = id2label[predicted_class]

    result = {
        "file": image_path,
        "predicted_label": label
    }
    print(json.dumps(result))

    return label

# --- Main ---
if __name__ == "__main__":
    upload_dir = "uploads"
    image_paths = glob.glob(os.path.join(upload_dir, "*"))

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join("temp", f"{base_name}.json")

        if os.path.exists(json_path):
            predict_layoutlmv3(image_path, json_path)
        else:
            print(f"⚠️ Skipping {image_path} — No corresponding JSON found at {json_path}")