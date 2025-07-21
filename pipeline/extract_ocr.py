# src/extract_ocr.py
import os
import gc
import glob
import simplejson
import re
import paddle
from pipeline.preload import CTX

def polygon_to_rect(polygon):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def reorder_keys(d):
    desired_order = ["id", "image_path", "full_text", "words", "bbox", "confidence"]
    return {k: d[k] for k in desired_order if k in d}


def remove_text_data(json_list: list, index_list: list):
    for idx in sorted(index_list, reverse=True):
        if 0 <= idx < len(json_list):
            del json_list[idx]
    return json_list


def clean_ocr_output(doc: dict):
    cleaned_texts = []
    removed_indices = []

    for idx, text in enumerate(doc['full_text']):
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[^\w\s!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', '', text)
        text = re.sub(r'(?<!\w)[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~](?!\w)', '', text)
        text = text.strip()

        if text:
            cleaned_texts.append(text)
        else:
            removed_indices.append(idx)

    cleaned_conf = remove_text_data(doc['confidence'], removed_indices)
    cleaned_bbox = remove_text_data(doc['bbox'], removed_indices)

    doc['full_text'] = " ".join(cleaned_texts).strip()
    doc['words'] = cleaned_texts
    doc['bbox'] = [polygon_to_rect(poly) for poly in cleaned_bbox]
    doc['confidence'] = cleaned_conf

    return reorder_keys(doc)


def extract_ocr_single_image(image_path):
    """
    Extract OCR texts from a single image and return the JSON object.
    
    :param image_path: Path to the input image.
    """
    result = CTX.ocr.predict(image_path)

    if not result:
        print(f"Skipping... OCR didn't capture anything in {image_path}")
        return None

    for res in result:
        text_blocks = res['rec_texts']
        bboxes = [arr.tolist() for arr in res['dt_polys']]
        confidences = res['rec_scores']

    merged_list = list(map(lambda x, y, z: [x, y, z], confidences, bboxes, text_blocks))
    filtered_list = list(filter(lambda x: x[0] >= 0.8, merged_list))

    text_blocks = [item[2] for item in filtered_list]
    bboxes = [item[1] for item in filtered_list]
    confidences = [item[0] for item in filtered_list]

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    doc = {
        "id": base_name,
        "full_text": text_blocks,
        "bbox": bboxes,
        "confidence": confidences,
        "image_path": image_path.replace("\\", "/")
    }

    # Free GPU memory
    paddle.device.cuda.empty_cache()
    gc.collect()

    return doc


# --- Main flow ---
if __name__ == "__main__":
    upload_dir = "uploads"
    image_paths = glob.glob(os.path.join(upload_dir, "*"))

    for image_path in image_paths:
        raw_doc = extract_ocr_single_image(image_path)
        if raw_doc:
            cleaned_doc = clean_ocr_output(raw_doc)

            os.makedirs("temp", exist_ok=True)
            output_path = os.path.join("temp", f"{cleaned_doc['id']}.json")

            with open(output_path, "w", encoding="utf-8") as f:
                simplejson.dump(cleaned_doc, f, ensure_ascii=False, indent=4)

            print(simplejson.dumps(cleaned_doc, ensure_ascii=False, indent=4))