# src/ocr_extraction.py
import os
import gc
import simplejson
import argparse
from tqdm import tqdm
import paddle
from paddleocr import PaddleOCR


def extract_ocr(image_dir, output_dir):
    """
        Extract OCR texts from images in the specified directory and save them in JSON format.
        - param image_dir: Directory containing images organized in class folders.  
        - param output_dir: Directory where the OCR results will be saved.
        - return: None, saves JSON files with OCR results in the output directory under 'ocr_texts' subdirectory.
    """
    ocr = PaddleOCR(device="gpu")
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )
    
    text_root = os.path.join(output_dir, "ocr_texts")
    os.makedirs(text_root, exist_ok=True)
    
    loop = tqdm(os.listdir(image_dir), desc="Extracting Texts from Classes")

    for class_folder in loop:
        class_path = os.path.join(image_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        text_class_dir = os.path.join(text_root, class_folder)
        os.makedirs(text_class_dir, exist_ok=True)


        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            result = ocr.predict(img_path)
            if not result:
                print("Skipping... {ocr doesn't captured anything!}")
                continue

            for res in result:
                text_blocks = res['rec_texts']
                bboxes = [arr.tolist() for arr in res['dt_polys']]
                confidences = res['rec_scores']
           
            merged_list = list(map(lambda x, y, z: [x, y, z,], confidences, bboxes, text_blocks))
            filtered_list = list(filter(lambda x: x[0] >= 0.8, merged_list))

            text_blocks = [item[2] for item in filtered_list]
            bboxes = [item[1] for item in filtered_list]
            confidences = [item[0] for item in filtered_list]

            base_name = os.path.splitext(img_file)[0]

            doc = {
                "id" : base_name,
                "full_text" : text_blocks,
                "bbox" : bboxes,
                "confidence" : confidences,
                "label" : class_folder,
                "image_path" : img_path.replace("\\","/")
            }
            
            # Save OCR texts
            text_file = os.path.join(text_class_dir, f"{base_name}.json")
            with open(text_file, 'w', encoding='utf-8') as f:
                simplejson.dump(doc, f, ensure_ascii=False, indent=4)

        
        # Free up memory
        paddle.device.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./data/raw_images", help="Input image directory")
    parser.add_argument("--output_dir", default="./data/extracts", help="OCR Output directory")
    args = parser.parse_args()
    extract_ocr(args.image_dir, args.output_dir)