from paddleocr import PaddleOCR


class GlobalContext:
    def __init__(self):
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            device="gpu"
        )

# Create singleton
CTX = GlobalContext()