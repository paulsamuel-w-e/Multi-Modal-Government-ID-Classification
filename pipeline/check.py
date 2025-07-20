import paddle
import paddle.utils as paddle_utils
from paddleocr import PaddleOCR



paddle_utils.run_check()
print("PaddlePaddle version:", paddle.__version__)
print("CUDA available:", paddle.is_compiled_with_cuda())

if paddle.is_compiled_with_cuda():
    print("CUDA device count:", paddle.device.cuda.device_count())
    print("CUDA device name:", paddle.device.cuda.get_device_name(0))
else:
    print("Running on CPU.")

class GlobalContext:
    def __init__(self):
        print("Hold on while we getting up things ready...")
        paddle_utils.run_check()
        print("PaddlePaddle version:", paddle.__version__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("CUDA available:", paddle.is_compiled_with_cuda())

        if paddle.is_compiled_with_cuda():
            print("CUDA device count:", paddle.device.cuda.device_count())
            print("CUDA device name:", paddle.device.cuda.get_device_name(0))
        else:
            print("Running on CPU.")

        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            device="gpu"
        )

        print("OCR initialized...")

# Create singleton
CTX = GlobalContext()
