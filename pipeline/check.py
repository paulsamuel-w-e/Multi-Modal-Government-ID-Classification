import paddle
import paddle.utils as paddle_utils
from paddleocr import PaddleOCR

print("Hold on while we getting up things ready...")
paddle_utils.run_check()
print("PaddlePaddle version:", paddle.__version__)
print("CUDA available:", paddle.is_compiled_with_cuda())

if paddle.is_compiled_with_cuda():
    print("CUDA device count:", paddle.device.cuda.device_count())
    print("CUDA device name:", paddle.device.cuda.get_device_name(0))
else:
    print("Running on CPU.")