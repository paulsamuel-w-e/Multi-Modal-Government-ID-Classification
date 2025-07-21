from fastapi import APIRouter, UploadFile, File
import subprocess
import os
from config import OCR_PYTHON

router = APIRouter()

@router.post("/ocr")
async def run_ocr(image: UploadFile = File(...)):
    image_path = f"uploads/{image.filename}"
    os.makedirs("uploads", exist_ok=True)

    with open(image_path, "wb") as f:
        f.write(await image.read())

    # subprocess.run(
    #     [OCR_PYTHON, "-m", "pipeline.preload"],
    #     capture_output=True,
    #     text=True
    # )

    result = subprocess.run(
        [OCR_PYTHON, "-m", "pipeline.extract_ocr"],
        capture_output=True,
        text=True
    )

    return {
        "message": "OCR done",
        "stdout": result.stdout,
        # "stderr": result.stderr,
        "image": image.filename
    }
