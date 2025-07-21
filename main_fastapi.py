# main_fastapi.py
from fastapi import FastAPI
from api.ocr_api import router as ocr_router
from api.layoutlm_api import router as layoutlm_router
from api.fusion_api import router as fusion_router

app = FastAPI()

app.include_router(ocr_router, prefix="/api")
app.include_router(layoutlm_router, prefix="/api")
app.include_router(fusion_router, prefix="/api")
