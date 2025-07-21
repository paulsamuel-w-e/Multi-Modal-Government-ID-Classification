from fastapi import APIRouter, UploadFile, File
import subprocess
import json
from config import HV_PYTHON

router = APIRouter()

@router.post("/layoutlm")
async def run_layoutlm():
        
    try:
        # Run your prediction script
        result = subprocess.run(
            [HV_PYTHON, "pipeline/call_LLMv3.py"],  # Adjust if using `-m` module execution
            capture_output=True,
            text=True,
            timeout=60  # Optional: to prevent hanging
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Try to parse stdout as JSON
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = {"raw_output": stdout}

        return {
            "message": "LayoutLMv3 done",
            "prediction": parsed,
            "stderr": stderr,
            "returncode": result.returncode
        }

    except Exception as e:
        return {
            "message": "LayoutLMv3 failed",
            "error": str(e)
        }
