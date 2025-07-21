from fastapi import APIRouter
import subprocess
import json
from config import HV_PYTHON

router = APIRouter()

@router.post("/fusion")
async def run_fusion():
        
    try:
        result = subprocess.run(
            [HV_PYTHON, "-m", "pipeline.call_fusion_mod"],  # Make sure this module exists and is correct
            capture_output=True,
            text=True,
            timeout=60
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = {"raw_output": stdout}

        return {
            "message": "Fusion model prediction done",
            "prediction": parsed,
            "stderr": stderr,
            "returncode": result.returncode
        }

    except Exception as e:
        return {
            "message": "Fusion model failed",
            "error": str(e)
        }
