# 1. Activate OCRenv
& "env\OCRenv\Scripts\Activate.ps1"
Write-Output "`nActivated OCRenv"

# 2. Run first script
python -m pipeline.check
python pipeline\extract_ocr.py

# 3. Deactivate OCRenv
cmd /c "deactivate"
Write-Output "`nDeactivated OCRenv"

# 4. Activate hvenv
& "env\hvenv\Scripts\Activate.ps1"
Write-Output "`nActivated hvenv"

# 5. Run second script
python -m pipeline.call_fusion_mod

# 6. Deactivate hvenv
cmd /c "deactivate"
Write-Output "`nDeactivated hvenv"