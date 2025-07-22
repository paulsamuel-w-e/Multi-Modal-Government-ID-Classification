$workDir = (Get-Location).Path

# === Launch FastAPI in a new terminal ===
$fastapiProc = Start-Process powershell -PassThru -ArgumentList @(
    "-NoExit", 
    "-Command", ". '$workDir\..\env\hvenv\Scripts\Activate.ps1'; uvicorn ..\main_fastapi:app --reload --port 8000"
) -WorkingDirectory $workDir -WindowStyle Normal

# === Launch Streamlit in a new terminal ===
$streamlitProc = Start-Process powershell -PassThru -ArgumentList @(
    "-NoExit", 
    "-Command", ". '$workDir\..\env\hvenv\Scripts\Activate.ps1'; streamlit run ..\streamlit_ui\app.py"
) -WorkingDirectory $workDir -WindowStyle Normal

# === Prompt to stop servers ===
function Ask-Stop {
    do {
        $input = Read-Host "`nDo you want to stop both servers? (y/n)"
        switch ($input.ToLower()) {
            "y" {
                Write-Host "Stopping servers..."

                # Stop child processes
                Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
                Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

                # Also close the terminals (PowerShell windows) that launched them
                if ($fastapiProc -ne $null) {
                    try { Stop-Process -Id $fastapiProc.Id -Force } catch {}
                }
                if ($streamlitProc -ne $null) {
                    try { Stop-Process -Id $streamlitProc.Id -Force } catch {}
                }

                Write-Host "Servers and terminals stopped."
                return
            }
            "n" {
                Write-Host "Servers will continue running..."
            }
            default {
                Write-Host "Invalid input. Please enter y or n."
            }
        }
    } while ($true)
}

Ask-Stop
