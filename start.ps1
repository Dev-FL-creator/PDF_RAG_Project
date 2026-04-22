# PDF RAG Project Auto Launcher

param(
    [string]$ProjectPath = "C:\Users\jinkliu\Desktop\PDF_RAG_Project",
    [string]$NodePath = "C:\Users\jinkliu\Desktop\nodejs-portable\node-v24.8.0-win-x64" 
)

Write-Host "========================================" -ForegroundColor Green
Write-Host "      PDF RAG Project Auto Launcher" -ForegroundColor Green  
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check project path
if (-not (Test-Path $ProjectPath)) {
    Write-Host "Error: Project path does not exist: $ProjectPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Node.js path if specified
if ($NodePath -and -not (Test-Path $NodePath)) {
    Write-Host "Error: Node.js path does not exist: $NodePath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/4] Activating Python virtual environment..." -ForegroundColor Yellow
try {
    & "$ProjectPath\backend\pdf_rag_venv\Scripts\Activate.ps1"
    Write-Host "Success: Python virtual environment activated" -ForegroundColor Green
}
catch {
    Write-Host "Error: Failed to activate virtual environment: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[2/4] Starting backend service..." -ForegroundColor Yellow

# Start backend in new window
$backendArgs = "-NoExit -Command `"cd '$ProjectPath'; & '$ProjectPath\backend\pdf_rag_venv\Scripts\Activate.ps1'; Write-Host 'Backend starting...' -ForegroundColor Cyan; cd backend; python app.py`""
Start-Process powershell -ArgumentList $backendArgs

Write-Host "Success: Backend service command executed" -ForegroundColor Green

Write-Host ""
Write-Host "[3/4] Waiting for backend initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "[4/4] Starting frontend service..." -ForegroundColor Yellow

# Start frontend in new window
if ($NodePath) {
    # Use portable Node.js
    $frontendArgs = "-NoExit -Command `"cd '$ProjectPath\frontend'; `$env:PATH = '$NodePath;' + `$env:PATH; Write-Host 'Frontend starting...' -ForegroundColor Cyan; & '$NodePath\npm.cmd' run dev`""
} else {
    # Use system npm
    $frontendArgs = "-NoExit -Command `"cd '$ProjectPath\frontend'; Write-Host 'Frontend starting...' -ForegroundColor Cyan; npm run dev`""
}
Start-Process powershell -ArgumentList $frontendArgs

Write-Host "Success: Frontend service command executed" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "           Launch Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "- Frontend App: http://localhost:5173" -ForegroundColor White
Write-Host "- Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host "- API Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Note: First startup may take 1-2 minutes" -ForegroundColor Yellow
Write-Host "To stop services, close the respective PowerShell windows" -ForegroundColor Yellow
Write-Host ""

Read-Host "Press Enter to exit launcher"
