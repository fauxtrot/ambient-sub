# Vision Pipeline Test Script
# Launches SpacetimeDB, Svelte UI, and Python vision agents in separate windows

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Vision Pipeline Test Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get project root (where this script is located)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SvelteDir = Join-Path $ProjectRoot "web\ambient-subconscious-svelte"

# 1. Start SpacetimeDB
Write-Host "[1/3] Starting SpacetimeDB..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host 'SpacetimeDB Server' -ForegroundColor Green
Write-Host 'Listening on: ws://127.0.0.1:3000' -ForegroundColor Gray
Write-Host 'Module: ambient-listener' -ForegroundColor Gray
Write-Host ''
spacetime start
"@

Start-Sleep -Seconds 3

# 2. Start Svelte dev server
Write-Host "[2/3] Starting Svelte UI..." -ForegroundColor Yellow
if (-not (Test-Path $SvelteDir)) {
    Write-Host "ERROR: Svelte directory not found at: $SvelteDir" -ForegroundColor Red
    exit 1
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host 'Svelte Dev Server' -ForegroundColor Green
Write-Host 'Starting in: $SvelteDir' -ForegroundColor Gray
Write-Host ''
cd '$SvelteDir'
npm run dev
"@

Start-Sleep -Seconds 5

# 3. Start Python vision agents
Write-Host "[3/3] Starting Python Vision Agents..." -ForegroundColor Yellow
Write-Host "NOTE: You'll need to activate your venv in the new window" -ForegroundColor Magenta

Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
Write-Host 'Python Vision Agents' -ForegroundColor Green
Write-Host 'Working directory: $ProjectRoot' -ForegroundColor Gray
Write-Host ''
Write-Host 'IMPORTANT: Activate your venv first!' -ForegroundColor Yellow
Write-Host 'Example: .venv\Scripts\Activate.ps1' -ForegroundColor Gray
Write-Host ''
Write-Host 'Then run one of:' -ForegroundColor White
Write-Host '  python -m ambient_subconscious.main start --agents webcam' -ForegroundColor Gray
Write-Host '  python -m ambient_subconscious.main start --agents screen' -ForegroundColor Gray
Write-Host '  python -m ambient_subconscious.main start --agents webcam,screen' -ForegroundColor Gray
Write-Host ''
cd '$ProjectRoot'
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services running in separate windows:" -ForegroundColor White
Write-Host "  1. SpacetimeDB    - ws://127.0.0.1:3000" -ForegroundColor Gray
Write-Host "  2. Svelte UI      - http://localhost:5173 (typically)" -ForegroundColor Gray
Write-Host "  3. Python Agents  - Manual start (activate venv first)" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. In the Python window, activate your venv" -ForegroundColor Gray
Write-Host "  2. Run: python -m ambient_subconscious.main start --agents webcam,screen" -ForegroundColor Gray
Write-Host "  3. Open browser to Svelte UI (check console for port)" -ForegroundColor Gray
Write-Host "  4. Watch frames being captured and stored!" -ForegroundColor Gray
Write-Host ""
