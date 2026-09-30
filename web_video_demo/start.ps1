param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Can Python 3.11. Hay cai Python 3.11 va chay lai.' }
}
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Cai thu vien that bai. Kiem tra Internet.' }
& $python prepare.py
if ($LASTEXITCODE -ne 0) { throw 'Tai model MediaPipe that bai.' }
Write-Host ''
Write-Host "PoseLab: mo http://127.0.0.1:$Port trong trinh duyet." -ForegroundColor Green
Write-Host 'Giu cua so nay mo. Nhan Ctrl+C de dung server.'
& $python -m uvicorn app:app --host 127.0.0.1 --port $Port
