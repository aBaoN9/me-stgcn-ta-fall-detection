$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python = if (Test-Path -LiteralPath '.venv-streamlit/Scripts/python.exe') { '.venv-streamlit/Scripts/python.exe' } else { 'python' }
    & $python -B -X utf8 -m streamlit run demo_streamlit/app.py --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 50 --server.headless true --browser.gatherUsageStats false --theme.base light --theme.primaryColor '#215bd8' --theme.backgroundColor '#f5f8fe' --theme.secondaryBackgroundColor '#eaf1fd' --theme.textColor '#183354'
    if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
