# Bloomberg API Installation Script
# Run this AFTER installing Microsoft C++ Build Tools

Write-Host "Installing Bloomberg API..." -ForegroundColor Green

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Set environment variable
$env:BLPAPI_ROOT = "C:\Users\ksharma12\fin_research\blpapi_cpp_3.25.3.1"

Write-Host "BLPAPI_ROOT set to: $env:BLPAPI_ROOT" -ForegroundColor Yellow

# Navigate to blpapi directory and install
Set-Location "blpapi-3.25.3"
pip install .

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "You can now run: python nvda_options_fetcher.py" -ForegroundColor Cyan 