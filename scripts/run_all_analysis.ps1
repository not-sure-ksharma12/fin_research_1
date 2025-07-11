# PowerShell script to run all analysis sequentially
# Run this script from the fin_research directory or any location

Write-Host "Starting comprehensive options analysis pipeline..." -ForegroundColor Green

# Step 1: Fetch Bloomberg data and create initial options file
Write-Host "Step 1: Fetching Bloomberg options data..." -ForegroundColor Yellow
python "$PSScriptRoot/fetch.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in fetch.py" -ForegroundColor Red
    exit 1
}
Write-Host "Bloomberg data fetched successfully" -ForegroundColor Green

# Step 2: Run Black-Scholes analysis
Write-Host "Step 2: Running Black-Scholes analysis..." -ForegroundColor Yellow
python "$PSScriptRoot/black_scholes_analysis.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in black_scholes_analysis.py" -ForegroundColor Red
    exit 1
}
Write-Host "Black-Scholes analysis completed" -ForegroundColor Green

# Step 3: Run comprehensive Heston analysis
Write-Host "Step 3: Running comprehensive Heston analysis..." -ForegroundColor Yellow
python "$PSScriptRoot/comprehensive_analysis.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in comprehensive_analysis.py" -ForegroundColor Red
    exit 1
}
Write-Host "Comprehensive Heston analysis completed" -ForegroundColor Green

# Step 4: Run model comparison summary (if matplotlib is available)
Write-Host "Step 4: Running model comparison summary..." -ForegroundColor Yellow
try {
    python "$PSScriptRoot/model_comparison_summary.py"
    Write-Host "Model comparison summary completed" -ForegroundColor Green
} catch {
    Write-Host "Model comparison summary skipped (matplotlib not available)" -ForegroundColor Yellow
}

# Summary of generated files
Write-Host "`nAnalysis Complete! Generated files:" -ForegroundColor Cyan
Write-Host "   analysis/bloomberg_options_top50.xlsx (Original Bloomberg data)" -ForegroundColor White
Write-Host "   analysis/black_scholes_quantlib_output.xlsx (BS vs Market analysis)" -ForegroundColor White
Write-Host "   analysis/heston_vs_market_direct.xlsx (Heston vs Market analysis)" -ForegroundColor White
Write-Host "   analysis/bs_heston_market_comparison.xlsx (BS vs Heston vs Market)" -ForegroundColor White

# Check if summary files exist
if (Test-Path "analysis/heston_vs_market_summary.xlsx") {
    Write-Host "   analysis/heston_vs_market_summary.xlsx (Error summary)" -ForegroundColor White
}
if (Test-Path "analysis/model_fit_counts.xlsx") {
    Write-Host "   analysis/model_fit_counts.xlsx (Model fit counts)" -ForegroundColor White
}

Write-Host "`nAll analysis completed successfully!" -ForegroundColor Green
Write-Host "You can now analyze the Excel files for model comparisons." -ForegroundColor Cyan 