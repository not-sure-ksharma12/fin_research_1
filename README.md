# Bloomberg Options Analysis Project

A comprehensive Python-based financial analysis toolkit for fetching Bloomberg option data and performing advanced pricing model comparisons including Black-Scholes and Heston stochastic volatility models.

## 📁 Project Structure

```
fin_research/
├── scripts/                          # Main analysis scripts
│   ├── fetch.py                     # Bloomberg data fetcher
│   ├── black_scholes_analysis.py    # Black-Scholes model analysis
│   ├── comprehensive_analysis.py     # Heston model & comprehensive analysis
│   └── run_all_analysis.ps1        # PowerShell pipeline runner
├── analysis/                         # Output files (auto-generated)
│   ├── bloomberg_options_top50.xlsx
│   ├── black_scholes_quantlib_output.xlsx
│   ├── heston_vs_market_direct.xlsx
│   └── bs_heston_market_comparison.xlsx
├── venv/                            # Python virtual environment
├── blpapi_cpp_3.25.3.1/            # Bloomberg C++ API
├── blpapi-3.25.3/                  # Bloomberg Python API
└── requirements.txt                 # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Bloomberg Terminal with API access
- Python 3.8+
- PowerShell (for Windows)

### Installation

1. **Clone/Download the project**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Bloomberg API:**
   - Ensure Bloomberg Terminal is running
   - Verify API connectivity (see `BLOOMBERG_API_SETUP.md`)

### Running the Analysis

**Option 1: Run the complete pipeline**
```powershell
.\scripts\run_all_analysis.ps1
```

**Option 2: Run individual scripts**
```bash
# Step 1: Fetch Bloomberg data
python scripts/fetch.py

# Step 2: Black-Scholes analysis
python scripts/black_scholes_analysis.py

# Step 3: Comprehensive Heston analysis
python scripts/comprehensive_analysis.py
```

## 📊 Analysis Components

### 1. Data Fetching (`fetch.py`)
- **Purpose:** Retrieves real-time option data from Bloomberg
- **Input:** None (uses hardcoded tickers: NVDA, SOUN, BBAI, CRCL, AVGO, AMD)
- **Output:** `analysis/bloomberg_options_top50.xlsx`
- **Features:**
  - Fetches option chains for multiple tickers
  - Retrieves Treasury yields for risk-free rate interpolation
  - Calculates interpolated risk-free rates based on time to expiry
  - Applies conditional formatting to output Excel files

### 2. Black-Scholes Analysis (`black_scholes_analysis.py`)
- **Purpose:** Compares Black-Scholes model prices with market data
- **Input:** `analysis/bloomberg_options_top50.xlsx`
- **Output:** `analysis/black_scholes_quantlib_output.xlsx`
- **Features:**
  - Uses QuantLib for accurate option pricing
  - Calculates all Greeks (Delta, Gamma, Theta, Vega, Rho)
  - Compares model vs market prices and Greeks
  - Includes implied volatility calculations

### 3. Comprehensive Analysis (`comprehensive_analysis.py`)
- **Purpose:** Implements Heston stochastic volatility model and comprehensive comparisons
- **Input:** Bloomberg data and Black-Scholes results
- **Output:** 
  - `analysis/heston_vs_market_direct.xlsx` (Heston vs Market)
  - `analysis/bs_heston_market_comparison.xlsx` (BS vs Heston vs Market)
- **Features:**
  - Heston model implementation with QuantLib
  - Three-way model comparison (Black-Scholes, Heston, Market)
  - Detailed difference analysis for all Greeks
  - Conditional formatting for easy visualization

## 📈 Output Files

### `bloomberg_options_top50.xlsx`
- Raw Bloomberg option data
- Includes market prices, bid/ask, Greeks, implied volatility
- Risk-free rates interpolated from Treasury yields

### `black_scholes_quantlib_output.xlsx`
- Black-Scholes model prices and Greeks
- Comparison with market data
- Price and Greek differences
- Implied volatility analysis

### `heston_vs_market_direct.xlsx`
- Heston model vs market data comparison
- Stochastic volatility model results
- Difference analysis for all metrics

### `bs_heston_market_comparison.xlsx`
- Three-way comparison (Black-Scholes vs Heston vs Market)
- Comprehensive difference analysis
- Model performance evaluation

## 🔧 Configuration

### Tickers Analyzed
Currently configured for: `NVDA`, `SOUN`, `BBAI`, `CRCL`, `AVGO`, `AMD`

To modify tickers, edit `scripts/fetch.py`:
```python
tickers = ['NVDA', 'SOUN', 'BBAI', 'CRCL', 'AVGO', 'AMD']
```

### Heston Model Parameters
Default parameters in `scripts/comprehensive_analysis.py`:
- `v0 = 0.04` (initial variance)
- `kappa = 2.0` (mean reversion speed)
- `theta = 0.04` (long-term variance)
- `sigma_v = 0.3` (volatility of volatility)
- `rho = -0.5` (correlation)

## 🛠️ Troubleshooting

### Common Issues

1. **Bloomberg API Connection Error**
   - Ensure Bloomberg Terminal is running
   - Check network connectivity
   - Verify API permissions

2. **Missing Dependencies**
   ```bash
   pip install pandas numpy openpyxl QuantLib blpapi
   ```

3. **PowerShell Execution Policy**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **File Path Issues**
   - Ensure you're running from the project root directory
   - Check that `analysis/` folder exists (auto-created by scripts)

### Debug Mode
Add debug prints by modifying the scripts or check the console output for detailed information about the data fetching and processing steps.

## 📚 Technical Details

### Models Implemented

1. **Black-Scholes Model**
   - Assumes constant volatility
   - Uses QuantLib's analytic European engine
   - Includes dividend yield support

2. **Heston Model**
   - Stochastic volatility model
   - Mean-reverting variance process
   - Captures volatility smile/skew effects

### Data Sources
- **Bloomberg Terminal:** Real-time option data, Treasury yields
- **QuantLib:** Financial modeling library for option pricing
- **Interpolation:** Linear interpolation of risk-free rates from Treasury curve

### Key Features
- **Risk-Free Rate Interpolation:** Uses Treasury yields for accurate discounting
- **Conditional Formatting:** Excel outputs with color-coded differences
- **Comprehensive Greeks:** Delta, Gamma, Theta, Vega, Rho calculations
- **Model Comparison:** Statistical analysis of model fit vs market data

## 🤝 Contributing

To extend this project:
1. Add new models in separate scripts
2. Update the PowerShell pipeline to include new analyses
3. Maintain the folder structure (scripts/, analysis/)
4. Add appropriate error handling and logging

## 📄 License

This project is for educational and research purposes. Ensure compliance with Bloomberg API terms of service and any applicable financial regulations.

---

**Note:** This analysis uses real market data and sophisticated financial models. Results should be validated and used responsibly in any trading or investment decisions. 