# Options Chain Analyzer

This repository contains Python scripts for fetching and analyzing options chain data using the `yfinance` library.

## Files

- `simple_options_example.py` - Your original code with minimal modifications
- `options_chain_analyzer.py` - Enhanced version with better error handling and analysis features
- `requirements.txt` - Required Python packages

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Simple Example (Original Code)

Run the simple example:
```bash
python simple_options_example.py
```

This will:
- Fetch all options data for AAPL
- Save calls and puts to CSV files
- Display sample data

### Enhanced Options Analyzer

Run the enhanced analyzer:
```bash
python options_chain_analyzer.py
```

This will:
- Fetch all options data for AAPL with retry logic
- Perform basic analysis (volume, strike ranges, etc.)
- Save timestamped CSV files
- Display detailed summaries

### Custom Usage

You can modify the ticker symbol in either script:

```python
# Change "AAPL" to any ticker symbol
calls_df, puts_df = get_full_options_chain("TSLA")  # For Tesla
calls_df, puts_df = get_full_options_chain("MSFT")  # For Microsoft
```

## Features

### Simple Example
- Basic options chain fetching
- CSV export
- Error handling for individual expiration dates

### Enhanced Analyzer
- Retry logic for failed requests
- Progress tracking
- Data analysis and summaries
- Timestamped file outputs
- Current stock price display
- Top volume contracts identification

## Output Files

The scripts will create CSV files with the following columns:
- `contractSymbol` - Option contract symbol
- `strike` - Strike price
- `lastPrice` - Last traded price
- `bid` - Current bid price
- `ask` - Current ask price
- `volume` - Trading volume
- `openInterest` - Open interest
- `expirationDate` - Expiration date
- `ticker` - Stock ticker symbol (enhanced version only)

## Notes

- The `yfinance` library fetches data from Yahoo Finance
- Data availability depends on market hours and ticker symbol
- Some tickers may have limited or no options data
- Consider rate limiting for large-scale data collection

## Troubleshooting

If you encounter errors:
1. Check your internet connection
2. Verify the ticker symbol is valid
3. Ensure the market is open (for real-time data)
4. Try running with a different ticker symbol 