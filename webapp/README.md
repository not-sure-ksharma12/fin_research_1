# CRCL Trading Dashboard

A modern web-based dashboard for visualizing CRCL trading data, including hourly data graphs and detailed trade logs.

## Features

- **Company Selection**: Dropdown to select company ticker (currently CRCL)
- **Date Range Selection**: Start and end date pickers for data filtering
- **Data Type Selection**: Choose between hourly data graphs or trades log
- **Real-time Data**: Fetches data from Excel files and log files
- **Interactive Charts**: Chart.js powered visualizations
- **Responsive Design**: Modern, mobile-friendly UI with Bootstrap 5
- **Summary Statistics**: Key metrics display (total trades, P&L, returns)

## Data Sources

The dashboard reads data from:
- **Hourly Data**: `../scripts/scripts/realtime_output/multi_company_sep19/CRCL_hourly_data.xlsx`
- **Trades Log**: `../scripts/scripts/realtime_output/multi_company_sep19/CRCL_trades.log`

## Installation

1. **Navigate to the webapp directory**:
   ```bash
   cd webapp
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5001
   ```

3. **Use the dashboard**:
   - Select date range using the date pickers
   - Choose data type (Hourly Data Graph or Trades Log)
   - Click "Load Data" to fetch and display information

## Dashboard Components

### Control Panel
- **Company Ticker**: Currently shows CRCL (expandable for other companies)
- **Start Date**: Beginning of data range
- **End Date**: End of data range
- **Data Type**: Choose between hourly data visualization or trades table

### Summary Statistics
- **Total Trades**: Count of all trades in the system
- **Total P&L**: Sum of all profit/loss from completed trades
- **Average Return**: Mean return percentage across all trades
- **Data Points**: Number of hourly data records available

### Data Visualization
- **Hourly Data**: Interactive line charts showing CRCL data over time
- **Trades Log**: Detailed table with all trade information including:
  - Timestamp
  - Action (ENTER/EXIT)
  - Option ID and Strike Price
  - Option Type (Call/Put)
  - Trade Type (BUY/SELL)
  - Price and P&L information

## API Endpoints

- **`/`**: Main dashboard page
- **`/api/data`**: Get data based on parameters (type, start_date, end_date)
- **`/api/summary`**: Get summary statistics

## File Structure

```
webapp/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Dashboard HTML template
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Customization

### Adding New Companies
1. Update the ticker dropdown in `templates/index.html`
2. Modify data file paths in `app.py`
3. Adjust data parsing logic for different file formats

### Adding New Data Types
1. Add new options to the data type dropdown
2. Create corresponding display functions in the JavaScript
3. Update the API endpoint to handle new data types

### Styling Changes
- Modify CSS in the `<style>` section of `index.html`
- Update Bootstrap classes for layout changes
- Customize Chart.js options for different chart styles

## Troubleshooting

### Common Issues

1. **File Not Found Errors**:
   - Ensure data files exist in the specified paths
   - Check file permissions

2. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility

3. **Chart Not Displaying**:
   - Check browser console for JavaScript errors
   - Verify data format matches expected structure

### Debug Mode
The app runs in debug mode by default. Check the terminal for detailed error messages and logging information.

## Dependencies

- **Flask**: Web framework
- **pandas**: Data manipulation and Excel file reading
- **openpyxl**: Excel file support
- **Chart.js**: Interactive charts (loaded via CDN)
- **Bootstrap 5**: Responsive UI framework (loaded via CDN)
- **Font Awesome**: Icons (loaded via CDN)

## Future Enhancements

- Real-time data updates
- Additional chart types (candlestick, volume, etc.)
- Export functionality (PDF, CSV)
- User authentication
- Multi-company support
- Advanced filtering and search
- Performance analytics and backtesting

## Support

For issues or questions, check the terminal logs when running the application in debug mode.
