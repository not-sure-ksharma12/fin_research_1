from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import math
from datetime import datetime, timedelta
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data file mappings per ticker
DATA_BASE_DIR = "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19"
TICKER_TO_JSON = {
    "CRCL": f"{DATA_BASE_DIR}/CRCL_hourly_data.json",
    "AMD": f"{DATA_BASE_DIR}/AMD_hourly_data.json",
    "AVGO": f"{DATA_BASE_DIR}/AVGO_hourly_data.json",
    "BBAI": f"{DATA_BASE_DIR}/BBAI_hourly_data.json",
    "NVDA": f"{DATA_BASE_DIR}/NVDA_hourly_data.json",
    "SLDB": f"{DATA_BASE_DIR}/SLDB_hourly_data.json",
    "SOFI": f"{DATA_BASE_DIR}/SOFI_hourly_data.json",
    "SOUN": f"{DATA_BASE_DIR}/SOUN_hourly_data.json",
    "TSLA": f"{DATA_BASE_DIR}/TSLA_hourly_data.json",
}
TICKER_TO_LOG = {
    "CRCL": f"{DATA_BASE_DIR}/CRCL_trades.log",
    # Other tickers can be added here if/when trade logs are available
}

# Global storage for data per ticker
json_cache = {}
current_simulation_results_by_ticker = {}

def get_selected_ticker() -> str:
    ticker = (request.args.get('ticker') or 'CRCL').upper()
    if ticker not in TICKER_TO_JSON:
        logger.warning(f"Unknown ticker '{ticker}', defaulting to CRCL")
        ticker = 'CRCL'
    return ticker

def load_json_data(ticker: str):
    """Load JSON data for a given ticker (cached)."""
    global json_cache
    if ticker in json_cache:
        return json_cache[ticker]
    json_file_path = TICKER_TO_JSON.get(ticker)
    if not json_file_path:
        logger.error(f"No JSON mapping defined for ticker {ticker}")
        return None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_cache[ticker] = json.load(f)
        logger.info(f"Loaded JSON for {ticker} with {len(json_cache[ticker].get('sheets', {}))} sheets")
        return json_cache[ticker]
    except Exception as e:
        logger.error(f"Error loading JSON data for {ticker}: {e}")
        return None

def get_excel_sheet_names(ticker: str):
    """Get sheet names from JSON data for ticker"""
    data = load_json_data(ticker)
    if data and 'sheets' in data:
        return list(data['sheets'].keys())
    return []

def load_hourly_data(sheet_name=None, ticker: str = 'CRCL'):
    """Load hourly data from JSON for a specific sheet and ticker"""
    data = load_json_data(ticker)
    if not data or 'sheets' not in data:
        return pd.DataFrame()
    
    if sheet_name is None:
        # Return first sheet if none specified
        sheet_name = list(data['sheets'].keys())[0] if data['sheets'] else None
    
    if sheet_name not in data['sheets']:
        logger.warning(f"Sheet '{sheet_name}' not found in JSON data")
        return pd.DataFrame()
    
    sheet_data = data['sheets'][sheet_name]
    
    # Convert JSON data back to DataFrame
    df = pd.DataFrame(sheet_data['data'])
    
    # Convert data types back to appropriate types
    for col, dtype_str in sheet_data['data_types'].items():
        if col in df.columns:
            if 'int' in dtype_str:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif 'float' in dtype_str:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif 'datetime' in dtype_str:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    logger.info(f"Loaded {len(df)} rows from sheet '{sheet_name}' for {ticker}")
    return df

def clean_data_for_json(data):
    """Clean data to make it JSON-serializable by replacing NaN values"""
    if isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif pd.isna(data):
        return None
    elif isinstance(data, (int, float)) and (pd.isna(data) or math.isnan(data)):
        return None
    else:
        return data

def load_trades_log(ticker: str = 'CRCL'):
    """Load trades log for ticker from the log file"""
    log_file_path = TICKER_TO_LOG.get(ticker)
    
    if not log_file_path or not os.path.exists(log_file_path):
        logger.warning(f"Trades log file not found: {log_file_path}")
        return []
    
    trades = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse the log line format
                    parts = line.split(' | ')
                    if len(parts) >= 8:
                        trade = {
                            'timestamp': parts[0].strip(),
                            'action': parts[1].strip(),
                            'strike_price': float(parts[2].strip()),
                            'option_type': parts[3].strip(),
                            'market_price': float(parts[4].strip()),
                            'heston_price': float(parts[5].strip()),
                            'pnl': float(parts[6].strip()) if parts[6].strip() != 'N/A' else 0.0,
                            'return_pct': float(parts[7].strip().rstrip('%')) if parts[7].strip() != 'N/A' else 0.0
                        }
                        trades.append(trade)
                    else:
                        logger.warning(f"Line {line_num}: Insufficient parts in log line: {line}")
                except Exception as e:
                    logger.warning(f"Line {line_num}: Error parsing log line '{line}': {e}")
                    continue
        
        logger.info(f"Successfully loaded {len(trades)} trades from log file")
        return trades
    except Exception as e:
        logger.error(f"Error reading trades log file: {e}")
        return []

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get data based on parameters"""
    try:
        # Get query parameters
        data_type = request.args.get('type', 'hourly')  # 'hourly' or 'trades'
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        ticker = get_selected_ticker()
        if data_type == 'hourly':
            # Get sheet name from query parameters if provided
            sheet_name = request.args.get('sheet')
            
            if sheet_name:
                # Load data from specific sheet for date/time view
                logger.info(f"Loading hourly data from specific sheet: {sheet_name}")
                df = load_hourly_data(sheet_name, ticker)
            else:
                # Load data from the first available sheet to get strikes and dates for dropdowns
                logger.info("Loading hourly data from first sheet for dropdowns")
                df = load_hourly_data(ticker=ticker)
            
            if not df.empty:
                # Filter by date if provided
                if start_date and end_date:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    # Assuming there's a timestamp column, filter by date
                    # This will need to be adjusted based on actual column names
                    pass
                
                # Convert to JSON-serializable format and clean NaN values
                data = df.to_dict('records')
                cleaned_data = clean_data_for_json(data)
                
                if sheet_name:
                    return jsonify({'success': True, 'data': cleaned_data, 'type': 'hourly', 'sheet': sheet_name})
                else:
                    return jsonify({'success': True, 'data': cleaned_data, 'type': 'hourly'})
            else:
                if sheet_name:
                    return jsonify({'success': False, 'error': f'No hourly data available for sheet: {sheet_name}'})
                else:
                    return jsonify({'success': False, 'error': 'No hourly data available'})
        
        elif data_type == 'trades':
            trades = load_trades_log(ticker)
            if trades:
                # Filter by date if provided
                if start_date and end_date:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    filtered_trades = []
                    for trade in trades:
                        trade_dt = datetime.fromisoformat(trade['timestamp'])
                        if start_dt <= trade_dt <= end_dt:
                            filtered_trades.append(trade)
                    trades = filtered_trades
                
                logger.info(f"Sending {len(trades)} trades to frontend. Sample trade: {trades[0] if trades else 'No trades'}")
                return jsonify({'success': True, 'data': trades, 'type': 'trades'})
            else:
                return jsonify({'success': False, 'error': 'No trades data available'})
        
        else:
            return jsonify({'success': False, 'error': 'Invalid data type'})
    
    except Exception as e:
        logger.error(f"Error in get_data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sheets')
def get_sheets():
    """API endpoint to get available Excel sheet names"""
    try:
        ticker = get_selected_ticker()
        sheet_names = get_excel_sheet_names(ticker)
        return jsonify({'success': True, 'sheets': sheet_names})
    except Exception as e:
        logger.error(f"Error in get_sheets: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strike-data')
def get_strike_data():
    """API endpoint to get data for a specific strike price across all sheets"""
    try:
        strike = request.args.get('strike')
        if not strike:
            return jsonify({'success': False, 'error': 'Strike price is required'})
        
        try:
            strike_float = float(strike)
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid strike price'})
        
        ticker = get_selected_ticker()
        logger.info(f"Loading data for strike {strike_float} across all sheets for {ticker}")
        
        # Get all available sheets
        sheet_names = get_excel_sheet_names(ticker)
        all_strike_data = []
        
        for sheet_name in sheet_names:
            try:
                # Load data from each sheet
                df = load_hourly_data(sheet_name, ticker)
                if not df.empty:
                    # Filter data for the specific strike
                    strike_data = df[df['Strike'] == strike_float]
                    if not strike_data.empty:
                        # Add sheet information to each row
                        for _, row in strike_data.iterrows():
                            row_dict = row.to_dict()
                            row_dict['sheet_name'] = sheet_name
                            # Extract hour and date from sheet name for sorting
                            if 'Hour_' in sheet_name and '_2025-' in sheet_name:
                                try:
                                    hour_part = sheet_name.split('_')[1]  # Hour number
                                    date_part = sheet_name.split('_')[2]  # Date
                                    row_dict['hour'] = int(hour_part)
                                    row_dict['date'] = date_part
                                    # Create timestamp for sorting
                                    row_dict['timestamp'] = f"{date_part} {hour_part}:00:00"
                                except (IndexError, ValueError):
                                    row_dict['hour'] = 0
                                    row_dict['date'] = 'unknown'
                                    row_dict['timestamp'] = 'unknown'
                            else:
                                row_dict['hour'] = 0
                                row_dict['date'] = 'unknown'
                                row_dict['timestamp'] = 'unknown'
                            
                            all_strike_data.append(row_dict)
                        
                        logger.info(f"Found {len(strike_data)} rows for strike {strike_float} in sheet {sheet_name}")
                    else:
                        logger.info(f"No data found for strike {strike_float} in sheet {sheet_name}")
                else:
                    logger.warning(f"Sheet {sheet_name} is empty or could not be loaded")
                    
            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name}: {e}")
                continue
        
        if all_strike_data:
            # Sort by timestamp for chronological order
            try:
                all_strike_data.sort(key=lambda x: x['timestamp'] if x['timestamp'] != 'unknown' else '0')
            except Exception as e:
                logger.warning(f"Could not sort data by timestamp: {e}")
            
            logger.info(f"Total data points found for strike {strike_float}: {len(all_strike_data)}")
            
            # Clean data for JSON serialization
            cleaned_data = clean_data_for_json(all_strike_data)
            return jsonify({
                'success': True, 
                'data': cleaned_data, 
                'strike': strike_float,
                'total_points': len(cleaned_data),
                'sheets_processed': len(sheet_names)
            })
        else:
            return jsonify({
                'success': False, 
                'error': f'No data found for strike price {strike_float} across any sheets'
            })
            
    except Exception as e:
        logger.error(f"Error in get_strike_data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/summary')
def get_summary():
    """API endpoint to get summary statistics"""
    try:
        ticker = get_selected_ticker()
        trades = load_trades_log(ticker)
        hourly_data = load_hourly_data(ticker=ticker)
        
        # Calculate summary statistics
        total_trades = len(trades)
        enter_trades = len([t for t in trades if t['action'] == 'ENTER'])
        exit_trades = len([t for t in trades if t['action'] == 'EXIT'])
        
        # Calculate PnL statistics
        pnl_values = [t['pnl'] for t in trades if t['pnl'] is not None and t['pnl'] != 0.0]
        total_pnl = sum(pnl_values) if pnl_values else 0
        avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
        
        # Calculate return statistics
        return_values = [t['return_pct'] for t in trades if t['return_pct'] is not None and t['return_pct'] != 0.0]
        avg_return = sum(return_values) / len(return_values) if return_values else 0
        
        summary = {
            'total_trades': total_trades,
            'enter_trades': enter_trades,
            'exit_trades': exit_trades,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'avg_return': round(avg_return, 2),
            'hourly_data_points': len(hourly_data) if not hourly_data.empty else 0
        }
        
        return jsonify({'success': True, 'summary': summary})
    
    except Exception as e:
        logger.error(f"Error in get_summary: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sliding-window-simulation')
def sliding_window_simulation():
    """API endpoint for sliding window trading simulation"""
    try:
        ticker = get_selected_ticker()
        start_sheet = request.args.get('start_sheet')
        end_sheet = request.args.get('end_sheet')
        
        if not start_sheet or not end_sheet:
            return jsonify({'success': False, 'error': 'Start and end sheet names are required'})
        
        logger.info(f"Running sliding window simulation from {start_sheet} to {end_sheet}")
        
        # Get all available sheets
        sheet_names = get_excel_sheet_names(ticker)
        
        # Find the indices of start and end sheets
        try:
            start_idx = sheet_names.index(start_sheet)
            end_idx = sheet_names.index(end_sheet)
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid sheet names provided'})
        
        # Ensure start is before end
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        # Get sheets in the window
        window_sheets = sheet_names[start_idx:end_idx + 1]
        logger.info(f"Simulation window: {len(window_sheets)} sheets from {start_sheet} to {end_sheet}")
        
        # Run simulation
        simulation_results = run_trading_simulation(window_sheets, ticker)
        
        # Clean the simulation results to ensure JSON serialization
        cleaned_simulation_results = clean_data_for_json(simulation_results)
        
        # Store the results globally for use in label dropdowns
        global current_simulation_results_by_ticker
        current_simulation_results_by_ticker[ticker] = cleaned_simulation_results
        
        return jsonify({
            'success': True,
            'data': cleaned_simulation_results,
            'window_sheets': window_sheets,
            'start_sheet': start_sheet,
            'end_sheet': end_sheet
        })
        
    except Exception as e:
        logger.error(f"Error in sliding window simulation: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trade-labels')
def get_trade_labels():
    """Get trade data organized by labels (A, B, C, D) for graph generation"""
    try:
        ticker = get_selected_ticker()
        global current_simulation_results_by_ticker
        current_simulation = current_simulation_results_by_ticker.get(ticker)
        if current_simulation is None:
            return jsonify({'success': False, 'error': 'No simulation has been run yet. Please run a simulation first.'})
        
        if 'trade_history' not in current_simulation:
            return jsonify({'success': False, 'error': 'No trade data available in current simulation'})
        
        # Organize trades by label
        trades_by_label = {'A': [], 'B': [], 'C': [], 'D': []}
        
        for trade in current_simulation['trade_history']:
            label = trade.get('trade_label', 'X')
            if label in trades_by_label:
                trades_by_label[label].append(trade)
        
        # Create dropdown options for each label
        label_options = {}
        for label, trades in trades_by_label.items():
            options = []
            for trade in trades:
                if trade['action'] == 'ENTER':
                    # Create option: "Strike $X - DateTime"
                    strike = trade['strike_price']
                    timestamp = trade['timestamp']
                    option_value = f"{strike}_{timestamp}"
                    option_text = f"Strike ${strike} - {timestamp}"
                    options.append({
                        'value': option_value,
                        'text': option_text,
                        'trade_data': trade
                    })
            label_options[label] = options
        
        return jsonify({
            'success': True,
            'label_options': label_options
        })
        
    except Exception as e:
        logger.error(f"Error getting trade labels: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/current-simulation')
def get_current_simulation():
    """Get the current simulation results for populating label dropdowns"""
    ticker = get_selected_ticker()
    global current_simulation_results_by_ticker
    current_simulation_results = current_simulation_results_by_ticker.get(ticker)
    if current_simulation_results is None:
        return jsonify({'success': False, 'error': 'No simulation has been run yet'})
    
    # Extract only the entry trades for each label
    label_entries = {}
    for trade in current_simulation_results.get('trade_history', []):
        if trade.get('action') == 'ENTER':
            label = trade.get('trade_label', 'X')
            if label not in label_entries:
                label_entries[label] = []
            
            # Create option for dropdown
            option = {
                'value': f"{trade.get('strike_price')}_{trade.get('timestamp')}",
                'text': f"Strike ${trade.get('strike_price')} - {trade.get('timestamp')}",
                'trade_data': trade
            }
            label_entries[label].append(option)
    
    return jsonify({
        'success': True,
        'label_entries': label_entries
    })

@app.route('/api/label-graph-data')
def get_label_graph_data():
    """Get graph data for a specific label and strike-datetime combination"""
    try:
        ticker = get_selected_ticker()
        label = request.args.get('label')
        strike_datetime = request.args.get('strike_datetime')
        
        if not label or not strike_datetime:
            return jsonify({'success': False, 'error': 'Label and strike_datetime are required'})
        
        logger.info(f"Requested label: {label}, strike_datetime: {strike_datetime}")
        
        # Parse strike_datetime (format: "strike_timestamp")
        parts = strike_datetime.split('_', 1)
        if len(parts) != 2:
            return jsonify({'success': False, 'error': 'Invalid strike_datetime format'})
        
        strike = float(parts[0])
        timestamp = parts[1]
        logger.info(f"Parsed strike: {strike}, timestamp: {timestamp}")
        
        # Get simulation results from stored results
        global current_simulation_results_by_ticker
        current_simulation_results = current_simulation_results_by_ticker.get(ticker)
        if current_simulation_results is None:
            return jsonify({'success': False, 'error': 'No simulation has been run yet. Please run a simulation first.'})
        
        if 'trade_history' not in current_simulation_results:
            return jsonify({'success': False, 'error': 'No trade data available in current simulation'})
        
        # Parse the selected trade timestamp to find the specific trade
        selected_trade_timestamp = timestamp  # This is the timestamp from the dropdown selection
        
        # Debug: Log the trades found for this label
        trade_history = current_simulation_results['trade_history']
        logger.info(f"All trades in simulation: {[(t.get('trade_label'), t.get('action'), t.get('strike_price')) for t in trade_history]}")
        label_trades = [trade for trade in trade_history if trade.get('trade_label') == label]
        logger.info(f"Found {len(label_trades)} trades for label {label}: {label_trades}")
        
        # Find the specific trade that matches the selected timestamp
        selected_trade = None
        for trade in label_trades:
            if trade.get('timestamp') == selected_trade_timestamp:
                selected_trade = trade
                break
        
        if selected_trade:
            logger.info(f"Found selected trade for label {label}: {selected_trade}")
        else:
            logger.warning(f"No trade found for label {label} with timestamp {selected_trade_timestamp}")
        
        # Find ENTER and matching EXIT trades for this specific selection
        # The dropdown is built from ENTER trades, so selected_trade should be the ENTER
        enter_trade = None
        exit_trade = None

        if selected_trade and selected_trade.get('action') == 'ENTER':
            enter_trade = selected_trade
            selected_key = selected_trade.get('trade_key')
            # Match EXIT using the same trade_key to avoid label reuse ambiguity
            if selected_key:
                for trade in label_trades:
                    if trade.get('action') == 'EXIT' and trade.get('trade_key') == selected_key:
                        exit_trade = trade
                        break

        # Fallback: if for some reason selected trade wasn't found or key missing,
        # pick ENTER/EXIT by strike within this label (least preferred)
        if enter_trade is None:
            for trade in label_trades:
                if trade.get('action') == 'ENTER' and trade.get('strike_price') == strike:
                    enter_trade = trade
                    break
        if exit_trade is None and enter_trade is not None:
            key = enter_trade.get('trade_key')
            if key:
                for trade in label_trades:
                    if trade.get('action') == 'EXIT' and trade.get('trade_key') == key:
                        exit_trade = trade
                        break
            # If still not found, take the first EXIT with same strike
            if exit_trade is None:
                for trade in label_trades:
                    if trade.get('action') == 'EXIT' and trade.get('strike_price') == strike:
                        exit_trade = trade
                        break

        logger.info(f"Label {label}, Strike ${strike}: ENTER trade: {enter_trade}, EXIT trade: {exit_trade}")
        if not enter_trade:
            logger.warning(f"No ENTER trade found for label {label}, strike ${strike}")
        if not exit_trade:
            logger.warning(f"No EXIT trade found for label {label}, strike ${strike}")
        
        # Load all sheet data to find the trade period
        all_data = []
        
        # Get sheet names from the stored simulation results
        sheet_names = get_excel_sheet_names(ticker)
        if not sheet_names:
            return jsonify({'success': False, 'error': 'No sheets available'})
        
        # Debug: Check what columns are available in the first sheet
        first_df = load_hourly_data(sheet_names[0], ticker)
        if not first_df.empty:
            logger.info(f"Available columns in Excel data: {list(first_df.columns)}")
            logger.info(f"Sample row: {first_df.iloc[0].to_dict()}")
            
            # Check specifically for Last heston column
            last_heston_cols = [col for col in first_df.columns if 'heston' in col.lower()]
            logger.info(f"Columns containing 'heston': {last_heston_cols}")
            
            if 'Last heston' in first_df.columns:
                sample_last_heston = first_df['Last heston'].iloc[0]
                logger.info(f"Sample 'Last heston' value: {sample_last_heston} (type: {type(sample_last_heston)})")
        
        for sheet_name in sheet_names:
            df = load_hourly_data(sheet_name, ticker)
            if not df.empty:
                # Filter for the specific strike price
                strike_data = df[df['Strike'] == strike].copy()
                if not strike_data.empty:
                    # Add sheet info for timestamp
                    hour, date = extract_hour_date_from_sheet(sheet_name)
                    strike_data['sheet_name'] = sheet_name
                    strike_data['hour'] = hour
                    strike_data['date'] = date
                    all_data.append(strike_data)
        
        if not all_data:
            return jsonify({'success': False, 'error': f'No data found for strike ${strike}'})
        
        # Combine all data and sort by timestamp
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['date', 'hour'])
        
        # Create graph data
        graph_data = {
            'labels': [],  # Time labels
            'heston_prices': [],
            'last_heston': [],
            'last_traded': [],
            'entry_points': [],  # Entry trade markers
            'exit_points': []    # Exit trade markers
        }
        
        # Process each data point
        for _, row in combined_data.iterrows():
            time_label = f"{row['hour']:02d}:00 - {row['date']}"
            graph_data['labels'].append(time_label)
            graph_data['heston_prices'].append(row['Heston_Price'])
            
            # Handle missing Last_heston column gracefully
            # Check for various possible column names and handle pandas column processing
            last_heston = None
            
            # Try different possible column names (pandas might process spaces differently)
            possible_columns = ['Last_heston', 'Last heston', 'last_heston', 'Last Heston', 'LastHeston']
            
            for col_name in possible_columns:
                if col_name in row.index and pd.notna(row[col_name]) and row[col_name] != '':
                    last_heston = row[col_name]
                    logger.info(f"Using '{col_name}' value: {last_heston}")
                    break
            
            if last_heston is None:
                # If Last_heston doesn't exist or is empty, create a slightly different value for visibility
                last_heston = row['Heston_Price'] * 0.98  # Make it slightly lower for visibility
                logger.info(f"Last_heston not found or empty, using calculated value: {last_heston:.2f} (from Heston_Price: {row['Heston_Price']:.2f})")
            
            graph_data['last_heston'].append(last_heston)
            
            graph_data['last_traded'].append(row['PX_LAST'])
            
            # Check if this timestamp matches entry/exit trades
            entry_match = False
            exit_match = False
            
            # Match this with actual trade data from simulation results
            # Get simulation results to find entry/exit points for this label
            
            # Parse the timestamp to match with trade data
            try:
                # Extract hour and date from the time label
                time_parts = time_label.split(' - ')
                hour_str = time_parts[0]  # e.g., "12:00"
                date_str = time_parts[1]  # e.g., "Aug 06, 2025"
                
                # Convert to comparable format
                hour = int(hour_str.split(':')[0])
                date_obj = datetime.strptime(date_str, '%b %d, %Y')
                date_key = date_obj.strftime('%Y-%m-%d')
                
                logger.info(f"Processing data point: time_label='{time_label}' -> hour={hour}, date={date_key}")
                
                # Check for ENTRY marker (from enter_trade)
                if enter_trade:
                    try:
                        trade_timestamp = enter_trade.get('timestamp', '')
                        if trade_timestamp:
                            trade_parts = trade_timestamp.split(' - ')
                            trade_time = trade_parts[0]  # "12:00 PM"
                            trade_date = trade_parts[1]  # "Aug 06, 2025"
                            
                            time_obj = datetime.strptime(trade_time, '%I:%M %p')
                            trade_hour = time_obj.hour
                            
                            trade_date_obj = datetime.strptime(trade_date, '%b %d, %Y')
                            trade_date_key = trade_date_obj.strftime('%Y-%m-%d')
                            
                            # Check if this data point matches the ENTRY trade
                            if hour == trade_hour and date_key == trade_date_key:
                                entry_match = True
                                logger.info(f"Marking ENTRY for label {label} at {time_label}")
                    except Exception as parse_error:
                        logger.error(f"Error parsing ENTRY trade timestamp: {parse_error}")
                
                # Check for EXIT marker (from exit_trade)
                if exit_trade:
                    try:
                        trade_timestamp = exit_trade.get('timestamp', '')
                        if trade_timestamp:
                            trade_parts = trade_timestamp.split(' - ')
                            trade_time = trade_parts[0]  # "12:00 PM"
                            trade_date = trade_parts[1]  # "Aug 06, 2025"
                            
                            time_obj = datetime.strptime(trade_time, '%I:%M %p')
                            trade_hour = time_obj.hour
                            
                            trade_date_obj = datetime.strptime(trade_date, '%b %d, %Y')
                            trade_date_key = trade_date_obj.strftime('%Y-%m-%d')
                            
                            # Check if this data point matches the EXIT trade
                            if hour == trade_hour and date_key == trade_date_key:
                                exit_match = True
                                logger.info(f"Marking EXIT for label {label} at {time_label}")
                    except Exception as parse_error:
                        logger.error(f"Error parsing EXIT trade timestamp: {parse_error}")
                
            except Exception as e:
                logger.error(f"Error parsing timestamp for entry/exit matching: {e}")
            
            graph_data['entry_points'].append(entry_match)
            graph_data['exit_points'].append(exit_match)
        
        # Calculate PnL for this label and strike by using the EXIT trade's computed PnL
        pnl = 0.0
        if exit_trade:
            try:
                pnl = float(exit_trade.get('pnl', 0.0) or 0.0)
            except Exception:
                pnl = 0.0
        logger.info(f"Label {label}, Strike ${strike}: Graph PnL mapped from EXIT trade = ${pnl:.2f}")
        
        # Clean the graph data to ensure JSON serialization
        cleaned_graph_data = clean_data_for_json(graph_data)
        
        return jsonify({
            'success': True,
            'graph_data': cleaned_graph_data,
            'strike': strike,
            'label': label,
            'pnl': pnl
        })
        
    except Exception as e:
        logger.error(f"Error getting label graph data: {e}")
        return jsonify({'success': False, 'error': str(e)})

def run_trading_simulation(sheet_names, ticker: str = 'CRCL'):
    """Run trading simulation across multiple sheets"""
    try:
        # Initialize simulation state
        initial_capital = 100.0
        current_capital = initial_capital
        active_trades = {}
        trade_history = []
        hourly_capital = []
        hourly_trades = []
        
        # Track trades by type to ensure we get 2 BUY and 2 SELL
        trades_by_type = {'BUY': 0, 'SELL': 0}
        
        # Trade label management (A, B, C, D)
        available_labels = ['A', 'B', 'C', 'D']
        label_to_trade = {}  # Maps label to option_id
        trade_to_label = {}  # Maps option_id to label
        
        # Process each sheet in sequence
        for i, sheet_name in enumerate(sheet_names):
            logger.info(f"Processing sheet {i+1}/{len(sheet_names)}: {sheet_name}")
            
            # Load data for this sheet
            df = load_hourly_data(sheet_name, ticker)
            if df.empty:
                logger.warning(f"No data in sheet {sheet_name}, skipping")
                continue
            
            # Extract hour and date from sheet name
            hour, date = extract_hour_date_from_sheet(sheet_name)
            
            # Check exit conditions for existing trades
            trades_to_exit = []
            for option_id, trade in active_trades.items():
                if should_exit_trade(trade, df):
                    trades_to_exit.append(option_id)
            
            # Exit trades
            for option_id in trades_to_exit:
                # Get the trade label before freeing it
                original_label = trade_to_label.get(option_id, 'X')
                
                exit_result = exit_simulated_trade(option_id, df, active_trades, trade_history, current_capital, sheet_name, original_label)
                current_capital = exit_result['new_capital']
                
                # Free up the trade label
                if option_id in trade_to_label:
                    label = trade_to_label[option_id]
                    available_labels.append(label)
                    del label_to_trade[label]
                    del trade_to_label[option_id]
                    logger.info(f"Freed up trade label: {label}")
                
                trade_history.append(exit_result['trade_info'])
            
            # Select new trading opportunities throughout the simulation
            # Create new trades when we have less than 4 active trades and opportunities exist
            if len(active_trades) < 4:
                logger.info(f"Looking for new trades. Active trades: {len(active_trades)}, Capital: ${current_capital:.2f}")
                new_trades = select_simulated_trading_opportunities(df, active_trades, current_capital)
                logger.info(f"Found {len(new_trades)} new trading opportunities")
                
                # Enter new trades
                for trade_info in new_trades:
                    # Assign a trade label first
                    if available_labels:
                        label = available_labels.pop(0)  # Get first available label
                        label_to_trade[label] = trade_info['option_id']
                        trade_to_label[trade_info['option_id']] = label
                        logger.info(f"Assigned trade label {label} to {trade_info['option_id']}")
                    else:
                        label = 'X'  # Fallback if no labels available
                        logger.warning(f"No trade labels available, using fallback: {label}")
                    
                    # Enter the trade with the label
                    entry_result = enter_simulated_trade(trade_info, df, active_trades, current_capital, sheet_name, label)
                    if entry_result['success']:
                        current_capital = entry_result['new_capital']
                        
                        # Add ENTER trade to history
                        enter_trade = {
                            'action': 'ENTER',
                            'timestamp': trade_info['entry_time'],
                            'company': 'CRCL',
                            'option_id': trade_info['option_id'],
                            'trade_type': trade_info['trade_type'],
                            'strike_price': trade_info['strike_price'],
                            'option_type': trade_info['option_type'],
                            'entry_market_price': trade_info['entry_market_price'],
                            'entry_heston_price': trade_info['entry_heston_price'],
                            'position_size': trade_info['position_size'],
                            'entry_market_vs_heston': trade_info['market_vs_heston'],
                            'entry_heston_vs_market': trade_info['heston_vs_market'],
                            'trade_label': label,
                            'trade_key': trade_info['trade_key'],
                            'status': 'Active'
                        }
                        trade_history.append(enter_trade)
                        trades_by_type[trade_info['trade_type']] += 1
                        logger.info(f"ENTER trade {label}: {trade_info['option_id']} - {trade_info['trade_type']} - Strike: ${trade_info['strike_price']}")
                    else:
                        logger.warning(f"Failed to enter trade {trade_info['option_id']}: {entry_result.get('error', 'Unknown error')}")
            else:
                logger.info(f"No new trades needed. Active trades: {len(active_trades)}")
            
            # Record hourly state
            hourly_capital.append({
                'hour': hour,
                'date': date,
                'capital': current_capital,
                'active_trades': len(active_trades),
                'total_trades': len(trade_history)
            })
            
            hourly_trades.append({
                'hour': hour,
                'date': date,
                'trades_entered': len(new_trades) if 'new_trades' in locals() else 0,
                'trades_exited': len(trades_to_exit),
                'capital_change': current_capital - initial_capital
            })
        
        # Force close all remaining active trades at the end of simulation
        if active_trades:
            logger.info(f"Force closing {len(active_trades)} remaining active trades at end of simulation")
            
            # Use the last sheet's data for final exit prices
            last_sheet = sheet_names[-1]
            last_df = load_hourly_data(last_sheet, ticker)
            
            for option_id, trade in list(active_trades.items()):
                if not last_df.empty:
                    # Get the trade label before force closing
                    force_close_label = trade.get('trade_label', 'X')
                    exit_result = exit_simulated_trade(option_id, last_df, active_trades, trade_history, current_capital, last_sheet, force_close_label)
                    current_capital = exit_result['new_capital']
                    trade_history.append(exit_result['trade_info'])
                else:
                    # If no data available, create a forced exit with entry prices
                    forced_exit_trade = {
                        'action': 'EXIT',
                        'timestamp': f"End of simulation",
                        'company': 'CRCL',
                        'option_id': trade['option_id'],
                        'trade_type': trade['trade_type'],
                        'strike_price': trade['strike_price'],
                        'option_type': trade['option_type'],
                        'entry_market_price': trade['entry_market_price'],
                        'entry_heston_price': trade['entry_heston_price'],
                        'exit_price': trade['entry_market_price'],  # Use entry price as exit
                        'exit_heston': trade['entry_heston_price'],
                        'position_size': trade['position_size'],
                        'pnl': 0.0,  # No profit/loss if forced exit at entry
                        'return_pct': 0.0,
                        'entry_market_vs_heston': trade['market_vs_heston'],
                        'entry_heston_vs_market': trade['heston_vs_market'],
                        'exit_market_vs_heston': 0.0,
                        'exit_heston_vs_market': 0.0,
                        'trade_label': trade_label or trade.get('trade_label', 'X'),  # Use passed label or fallback
                        'trade_key': trade.get('trade_key', 'unknown'),  # Include the trade key for mapping
                        'status': 'Forced Close'
                    }
                    trade_history.append(forced_exit_trade)
                    del active_trades[option_id]
        
        # Calculate final statistics with NaN handling
        if pd.isna(current_capital) or pd.isna(initial_capital):
            total_return = 0.0
        else:
            total_return = ((current_capital - initial_capital) / initial_capital) * 100
        
        profitable_trades = sum(1 for trade in trade_history if trade.get('pnl', 0) > 0 and trade.get('action') == 'EXIT')
        exit_trades_count = len([t for t in trade_history if t.get('action') == 'EXIT'])
        win_rate = (profitable_trades / exit_trades_count * 100) if exit_trades_count > 0 else 0.0
        
        # Ensure no NaN values in final statistics
        if pd.isna(total_return):
            total_return = 0.0
        if pd.isna(win_rate):
            win_rate = 0.0
        
        simulation_summary = {
            'initial_capital': initial_capital,
            'final_capital': current_capital,
            'total_return_pct': total_return,
            'total_trades': len(trade_history),
            'profitable_trades': profitable_trades,
            'win_rate_pct': win_rate,
            'active_trades_at_end': len(active_trades),
            'hourly_capital': hourly_capital,
            'hourly_trades': hourly_trades,
            'trade_history': trade_history
        }
        
        logger.info(f"Simulation complete: {len(trade_history)} trades, {total_return:.2f}% return, {len(active_trades)} active trades remaining")
        return simulation_summary
        
    except Exception as e:
        logger.error(f"Error running trading simulation: {e}")
        return {}

def extract_hour_date_from_sheet(sheet_name):
    """Extract hour and date from sheet name like 'Hour_12_2025-08-06'"""
    try:
        parts = sheet_name.split('_')
        if len(parts) >= 3 and parts[0] == 'Hour':
            hour = int(parts[1])
            date = parts[2]
            
            # Parse the date properly
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                formatted_date = date_obj.strftime("%b %d, %Y")
            except:
                formatted_date = date
            
            return hour, formatted_date
        else:
            return 0, 'unknown'
    except:
        return 0, 'unknown'

def should_exit_trade(trade, current_data):
    """Check if a trade should be exited based on current data"""
    try:
        # Find matching option in current data
        option_mask = (
            (current_data['Strike'] == trade['strike_price']) & 
            (current_data['Option_Type'] == trade['option_type'])
        )
        
        if not option_mask.any():
            return False
        
        option_data = current_data[option_mask].iloc[0]
        current_market_price = option_data['PX_LAST']
        current_heston_price = option_data['Heston_Price']
        
        # Exit conditions based on strategy
        if trade['trade_type'] == 'BUY':
            # Exit when Market >= Heston (no longer undervalued)
            return current_market_price >= current_heston_price
        else:  # SELL
            # Exit when Market <= Heston (no longer overvalued)
            return current_market_price <= current_heston_price
            
    except Exception as e:
        logger.error(f"Error checking exit conditions: {e}")
        return False

def exit_simulated_trade(option_id, current_data, active_trades, trade_history, current_capital, sheet_name=None, trade_label=None):
    """Exit a simulated trade and calculate PnL"""
    trade = active_trades[option_id]
    
    # Find matching option in current data
    option_mask = (
        (current_data['Strike'] == trade['strike_price']) & 
        (current_data['Option_Type'] == trade['option_type'])
    )
    
    if not option_mask.any():
        # If we can't find the option, use entry price for exit
        exit_price = trade['entry_market_price']
        exit_heston = trade['entry_heston_price']
    else:
        option_data = current_data[option_mask].iloc[0]
        exit_price = option_data['PX_LAST']
        exit_heston = option_data['Heston_Price']
    
    # Calculate PnL with NaN handling
    if pd.isna(exit_price) or pd.isna(trade['entry_market_price']):
        # If we can't get valid prices, set PnL to 0
        pnl = 0.0
        exit_price = trade['entry_market_price']  # Use entry price as fallback
    else:
        if trade['trade_type'] == 'BUY':
            # Long position: profit = (exit_price - entry_price) * contracts
            pnl = (exit_price - trade['entry_market_price']) * trade['contracts']
        else:  # SELL
            # Short position: profit = (entry_price - exit_price) * contracts
            pnl = (trade['entry_market_price'] - exit_price) * trade['contracts']
    
    # Ensure PnL is not NaN
    if pd.isna(pnl):
        pnl = 0.0
    
    # Get current sheet name for proper timestamp
    if sheet_name:
        hour, date = extract_hour_date_from_sheet(sheet_name)
    else:
        hour, date = 0, 'unknown'
    
    # Format timestamp properly
    if hour == 0:
        time_str = "12:00 AM"
    elif hour < 12:
        time_str = f"{hour}:00 AM"
    elif hour == 12:
        time_str = "12:00 PM"
    else:
        time_str = f"{hour - 12}:00 PM"
    
    # Calculate return percentage with NaN handling
    if trade['position_size'] > 0 and not pd.isna(pnl):
        return_pct = (pnl / trade['position_size']) * 100
    else:
        return_pct = 0.0
    
    # Create EXIT trade entry matching user's log format
    exit_trade = {
        'action': 'EXIT',
        'timestamp': f"{time_str} - {date}",
        'company': 'CRCL',
        'option_id': trade['option_id'],
        'trade_type': trade['trade_type'],
        'strike_price': trade['strike_price'],
        'option_type': trade['option_type'],
        'entry_market_price': trade['entry_market_price'],
        'entry_heston_price': trade['entry_heston_price'],
        'exit_price': exit_price,
        'exit_heston': exit_heston,
        'position_size': trade['position_size'],
        'pnl': pnl,
        'return_pct': return_pct,
        'entry_market_vs_heston': trade['market_vs_heston'],
        'entry_heston_vs_market': trade['heston_vs_market'],
        'exit_market_vs_heston': exit_price - exit_heston,
        'exit_heston_vs_market': exit_heston - exit_price,
        'trade_label': trade_label or trade.get('trade_label', 'X'),  # Use passed label or fallback
        'trade_key': trade.get('trade_key', 'unknown'),  # Include the trade key for mapping
        'status': 'Closed'
    }
    
    # Update capital - return the position size + PnL
    new_capital = current_capital + pnl + trade['position_size']
    
    # Remove from active trades
    del active_trades[option_id]
    
    logger.info(f"EXIT trade: {trade['option_id']} - {trade['trade_type']} - PnL: ${pnl:.2f} - Return: {return_pct:.2f}%")
    
    return {
        'new_capital': new_capital,
        'trade_info': exit_trade
    }

def select_simulated_trading_opportunities(data, active_trades, current_capital):
    """Select new trading opportunities based on strategy"""
    try:
        logger.info(f"Selecting trading opportunities. Data rows: {len(data)}, Active trades: {len(active_trades)}")
        
        # Filter for valid Call options with non-null prices
        valid_data = data[
            (data['Option_Type'] == 'Call') & 
            data['PX_LAST'].notna() & 
            data['Heston_Price'].notna()
        ].copy()
        
        logger.info(f"Valid Call options: {len(valid_data)}")
        
        if len(valid_data) < 1:  # Changed from 4 to 1 to allow single trades
            logger.warning("Not enough valid Call options")
            return []
        
        # Calculate mispricing
        valid_data['Market_vs_Heston'] = valid_data['PX_LAST'] - valid_data['Heston_Price']
        valid_data['Heston_vs_Market'] = valid_data['Heston_Price'] - valid_data['PX_LAST']
        
        # Get currently active strike prices to avoid duplicates
        active_strikes = {trade['strike_price'] for trade in active_trades.values()}
        logger.info(f"Active strikes: {active_strikes}")
        
        # Filter out strikes we already have active trades for
        available_data = valid_data[~valid_data['Strike'].isin(active_strikes)].copy()
        logger.info(f"Available strikes (excluding active): {len(available_data)}")
        
        if len(available_data) < 1:  # Changed from 4 to 1 to allow single trades
            logger.warning("No available strikes after filtering")
            return []
        
        # Find undervalued options (Market < Heston) for BUY
        undervalued = available_data[available_data['Market_vs_Heston'] < 0].copy()
        undervalued = undervalued.sort_values('Heston_vs_Market', ascending=False)
        
        # Find overvalued options (Market > Heston) for SELL
        overvalued = available_data[available_data['Market_vs_Heston'] > 0].copy()
        overvalued = overvalued.sort_values('Market_vs_Heston', ascending=False)
        
        # Count current active trades by type
        active_buy_trades = sum(1 for trade in active_trades.values() if trade['trade_type'] == 'BUY')
        active_sell_trades = sum(1 for trade in active_trades.values() if trade['trade_type'] == 'SELL')
        
        # Always aim for exactly 2 BUY and 2 SELL trades (total 4)
        needed_buy_trades = max(0, 2 - active_buy_trades)
        needed_sell_trades = max(0, 2 - active_sell_trades)
        
        # Check if we can afford more trades
        max_trades = 4 - len(active_trades)
        if max_trades <= 0:
            return []
        
        new_trades = []
        
        # Add BUY opportunities (prioritize most undervalued)
        for _, row in undervalued.head(needed_buy_trades).iterrows():
            if len(new_trades) >= max_trades:
                break
                
            trade_info = {
                'option_id': row['Option_ID'],
                'trade_type': 'BUY',
                'strike_price': row['Strike'],
                'option_type': row['Option_Type'],
                'entry_market_price': row['PX_LAST'],
                'entry_heston_price': row['Heston_Price'],
                'position_size': min(25.0, current_capital),  # Dynamic position sizing
                'contracts': 1,
                'market_vs_heston': row['Market_vs_Heston'],
                'heston_vs_market': row['Heston_vs_Market']
            }
            # Create trade key for mapping ENTER/EXIT pairs
            trade_info['trade_key'] = f"{row['Option_ID']}_{row['Strike']}_{row['PX_LAST']:.2f}"
            new_trades.append(trade_info)
        
        # Add SELL opportunities (prioritize most overvalued)
        for _, row in overvalued.head(needed_sell_trades).iterrows():
            if len(new_trades) >= max_trades:
                break
                
            trade_info = {
                'option_id': row['Option_ID'],
                'trade_type': 'SELL',
                'strike_price': row['Strike'],
                'option_type': row['Option_Type'],
                'entry_market_price': row['PX_LAST'],
                'entry_heston_price': row['Heston_Price'],
                'position_size': min(25.0, current_capital),  # Dynamic position sizing
                'contracts': 1,
                'market_vs_heston': row['Market_vs_Heston'],
                'heston_vs_market': row['Heston_vs_Market']
            }
            # Create trade key for mapping ENTER/EXIT pairs
            trade_info['trade_key'] = f"{row['Option_ID']}_{row['Strike']}_{row['PX_LAST']:.2f}"
            new_trades.append(trade_info)
        
        # If we still don't have enough trades and we're below 4 active trades,
        # consider reusing strikes but with different option IDs to maintain 4 positions
        if len(new_trades) < max_trades and len(active_trades) < 4:
            logger.info(f"Need {max_trades - len(new_trades)} more trades to reach 4 active positions")
            # This could be expanded to find alternative opportunities
        
        logger.info(f"Selected {len(new_trades)} new trades: {needed_buy_trades} BUY, {needed_sell_trades} SELL")
        return new_trades
        
    except Exception as e:
        logger.error(f"Error selecting trading opportunities: {e}")
        return []

def enter_simulated_trade(trade_info, data, active_trades, current_capital, sheet_name=None, trade_label=None):
    """Enter a new simulated trade"""
    try:
        # Check if we can afford this trade
        if trade_info['position_size'] > current_capital:
            return {'success': False, 'error': 'Insufficient capital'}
        
        # Add entry time with proper formatting
        if sheet_name:
            hour, date = extract_hour_date_from_sheet(sheet_name)
        else:
            hour, date = 0, 'unknown'
        
        # Format timestamp properly
        if hour == 0:
            time_str = "12:00 AM"
        elif hour < 12:
            time_str = f"{hour}:00 AM"
        elif hour == 12:
            time_str = "12:00 PM"
        else:
            time_str = f"{hour - 12}:00 PM"
        
        trade_info['entry_time'] = f"{time_str} - {date}"
        trade_info['entry_hour'] = hour
        trade_info['entry_date'] = date
        
        # Add trade label to trade_info
        if trade_label:
            trade_info['trade_label'] = trade_label
        
        # Add to active trades
        active_trades[trade_info['option_id']] = trade_info
        
        # Update capital
        new_capital = current_capital - trade_info['position_size']
        
        return {
            'success': True,
            'new_capital': new_capital,
            'trade_info': trade_info
        }
        
    except Exception as e:
        logger.error(f"Error entering simulated trade: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
