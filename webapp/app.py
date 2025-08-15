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

# Data file paths
HOURLY_DATA_PATH = "../scripts/scripts/realtime_output/multi_company_sep19/CRCL_hourly_data.xlsx"
TRADES_LOG_PATH = "../scripts/scripts/realtime_output/multi_company_sep19/CRCL_trades.log"

def get_excel_sheet_names():
    """Get list of available sheet names from Excel file"""
    try:
        if os.path.exists(HOURLY_DATA_PATH):
            xl_file = pd.ExcelFile(HOURLY_DATA_PATH)
            sheet_names = xl_file.sheet_names
            logger.info(f"Available sheets: {sheet_names}")
            return sheet_names
        else:
            logger.warning(f"Excel file not found: {HOURLY_DATA_PATH}")
            return []
    except Exception as e:
        logger.error(f"Error getting sheet names: {e}")
        return []

def load_hourly_data(sheet_name=None):
    """Load CRCL hourly data from Excel file"""
    try:
        if os.path.exists(HOURLY_DATA_PATH):
            if sheet_name:
                df = pd.read_excel(HOURLY_DATA_PATH, sheet_name=sheet_name)
                logger.info(f"Loaded hourly data from sheet '{sheet_name}': {len(df)} rows")
            else:
                # Load first sheet if no specific sheet specified
                df = pd.read_excel(HOURLY_DATA_PATH)
                logger.info(f"Loaded hourly data from first sheet: {len(df)} rows")
            return df
        else:
            logger.warning(f"Hourly data file not found: {HOURLY_DATA_PATH}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading hourly data: {e}")
        return pd.DataFrame()

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

def load_trades_log():
    """Load CRCL trades from log file"""
    try:
        if os.path.exists(TRADES_LOG_PATH):
            trades = []
            with open(TRADES_LOG_PATH, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        # Parse log line: [2025-08-06 12:56:47] ENTER - CRCL - CRCL_95.0_Call - BUY - $25.00
                        try:
                            # Extract timestamp
                            timestamp_str = line[1:20]  # [2025-08-06 12:56:47]
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            
                            # Parse action and details
                            parts = line.split(" - ")
                            logger.info(f"Line {line_num}: Parsing '{line}' -> {len(parts)} parts: {parts}")
                            
                            if len(parts) >= 5:
                                # The first part contains timestamp and action, need to split it
                                first_part = parts[0]  # [2025-08-06 12:56:47] ENTER
                                action = first_part.split("] ")[1]  # Extract ENTER/EXIT
                                company = parts[1]  # CRCL
                                option_id = parts[2]  # CRCL_95.0_Call
                                trade_type = parts[3]  # BUY or SELL
                                
                                # Extract strike and option type from option_id
                                option_parts = option_id.split("_")
                                strike = None
                                option_type = None
                                
                                if len(option_parts) >= 3:
                                    try:
                                        strike = float(option_parts[1])
                                        option_type = option_parts[2]
                                    except (ValueError, IndexError):
                                        strike = 0.0
                                        option_type = "Unknown"
                                else:
                                    strike = 0.0
                                    option_type = "Unknown"
                                
                                # Extract price and PnL if available
                                price = None
                                pnl = None
                                return_pct = None
                                
                                if action == "ENTER":
                                    if len(parts) > 4:
                                        try:
                                            price_str = parts[4]
                                            price = float(price_str.replace("$", ""))
                                        except (ValueError, AttributeError):
                                            price = 0.0
                                    else:
                                        price = 0.0
                                elif action == "EXIT":
                                    if len(parts) > 4:
                                        try:
                                            price_str = parts[4]
                                            price = float(price_str.replace("$", ""))
                                        except (ValueError, AttributeError):
                                            price = 0.0
                                    else:
                                        price = 0.0
                                        
                                    if len(parts) > 5:
                                        try:
                                            pnl_str = parts[5]  # PnL: $13.79
                                            pnl = float(pnl_str.split("$")[1])
                                        except (ValueError, IndexError):
                                            pnl = 0.0
                                    if len(parts) > 6:
                                        try:
                                            return_str = parts[6]  # Return: 55.16%
                                            logger.info(f"Processing return string: '{return_str}'")
                                            # Extract just the number part after "Return: " and before "%"
                                            return_pct = float(return_str.replace("Return: ", "").replace("%", ""))
                                            logger.info(f"Extracted return percentage: {return_pct}")
                                        except (ValueError, IndexError) as e:
                                            logger.warning(f"Failed to parse return percentage from '{return_str}': {e}")
                                            return_pct = 0.0
                                
                                trade_obj = {
                                    'timestamp': timestamp.isoformat(),
                                    'action': action,
                                    'company': company,
                                    'option_id': option_id,
                                    'strike': strike,
                                    'option_type': option_type,
                                    'trade_type': trade_type,
                                    'price': price,
                                    'pnl': pnl,
                                    'return_pct': return_pct
                                }
                                
                                # Debug logging for EXIT trades
                                if action == "EXIT":
                                    logger.info(f"EXIT trade created: {trade_obj}")
                                
                                trades.append(trade_obj)
                        except Exception as e:
                            logger.warning(f"Could not parse line: {line}, error: {e}")
                            continue
            
            logger.info(f"Loaded trades log: {len(trades)} trades")
            return trades
        else:
            logger.warning(f"Trades log file not found: {TRADES_LOG_PATH}")
            return []
    except Exception as e:
        logger.error(f"Error loading trades log: {e}")
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
        
        if data_type == 'hourly':
            # Get sheet name from query parameters if provided
            sheet_name = request.args.get('sheet')
            
            if sheet_name:
                # Load data from specific sheet for date/time view
                logger.info(f"Loading hourly data from specific sheet: {sheet_name}")
                df = load_hourly_data(sheet_name)
            else:
                # Load data from the first available sheet to get strikes and dates for dropdowns
                logger.info("Loading hourly data from first sheet for dropdowns")
                df = load_hourly_data()
            
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
            trades = load_trades_log()
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
        sheet_names = get_excel_sheet_names()
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
        
        logger.info(f"Loading data for strike {strike_float} across all sheets")
        
        # Get all available sheets
        sheet_names = get_excel_sheet_names()
        all_strike_data = []
        
        for sheet_name in sheet_names:
            try:
                # Load data from each sheet
                df = load_hourly_data(sheet_name)
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
        trades = load_trades_log()
        hourly_data = load_hourly_data()
        
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
