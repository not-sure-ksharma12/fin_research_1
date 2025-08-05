import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_aur_hour_logic():
    """Debug the hour logic for AUR trades"""
    
    log_file = "realtime_output/multi_company_sep19/AUR_trades.log"
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        active_trades = {}
        exited_trades = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse log entry: [timestamp] ACTION - COMPANY - OPTION_ID - TRADE_TYPE - AMOUNT
            parts = line.split(' - ')
            if len(parts) >= 5:
                # First part contains timestamp and action: [timestamp] ACTION
                first_part = parts[0]
                if ']' in first_part:
                    timestamp_str = first_part.split(']')[0].strip('[')
                    action = first_part.split(']')[1].strip()
                else:
                    timestamp_str = first_part
                    action = parts[1].strip() if len(parts) > 1 else ""
                
                company = parts[1].strip() if ']' not in first_part else parts[1].strip()
                option_id = parts[2].strip() if ']' not in first_part else parts[2].strip()
                trade_type_part = parts[3].strip() if ']' not in first_part else parts[3].strip()
                
                # Extract amount (remove $ sign)
                amount = trade_type_part.split('$')[1].split()[0] if '$' in trade_type_part else "25.00"
                trade_type = trade_type_part.split('$')[0].strip()
                
                # Parse hour from timestamp
                entry_hour = int(timestamp_str.split(' ')[1].split(':')[0])
                
                logger.info(f"Parsed: {option_id} - {action} - Hour: {entry_hour} - Timestamp: {timestamp_str}")
                
                if action == 'ENTER':
                    active_trades[option_id] = {
                        'company': company,
                        'trade_type': trade_type,
                        'position_size': float(amount),
                        'entry_time': timestamp_str
                    }
                elif action == 'EXIT':
                    # Remove from active trades and add to exited trades
                    if option_id in active_trades:
                        trade_info = active_trades.pop(option_id)
                        trade_info['exit_time'] = timestamp_str
                        
                        # Parse PnL if available
                        if len(parts) >= 6:
                            pnl_part = parts[5]
                            if 'PnL:' in pnl_part:
                                pnl_str = pnl_part.split('PnL: $')[1].split()[0]
                                trade_info['pnl'] = float(pnl_str)
                        
                        exited_trades[option_id] = trade_info
        
        logger.info(f"\nActive trades: {len(active_trades)}")
        for option_id, trade in active_trades.items():
            entry_hour = int(trade['entry_time'].split(' ')[1].split(':')[0])
            logger.info(f"  {option_id}: entered at hour {entry_hour}")
        
        logger.info(f"\nExited trades: {len(exited_trades)}")
        for option_id, trade in exited_trades.items():
            exit_hour = int(trade['exit_time'].split(' ')[1].split(':')[0])
            logger.info(f"  {option_id}: exited at hour {exit_hour}")
        
        # Test hour logic
        for hour in [8, 9]:
            logger.info(f"\n=== Testing Hour {hour} ===")
            trades_for_this_hour = {}
            
            # Add active trades that were entered in this hour or earlier
            for option_id, trade in active_trades.items():
                entry_hour = int(trade['entry_time'].split(' ')[1].split(':')[0])
                logger.info(f"  {option_id}: entry_hour={entry_hour}, current_hour={hour}, include={entry_hour <= hour}")
                if entry_hour <= hour:
                    trades_for_this_hour[option_id] = {'trade': trade, 'status': 'Entered'}
            
            # Add exited trades that were exited in this hour
            for option_id, trade in exited_trades.items():
                exit_hour = int(trade['exit_time'].split(' ')[1].split(':')[0])
                logger.info(f"  {option_id}: exit_hour={exit_hour}, current_hour={hour}")
                if exit_hour == hour:
                    trades_for_this_hour[option_id] = {'trade': trade, 'status': 'Exited'}
                    logger.info(f"    -> Marked as Exited")
                elif exit_hour > hour:
                    # Trade was still active in this hour
                    trades_for_this_hour[option_id] = {'trade': trade, 'status': 'Entered'}
                    logger.info(f"    -> Marked as Entered (still active)")
            
            logger.info(f"  Total trades for hour {hour}: {len(trades_for_this_hour)}")
            for option_id, info in trades_for_this_hour.items():
                logger.info(f"    {option_id}: {info['status']}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_aur_hour_logic() 