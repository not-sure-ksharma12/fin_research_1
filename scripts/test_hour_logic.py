import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hour_logic():
    """Test the hour logic with different scenarios"""
    
    # Test data: trades entered at different hours
    test_trades = {
        'TRADE_8': {'entry_time': '2025-08-05 08:01:20', 'status': 'active'},
        'TRADE_9': {'entry_time': '2025-08-05 09:01:20', 'status': 'active'},
        'TRADE_10': {'entry_time': '2025-08-05 10:01:20', 'status': 'active'},
    }
    
    logger.info("Testing hour logic with trades entered at different hours:")
    for trade_id, trade in test_trades.items():
        entry_hour = int(trade['entry_time'].split(' ')[1].split(':')[0])
        logger.info(f"  {trade_id}: entered at hour {entry_hour}")
    
    # Test for each hour
    for hour in [8, 9, 10]:
        logger.info(f"\n=== Testing Hour {hour} ===")
        trades_for_this_hour = {}
        
        for trade_id, trade in test_trades.items():
            entry_hour = int(trade['entry_time'].split(' ')[1].split(':')[0])
            logger.info(f"  {trade_id}: entry_hour={entry_hour}, current_hour={hour}, condition={entry_hour} <= {hour} = {entry_hour <= hour}")
            
            if entry_hour <= hour:
                trades_for_this_hour[trade_id] = {'trade': trade, 'status': 'Entered'}
                logger.info(f"    -> INCLUDED in hour {hour}")
            else:
                logger.info(f"    -> EXCLUDED from hour {hour}")
        
        logger.info(f"  Trades for hour {hour}: {list(trades_for_this_hour.keys())}")

if __name__ == "__main__":
    test_hour_logic() 