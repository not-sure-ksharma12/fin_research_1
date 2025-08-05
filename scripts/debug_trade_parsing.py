import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_trade_parsing():
    """Debug the trade log parsing"""
    
    log_file = "realtime_output/multi_company_sep19/CRCL_trades.log"
    
    logger.info(f"File exists: {os.path.exists(log_file)}")
    logger.info(f"File path: {os.path.abspath(log_file)}")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info(f"Total lines in log: {len(lines)}")
        
        for i, line in enumerate(lines):
            line = line.strip()
            logger.info(f"Line {i+1}: '{line}'")
            
            if not line:
                continue
                
            # Parse log entry: [timestamp] ACTION - COMPANY - OPTION_ID - TRADE_TYPE - AMOUNT
            parts = line.split(' - ')
            logger.info(f"  Parts: {parts}")
            logger.info(f"  Number of parts: {len(parts)}")
            
            if len(parts) >= 5:
                timestamp_str = parts[0].strip('[]')
                action = parts[1].strip()
                company = parts[2].strip()
                option_id = parts[3].strip()
                trade_type_part = parts[4].strip()
                
                logger.info(f"  Parsed: timestamp='{timestamp_str}', action='{action}', company='{company}', option_id='{option_id}', trade_type_part='{trade_type_part}'")
                
                # Extract amount (remove $ sign)
                if '$' in trade_type_part:
                    amount = trade_type_part.split('$')[1].split()[0]
                    trade_type = trade_type_part.split('$')[0].strip()
                    logger.info(f"  Extracted: trade_type='{trade_type}', amount='{amount}'")
                else:
                    logger.info(f"  No $ found in trade_type_part")
                
                if len(parts) >= 6:
                    logger.info(f"  Additional parts: {parts[5:]}")
        
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_trade_parsing() 