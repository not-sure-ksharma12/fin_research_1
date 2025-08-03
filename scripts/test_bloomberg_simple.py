import sys
import logging

# Add scripts directory to path
sys.path.append(r"C:\Users\ksharma12\fin_research\scripts")

# Import the existing modules
from fetch_options_to_excel import connect_to_bloomberg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bloomberg_connection():
    """Test basic Bloomberg connection"""
    
    try:
        logger.info("Testing Bloomberg connection...")
        
        # Try to connect
        session = connect_to_bloomberg()
        logger.info("✅ Successfully connected to Bloomberg Terminal")
        
        # Close session
        session.stop()
        logger.info("✅ Bloomberg session closed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Bloomberg: {e}")
        logger.info("💡 Please ensure Bloomberg Terminal is running and you are logged in")
        return False

if __name__ == "__main__":
    test_bloomberg_connection() 