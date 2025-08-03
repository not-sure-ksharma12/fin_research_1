import sys
import logging

# Add scripts directory to path
sys.path.append(r"C:\Users\ksharma12\fin_research\scripts")

# Import the existing modules
from fetch_options_to_excel import connect_to_bloomberg
import blpapi
from blpapi.event import Event

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_options_chain():
    """Test options chain retrieval with detailed error handling"""
    
    try:
        logger.info("Testing Bloomberg options chain retrieval...")
        
        # Connect to Bloomberg
        session = connect_to_bloomberg()
        logger.info("✅ Connected to Bloomberg")
        
        # Test with a simple ticker first
        test_ticker = "NVDA US Equity"
        logger.info(f"Testing with ticker: {test_ticker}")
        
        ref_data = session.getService("//blp/refdata")
        request = ref_data.createRequest("ReferenceDataRequest")
        request.append("securities", test_ticker)
        request.append("fields", "OPT_CHAIN")
        
        logger.info("Sending request...")
        session.sendRequest(request)
        
        chain_tickers = []
        while True:
            ev = session.nextEvent(500)
            logger.info(f"Event type: {ev.eventType()}")
            
            for msg in ev:
                logger.info(f"Message type: {msg.messageType()}")
                
                if msg.messageType() == "ReferenceDataResponse":
                    logger.info("Got ReferenceDataResponse")
                    
                    # Check if securityData exists
                    if msg.hasElement("securityData"):
                        sec_data = msg.getElement("securityData")
                        logger.info(f"Found securityData with {sec_data.numValues()} values")
                        
                        for sec in sec_data.values():
                            logger.info(f"Processing security: {sec.getElementAsString('security')}")
                            
                            if sec.hasElement("fieldData"):
                                field_data = sec.getElement("fieldData")
                                logger.info("Found fieldData")
                                
                                if field_data.hasElement("OPT_CHAIN"):
                                    chain = field_data.getElement("OPT_CHAIN")
                                    logger.info(f"Found OPT_CHAIN with {chain.numValues()} options")
                                    
                                    for i in range(chain.numValues()):
                                        opt_elem = chain.getValue(i)
                                        if opt_elem.hasElement("Security Description"):
                                            desc = opt_elem.getElementAsString("Security Description")
                                            chain_tickers.append(desc)
                                            logger.info(f"Option: {desc}")
                                else:
                                    logger.warning("OPT_CHAIN field not found")
                            else:
                                logger.warning("fieldData not found")
                    else:
                        logger.error("securityData element not found in response")
                        
                        # Try to see what elements are available
                        logger.info("Available elements in response:")
                        for i in range(msg.numElements()):
                            elem = msg.getElement(i)
                            logger.info(f"  {elem.name()}: {elem.datatype()}")
                
                elif msg.messageType() == "ReferenceDataError":
                    logger.error("ReferenceDataError received")
                    if msg.hasElement("errorInfo"):
                        error_info = msg.getElement("errorInfo")
                        logger.error(f"Error: {error_info}")
                
                # Check for responseError in ReferenceDataResponse
                if msg.messageType() == "ReferenceDataResponse" and msg.hasElement("responseError"):
                    logger.error("ReferenceDataResponse contains responseError")
                    response_error = msg.getElement("responseError")
                    logger.error(f"Response Error: {response_error}")
                    
                    # Try to get more details about the error
                    if response_error.hasElement("source"):
                        source = response_error.getElementAsString("source")
                        logger.error(f"Error Source: {source}")
                    if response_error.hasElement("code"):
                        code = response_error.getElementAsInteger("code")
                        logger.error(f"Error Code: {code}")
                    if response_error.hasElement("category"):
                        category = response_error.getElementAsString("category")
                        logger.error(f"Error Category: {category}")
                    if response_error.hasElement("subcategory"):
                        subcategory = response_error.getElementAsString("subcategory")
                        logger.error(f"Error Subcategory: {subcategory}")
                    if response_error.hasElement("message"):
                        message = response_error.getElementAsString("message")
                        logger.error(f"Error Message: {message}")
            
            if ev.eventType() == Event.RESPONSE:
                break
        
        logger.info(f"Total options found: {len(chain_tickers)}")
        
        # Close session
        session.stop()
        logger.info("✅ Bloomberg session closed")
        
        return chain_tickers
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    test_options_chain() 