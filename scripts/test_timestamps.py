import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_timestamps():
    """Test timestamp generation"""
    
    # Simulate the current approach
    print("Current approach:")
    for hour in range(5):
        timestamp = datetime.now() + timedelta(hours=hour)
        print(f"Hour {hour}: {timestamp}")
    
    print("\nFixed approach:")
    start_time = datetime.now()
    for hour in range(5):
        timestamp = start_time + timedelta(hours=hour)
        print(f"Hour {hour}: {timestamp}")

if __name__ == "__main__":
    test_timestamps() 