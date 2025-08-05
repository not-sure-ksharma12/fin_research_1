#!/usr/bin/env python3
"""
Script to regenerate Excel files with active trades
This fixes the issue where trades are logged but don't show up in Excel files
"""

import sys
import os
from datetime import datetime

# Add scripts directory to path
sys.path.append(r"C:\Users\ksharma12\fin_research\scripts")

from multi_company_strategy import MultiCompanyRealTimeTrading

def main():
    """Regenerate Excel files with active trades"""
    
    print("=" * 80)
    print("🔄 REGENERATING EXCEL FILES WITH ACTIVE TRADES...")
    print("=" * 80)
    
    # Define the companies
    companies = ['NVDA', 'AUR', 'TSLA', 'SOFI', 'SOUN', 'AMD', 'AVGO', 'CRCL', 'BBAI', 'SLDB']
    
    # Initialize multi-company system
    trading_system = MultiCompanyRealTimeTrading(companies, capital_per_company=100.0)
    
    # Regenerate all Excel files
    trading_system.regenerate_all_excel_files()
    
    print("=" * 80)
    print("✅ EXCEL FILES REGENERATED SUCCESSFULLY!")
    print("=" * 80)
    print("Check the Excel files in: scripts/realtime_output/multi_company_sep19/")
    print("The trades should now be visible with proper color coding.")

if __name__ == "__main__":
    main() 