#!/usr/bin/env python3
"""
Startup script for CRCL Trading Dashboard
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import pandas
        import openpyxl
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if data files exist"""
    hourly_path = "../scripts/scripts/realtime_output/multi_company_sep19/CRCL_hourly_data.xlsx"
    trades_path = "../scripts/scripts/realtime_output/multi_company_sep19/CRCL_trades.log"
    
    files_exist = True
    
    if not os.path.exists(hourly_path):
        print(f"⚠️  Hourly data file not found: {hourly_path}")
        files_exist = False
    else:
        print(f"✅ Hourly data file found: {hourly_path}")
    
    if not os.path.exists(trades_path):
        print(f"⚠️  Trades log file not found: {trades_path}")
        files_exist = False
    else:
        print(f"✅ Trades log file found: {trades_path}")
    
    return files_exist

def main():
    """Main startup function"""
    print("🚀 Starting CRCL Trading Dashboard...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print()
    
    # Check data files
    if not check_data_files():
        print("\n⚠️  Some data files are missing. The dashboard may not work properly.")
        print("   Please ensure the data files exist before proceeding.")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print()
    print("🌐 Starting Flask application...")
    print("   Dashboard will be available at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
