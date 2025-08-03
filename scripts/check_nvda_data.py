import pandas as pd
import os

def check_nvda_data():
    """Check the structure of NVDA data files"""
    
    # Try different possible file paths
    possible_files = [
        "scripts/excels/nvda_options_2025-08-15_heston.xlsx",
        "analysis/nvda_options_2025-12-19_heston.xlsx",
        "scripts/excels/nvda_options_2025-08-15.xlsx"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"\nFound file: {file_path}")
            try:
                df = pd.read_excel(file_path)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"First few rows:")
                print(df.head(3))
                print(f"Data types:")
                print(df.dtypes)
                return df
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print("No NVDA data files found!")
    return None

if __name__ == "__main__":
    check_nvda_data() 