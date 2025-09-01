#!/usr/bin/env python3
"""
Excel to JSON Converter for Multiple Company Hourly Data Files
Converts multiple Excel files with multiple sheets to JSON format
"""

import pandas as pd
import json
import os
from datetime import datetime
import numpy as np

def clean_data_for_json(obj):
    """Convert pandas/numpy objects to JSON-serializable types"""
    if pd.isna(obj) or obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    else:
        return obj

def convert_excel_to_json(excel_file_path, output_dir=None):
    """
    Convert Excel file with multiple sheets to JSON format
    
    Args:
        excel_file_path (str): Path to the Excel file
        output_dir (str): Directory to save JSON files (optional)
    
    Returns:
        dict: Dictionary containing all sheet data
    """
    
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
    
    print(f"Reading Excel file: {excel_file_path}")
    
    # Read all sheet names
    xl = pd.ExcelFile(excel_file_path)
    sheet_names = xl.sheet_names
    
    print(f"Found {len(sheet_names)} sheets: {sheet_names}")
    
    # Dictionary to store all data
    all_data = {
        "metadata": {
            "source_file": excel_file_path,
            "conversion_date": datetime.now().isoformat(),
            "total_sheets": len(sheet_names),
            "sheet_names": sheet_names
        },
        "sheets": {}
    }
    
    # Process each sheet
    for sheet_name in sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")
        
        try:
            # Read the sheet
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            
            # Clean column names (remove spaces, special characters)
            df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
            
            # Convert data to JSON-serializable format
            cleaned_data = []
            for _, row in df.iterrows():
                row_dict = {}
                for col in df.columns:
                    row_dict[col] = clean_data_for_json(row[col])
                cleaned_data.append(row_dict)
            
            # Store sheet data
            all_data["sheets"][sheet_name] = {
                "row_count": len(cleaned_data),
                "columns": list(df.columns),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "data": cleaned_data
            }
            
            print(f"  ✓ Processed {len(cleaned_data)} rows with {len(df.columns)} columns")
            
        except Exception as e:
            print(f"  ✗ Error processing sheet {sheet_name}: {str(e)}")
            all_data["sheets"][sheet_name] = {
                "error": str(e),
                "row_count": 0,
                "columns": [],
                "data_types": {},
                "data": []
            }
    
    # Save to JSON file
    if output_dir is None:
        output_dir = os.path.dirname(excel_file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(excel_file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.json")
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✓ JSON file saved: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return all_data

def create_summary_report(all_conversions, output_dir):
    """Create a summary report of all conversions"""
    
    report_file = os.path.join(output_dir, "conversion_summary.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("MULTIPLE EXCEL TO JSON CONVERSION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Conversion Date: {datetime.now().isoformat()}\n")
        f.write(f"Total Files Processed: {len(all_conversions)}\n\n")
        
        for file_path, conversion_data in all_conversions.items():
            f.write(f"FILE: {os.path.basename(file_path)}\n")
            f.write("-" * 50 + "\n")
            
            if conversion_data:
                f.write(f"Status: SUCCESS\n")
                f.write(f"Total Sheets: {conversion_data['metadata']['total_sheets']}\n")
                f.write(f"Total Rows: {sum(sheet['row_count'] for sheet in conversion_data['sheets'].values() if 'error' not in sheet):,}\n")
                f.write(f"Output: {os.path.basename(file_path).replace('.xlsx', '.json')}\n")
            else:
                f.write(f"Status: FAILED\n")
            
            f.write("\n")
    
    print(f"✓ Summary report saved: {report_file}")

def convert_multiple_files(file_paths, output_dir=None):
    """
    Convert multiple Excel files to JSON format
    
    Args:
        file_paths (list): List of Excel file paths to convert
        output_dir (str): Directory to save JSON files (optional)
    
    Returns:
        dict: Dictionary mapping file paths to conversion results
    """
    
    all_conversions = {}
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING FILE {i}/{len(file_paths)}: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        try:
            result = convert_excel_to_json(file_path, output_dir)
            all_conversions[file_path] = result
            print(f"✓ SUCCESS: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"✗ FAILED: {os.path.basename(file_path)} - Error: {str(e)}")
            all_conversions[file_path] = None
    
    return all_conversions

def main():
    """Main function to run the converter for multiple files"""
    
    # List of Excel files to convert
    excel_files = [
        "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19/AMD_hourly_data.xlsx",
        "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19/AVGO_hourly_data.xlsx",
        "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19/BBAI_hourly_data.xlsx",
        "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19/NVDA_hourly_data.xlsx",
        "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19/SLDB_hourly_data.xlsx",
        "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19/SOFI_hourly_data.xlsx",
        "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19/SOUN_hourly_data.xlsx",
        "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19/TSLA_hourly_data.xlsx"
    ]
    
    # Output directory (same as input files)
    output_dir = "/Users/kalpit/fin_research_1/scripts/scripts/realtime_output/multi_company_sep19"
    
    print("MULTIPLE EXCEL TO JSON CONVERTER")
    print("=" * 60)
    print(f"Converting {len(excel_files)} Excel files to JSON format...")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Convert all files
        all_conversions = convert_multiple_files(excel_files, output_dir)
        
        # Create summary report
        create_summary_report(all_conversions, output_dir)
        
        # Print final statistics
        print("\n" + "="*80)
        print("ALL CONVERSIONS COMPLETED!")
        print("="*80)
        
        successful = sum(1 for result in all_conversions.values() if result is not None)
        failed = len(all_conversions) - successful
        
        print(f"✓ Successful conversions: {successful}")
        if failed > 0:
            print(f"✗ Failed conversions: {failed}")
        
        # Show total statistics
        total_rows = 0
        total_sheets = 0
        for result in all_conversions.values():
            if result:
                total_rows += sum(sheet['row_count'] for sheet in result['sheets'].values() if 'error' not in sheet)
                total_sheets += result['metadata']['total_sheets']
        
        print(f"📊 Total rows processed: {total_rows:,}")
        print(f"📊 Total sheets processed: {total_sheets}")
        
        # List all generated JSON files
        print(f"\n📁 Generated JSON files:")
        for file_path in excel_files:
            if all_conversions[file_path]:
                json_file = os.path.basename(file_path).replace('.xlsx', '.json')
                print(f"  ✓ {json_file}")
            else:
                print(f"  ✗ {os.path.basename(file_path)} (failed)")
        
    except Exception as e:
        print(f"Error during conversion process: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
