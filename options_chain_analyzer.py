import yfinance as yf
import pandas as pd
from datetime import datetime
import time

def get_full_options_chain(ticker_symbol, max_retries=3, delay=1):
    """
    Fetch complete options chain data for a given ticker symbol.
    
    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'TSLA')
        max_retries (int): Maximum number of retries for failed requests
        delay (int): Delay between retries in seconds
    
    Returns:
        tuple: (calls_dataframe, puts_dataframe)
    """
    ticker = yf.Ticker(ticker_symbol)
    
    print(f"Fetching options data for: {ticker_symbol}")
    print(f"Current stock price: ${ticker.info.get('regularMarketPrice', 'N/A')}")
    
    options_dates = ticker.options
    if not options_dates:
        print(f"No options data available for {ticker_symbol}")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Found {len(options_dates)} expiration dates")
    
    all_calls = []
    all_puts = []

    for i, exp_date in enumerate(options_dates, 1):
        print(f"Processing expiration {i}/{len(options_dates)}: {exp_date}")
        
        for attempt in range(max_retries):
            try:
                opt = ticker.option_chain(exp_date)
                calls = opt.calls.copy()
                puts = opt.puts.copy()

                # Add expiration date and ticker symbol
                calls["expirationDate"] = exp_date
                calls["ticker"] = ticker_symbol
                puts["expirationDate"] = exp_date
                puts["ticker"] = ticker_symbol

                all_calls.append(calls)
                all_puts.append(puts)
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for {exp_date}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    print(f"Failed to fetch data for {exp_date} after {max_retries} attempts")

    # Combine all expirations
    if all_calls:
        df_calls = pd.concat(all_calls, ignore_index=True)
        print(f"Total calls: {len(df_calls)}")
    else:
        df_calls = pd.DataFrame()
        print("No calls data retrieved")
    
    if all_puts:
        df_puts = pd.concat(all_puts, ignore_index=True)
        print(f"Total puts: {len(df_puts)}")
    else:
        df_puts = pd.DataFrame()
        print("No puts data retrieved")

    return df_calls, df_puts

def analyze_options_data(calls_df, puts_df, ticker_symbol):
    """
    Perform basic analysis on the options data.
    
    Args:
        calls_df (DataFrame): Calls data
        puts_df (DataFrame): Puts data
        ticker_symbol (str): Stock ticker symbol
    """
    print(f"\n=== Options Analysis for {ticker_symbol} ===")
    
    if not calls_df.empty:
        print(f"\nCalls Summary:")
        print(f"Total call contracts: {len(calls_df)}")
        print(f"Expiration dates: {calls_df['expirationDate'].nunique()}")
        print(f"Strike price range: ${calls_df['strike'].min():.2f} - ${calls_df['strike'].max():.2f}")
        
        # Most active calls (by volume)
        if 'volume' in calls_df.columns:
            top_calls = calls_df.nlargest(5, 'volume')[['strike', 'expirationDate', 'volume', 'openInterest', 'lastPrice']]
            print(f"\nTop 5 calls by volume:")
            print(top_calls.to_string(index=False))
    
    if not puts_df.empty:
        print(f"\nPuts Summary:")
        print(f"Total put contracts: {len(puts_df)}")
        print(f"Expiration dates: {puts_df['expirationDate'].nunique()}")
        print(f"Strike price range: ${puts_df['strike'].min():.2f} - ${puts_df['strike'].max():.2f}")
        
        # Most active puts (by volume)
        if 'volume' in puts_df.columns:
            top_puts = puts_df.nlargest(5, 'volume')[['strike', 'expirationDate', 'volume', 'openInterest', 'lastPrice']]
            print(f"\nTop 5 puts by volume:")
            print(top_puts.to_string(index=False))

def save_options_data(calls_df, puts_df, ticker_symbol, output_dir="."):
    """
    Save options data to CSV files.
    
    Args:
        calls_df (DataFrame): Calls data
        puts_df (DataFrame): Puts data
        ticker_symbol (str): Stock ticker symbol
        output_dir (str): Directory to save files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not calls_df.empty:
        calls_filename = f"{output_dir}/{ticker_symbol}_calls_{timestamp}.csv"
        calls_df.to_csv(calls_filename, index=False)
        print(f"Calls data saved to: {calls_filename}")
    
    if not puts_df.empty:
        puts_filename = f"{output_dir}/{ticker_symbol}_puts_{timestamp}.csv"
        puts_df.to_csv(puts_filename, index=False)
        print(f"Puts data saved to: {puts_filename}")

def main():
    """Main function to demonstrate usage."""
    # Example usage
    ticker_symbol = "AAPL"  # You can change this to any ticker
    
    try:
        # Fetch options data
        calls_df, puts_df = get_full_options_chain(ticker_symbol)
        
        if calls_df.empty and puts_df.empty:
            print("No options data retrieved. Exiting.")
            return
        
        # Analyze the data
        analyze_options_data(calls_df, puts_df, ticker_symbol)
        
        # Save to CSV
        save_options_data(calls_df, puts_df, ticker_symbol)
        
        # Preview data
        if not calls_df.empty:
            print(f"\nSample Calls (first 5 rows):")
            print(calls_df.head().to_string(index=False))
        
        if not puts_df.empty:
            print(f"\nSample Puts (first 5 rows):")
            print(puts_df.head().to_string(index=False))
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 