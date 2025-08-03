import yfinance as yf
import pandas as pd

def get_full_options_chain(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    
    print(f"Fetching data for: {ticker_symbol}")
    
    options_dates = ticker.options
    all_calls = []
    all_puts = []

    for exp_date in options_dates:
        try:
            opt = ticker.option_chain(exp_date)
            calls = opt.calls.copy()
            puts = opt.puts.copy()

            calls["expirationDate"] = exp_date
            puts["expirationDate"] = exp_date

            all_calls.append(calls)
            all_puts.append(puts)
        except Exception as e:
            print(f"Error fetching data for {exp_date}: {e}")

    # Combine all expirations
    df_calls = pd.concat(all_calls, ignore_index=True)
    df_puts = pd.concat(all_puts, ignore_index=True)

    return df_calls, df_puts

# Example usage:
if __name__ == "__main__":
    calls_df, puts_df = get_full_options_chain("AAPL")

    # Save to CSV (optional)
    calls_df.to_csv("AAPL_calls.csv", index=False)
    puts_df.to_csv("AAPL_puts.csv", index=False)

    # Preview
    print("\nSample Calls:\n", calls_df.head())
    print("\nSample Puts:\n", puts_df.head()) 