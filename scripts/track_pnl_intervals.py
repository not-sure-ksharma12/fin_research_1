import pandas as pd
import time
from datetime import datetime
import os
from fetch_options_to_excel import connect_to_bloomberg, get_current_price

# List of input files (full paths)
STRIKES_FILES = [
    r'C:\Users\ksharma12\fin_research\scripts\excels\amd_strikes_below_market.xlsx',
    r'C:\Users\ksharma12\fin_research\scripts\excels\avgo_strikes_below_market.xlsx',
    r'C:\Users\ksharma12\fin_research\scripts\excels\bbai_strikes_below_market.xlsx',
    r'C:\Users\ksharma12\fin_research\scripts\excels\crcl_strikes_below_market.xlsx',
    r'C:\Users\ksharma12\fin_research\scripts\excels\nvda_strikes_below_market.xlsx',
    r'C:\Users\ksharma12\fin_research\scripts\excels\sound_strikes_below_market.xlsx',
    r'C:\Users\ksharma12\fin_research\scripts\excels\tsla_strikes_below_market.xlsx',
]
INTERVALS = {
    'hour': 3600,
    'day': 86400,
    'week': 604800
}

def fetch_latest_prices(session, tickers):
    prices = {}
    for ticker in tickers:
        prices[ticker] = get_current_price(session, ticker)
        time.sleep(0.1)
    return prices

def process_file(strikes_file, interval, session):
    base = os.path.splitext(os.path.basename(strikes_file))[0].replace('_strikes_below_market','')
    output_file = os.path.join(os.path.dirname(strikes_file), f'{base}_pnl_tracking.xlsx')
    log_file = os.path.join(os.path.dirname(strikes_file), f'{base}_pnl_log.xlsx')
    df = pd.read_excel(strikes_file)
    if 'Option Ticker' in df.columns:
        tickers = df['Option Ticker'].tolist()
    else:
        raise ValueError(f'Option Ticker column is required in the strikes file: {strikes_file}')
    strikes = df['Strike'].tolist()
    heston_prices = dict(zip(tickers, df['Heston_Price']))
    # Prepare output files
    if not os.path.exists(output_file):
        pd.DataFrame(columns=['Interval', 'Timestamp', 'Total_PnL']).to_excel(output_file, index=False)
    if not os.path.exists(log_file):
        pd.DataFrame(columns=['Interval', 'Timestamp', 'Option Ticker', 'Strike', 'Heston_Price', 'Latest_Market_Price', 'PnL']).to_excel(log_file, index=False)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    latest_prices = fetch_latest_prices(session, tickers)
    log_rows = []
    pnls = []
    for ticker, strike in zip(tickers, strikes):
        heston_price = heston_prices.get(ticker)
        latest_price = latest_prices.get(ticker)
        print(f"[{base.upper()}] Ticker: {ticker}, Heston: {heston_price}, Market: {latest_price}")
        if pd.notna(heston_price) and pd.notna(latest_price):
            pnl = latest_price - heston_price
            pnls.append(pnl)
            log_rows.append({
                'Interval': interval,
                'Timestamp': now,
                'Option Ticker': ticker,
                'Strike': strike,
                'Heston_Price': heston_price,
                'Latest_Market_Price': latest_price,
                'PnL': pnl
            })
    total_pnl = sum(pnls)
    print(f"[{base.upper()}] {now} | Interval: {interval} | Total PnL: {total_pnl:.2f}")
    # Append to summary Excel
    df_out = pd.read_excel(output_file)
    new_row = pd.DataFrame([{ 'Interval': interval, 'Timestamp': now, 'Total_PnL': total_pnl }])
    if not new_row.empty:
        df_out = pd.concat([
            df_out,
            new_row
        ], ignore_index=True)
        df_out.to_excel(output_file, index=False)
    # Append to log Excel
    if log_rows:
        df_log = pd.read_excel(log_file)
        log_df = pd.DataFrame(log_rows)
        if not log_df.empty:
            df_log = pd.concat([
                df_log,
                log_df
            ], ignore_index=True)
            df_log.to_excel(log_file, index=False)

def main(interval='hour'):
    if interval not in INTERVALS:
        raise ValueError(f"Interval must be one of {list(INTERVALS.keys())}")
    sleep_time = INTERVALS[interval]
    session = connect_to_bloomberg()
    print(f"Tracking PnL for all tickers at interval: {interval}")
    while True:
        for strikes_file in STRIKES_FILES:
            process_file(strikes_file, interval, session)
        time.sleep(sleep_time)

if __name__ == '__main__':
    import sys
    interval = sys.argv[1] if len(sys.argv) > 1 else 'hour'
    main(interval) 