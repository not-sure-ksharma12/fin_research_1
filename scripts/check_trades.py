import re
from datetime import datetime
import os

def check_all_trades():
    """Check all trades from the logs"""
    log_file = "scripts/realtime_output/multi_company_sep19/logs/trading_activities.log"
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    trades = []
    current_trade = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # Look for trade entry
        if "TRADE ENTERED:" in line or "🎯 TRADE ENTERED:" in line:
            match = re.search(r'TRADE ENTERED: (\w+) (\S+)', line)
            if match:
                trade_type = match.group(1)
                option_id = match.group(2)
                company = option_id.split('_')[0]
                
                current_trade = {
                    'company': company,
                    'option_id': option_id,
                    'trade_type': trade_type,
                    'entry_time': None,
                    'entry_price': None,
                    'entry_heston': None,
                    'exit_time': None,
                    'exit_price': None,
                    'exit_heston': None,
                    'pnl': None,
                    'status': 'active'
                }
        
        # Extract entry details
        elif current_trade and "Entry Time:" in line:
            time_match = re.search(r'Entry Time: (.+)', line)
            if time_match:
                current_trade['entry_time'] = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
        
        elif current_trade and "Entry Price:" in line:
            price_match = re.search(r'Entry Price: \$([\d.]+)', line)
            if price_match:
                current_trade['entry_price'] = float(price_match.group(1))
        
        elif current_trade and "Entry Heston:" in line:
            heston_match = re.search(r'Entry Heston: \$([\d.]+)', line)
            if heston_match:
                current_trade['entry_heston'] = float(heston_match.group(1))
        
        # Look for trade exit
        elif "TRADE EXITED:" in line or "💰 TRADE EXITED:" in line:
            match = re.search(r'TRADE EXITED: (\w+) (\S+)', line)
            if match and current_trade and current_trade['option_id'] == match.group(2):
                current_trade['status'] = 'exited'
        
        # Extract exit details
        elif current_trade and current_trade['status'] == 'exited' and "Exit Time:" in line:
            time_match = re.search(r'Exit Time: (.+)', line)
            if time_match:
                current_trade['exit_time'] = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
        
        elif current_trade and current_trade['status'] == 'exited' and "Exit Price:" in line:
            price_match = re.search(r'Exit Price: \$([\d.]+)', line)
            if price_match:
                current_trade['exit_price'] = float(price_match.group(1))
        
        elif current_trade and current_trade['status'] == 'exited' and "Exit Heston:" in line:
            heston_match = re.search(r'Exit Heston: \$([\d.]+)', line)
            if heston_match:
                current_trade['exit_heston'] = float(heston_match.group(1))
        
        elif current_trade and current_trade['status'] == 'exited' and "PnL:" in line:
            pnl_match = re.search(r'PnL: \$([-\d.]+)', line)
            if pnl_match:
                current_trade['pnl'] = float(pnl_match.group(1))
                trades.append(current_trade.copy())
                current_trade = None
    
    # Add any remaining active trade
    if current_trade and current_trade['status'] == 'active':
        trades.append(current_trade)
    
    # Group trades by company
    companies = {}
    for trade in trades:
        company = trade['company']
        if company not in companies:
            companies[company] = []
        companies[company].append(trade)
    
    # Display results
    print("=" * 80)
    print("TRADE SUMMARY BY COMPANY")
    print("=" * 80)
    
    total_trades = 0
    for company, company_trades in companies.items():
        print(f"\n📊 {company}: {len(company_trades)} trades")
        print("-" * 50)
        
        for i, trade in enumerate(company_trades, 1):
            status = "✅ COMPLETED" if trade.get('pnl') is not None else "🔄 ACTIVE"
            print(f"  {i}. {trade['option_id']} - {trade['trade_type']}")
            print(f"     Entry: ${trade.get('entry_price', 'N/A')} | Exit: ${trade.get('exit_price', 'N/A')}")
            print(f"     PnL: ${trade.get('pnl', 'N/A')} | Status: {status}")
            if trade.get('entry_time'):
                print(f"     Entry Time: {trade['entry_time']}")
            if trade.get('exit_time'):
                print(f"     Exit Time: {trade['exit_time']}")
            print()
            total_trades += 1
    
    print("=" * 80)
    print(f"TOTAL TRADES: {total_trades}")
    print("=" * 80)
    
    return companies

if __name__ == "__main__":
    check_all_trades() 