import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_soun_animation():
    """
    Create animated visualization for SOUN with trade data
    """
    # Load SOUN data
    data_dir = "scripts/realtime_output/multi_company_sep19"
    excel_file = os.path.join(data_dir, "SOUN_hourly_data.xlsx")
    
    if not os.path.exists(excel_file):
        logger.error(f"Excel file not found: {excel_file}")
        return
    
    # Load hourly data
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    company_data = []
    
    for sheet_name, sheet_data in excel_data.items():
        if sheet_name.startswith('Hour_'):
            parts = sheet_name.split('_')
            if len(parts) >= 3:
                hour = int(parts[1])
                date_str = '_'.join(parts[2:])
                sheet_data['Hour'] = hour
                sheet_data['Date'] = date_str
                company_data.append(sheet_data)
    
    if not company_data:
        logger.error("No hourly data found for SOUN")
        return
    
    data = pd.concat(company_data, ignore_index=True)
    data = data.sort_values(['Date', 'Hour']).reset_index(drop=True)
    
    # Trade data from logs (manually extracted)
    trades = [{
        'option_id': 'SOUN_3.0_Call',
        'trade_type': 'SELL',
        'entry_time': datetime(2025, 8, 8, 8, 35, 36),
        'entry_price': 11.17,
        'entry_heston': 10.83,
        'exit_time': datetime(2025, 8, 8, 8, 45, 49),
        'exit_price': 11.17,
        'exit_heston': 10.55,
        'pnl': 0.0
    }]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Animated Trading Visualization - SOUN', fontsize=16, fontweight='bold')
    
    # Initialize lines
    market_line, = ax1.plot([], [], 'b-', label='Market Price', linewidth=2)
    heston_line, = ax1.plot([], [], 'r-', label='Heston Price', linewidth=2)
    
    # Initialize scatter plots
    entry_scatter = ax1.scatter([], [], c='green', s=150, marker='^', label='Entry', alpha=0.9, edgecolors='black', linewidth=2)
    exit_scatter = ax1.scatter([], [], c='red', s=150, marker='v', label='Exit', alpha=0.9, edgecolors='black', linewidth=2)
    
    # Set up axes
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('SOUN - Market vs Heston Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Trade')
    ax2.set_ylabel('PnL ($)')
    ax2.set_title('Trade PnL')
    ax2.grid(True, alpha=0.3)
    
    # Get time range
    all_times = []
    for _, row in data.iterrows():
        try:
            date_obj = datetime.strptime(row['Date'], '%Y-%m-%d')
            time_obj = date_obj.replace(hour=row['Hour'])
            all_times.append(time_obj)
        except:
            continue
    
    if not all_times:
        logger.error("Could not parse time data")
        return
    
    time_range = (min(all_times), max(all_times))
    
    # Animation function
    def animate(frame):
        total_frames = 50  # Reduced for faster generation
        progress = frame / total_frames
        current_time = time_range[0] + (time_range[1] - time_range[0]) * progress
        
        # Filter data up to current time
        current_data = []
        for _, row in data.iterrows():
            try:
                date_obj = datetime.strptime(row['Date'], '%Y-%m-%d')
                time_obj = date_obj.replace(hour=row['Hour'])
                if time_obj <= current_time:
                    current_data.append({
                        'time': time_obj,
                        'market_price': row.get('PX_LAST', 0),
                        'heston_price': row.get('Heston_Price', 0)
                    })
            except:
                continue
        
        if not current_data:
            return market_line, heston_line, entry_scatter, exit_scatter
        
        # Update price lines
        times = [d['time'] for d in current_data]
        market_prices = [d['market_price'] for d in current_data]
        heston_prices = [d['heston_price'] for d in current_data]
        
        market_line.set_data(times, market_prices)
        heston_line.set_data(times, heston_prices)
        
        # Clear previous annotations
        for artist in ax1.texts:
            artist.remove()
        
        # Update trade markers
        entry_times = []
        entry_prices = []
        exit_times = []
        exit_prices = []
        
        for trade in trades:
            if trade.get('entry_time') and trade['entry_time'] <= current_time:
                entry_times.append(trade['entry_time'])
                entry_prices.append(trade.get('entry_price', 0))
                
                # Add annotation for entry
                ax1.annotate(f"{trade['trade_type']}\n{trade['option_id']}\n${trade.get('entry_price', 0):.2f}", 
                           xy=(trade['entry_time'], trade.get('entry_price', 0)),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                           fontsize=10, color='white', fontweight='bold')
                
            if trade.get('exit_time') and trade['exit_time'] <= current_time:
                exit_times.append(trade['exit_time'])
                exit_prices.append(trade.get('exit_price', 0))
                
                # Add annotation for exit
                pnl = trade.get('pnl', 0)
                color = 'green' if pnl > 0 else 'red'
                ax1.annotate(f"EXIT\n{trade['option_id']}\n${trade.get('exit_price', 0):.2f}\nPnL: ${pnl:.2f}", 
                           xy=(trade['exit_time'], trade.get('exit_price', 0)),
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                           fontsize=10, color='white', fontweight='bold')
        
        if entry_times:
            entry_scatter.set_offsets(list(zip(entry_times, entry_prices)))
        if exit_times:
            exit_scatter.set_offsets(list(zip(exit_times, exit_prices)))
        
        # Update PnL chart
        completed_trades = [t for t in trades if t.get('exit_time') and t['exit_time'] <= current_time]
        if completed_trades:
            ax2.clear()
            trade_labels = [f"Trade {i+1}" for i in range(len(completed_trades))]
            pnl_values = [t.get('pnl', 0) for t in completed_trades]
            
            bars = ax2.bar(trade_labels, pnl_values, 
                          color=['green' if pnl > 0 else 'red' for pnl in pnl_values],
                          alpha=0.7)
            ax2.set_xlabel('Trade')
            ax2.set_ylabel('PnL ($)')
            ax2.set_title('Trade PnL')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, pnl in zip(bars, pnl_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'${pnl:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Update axis limits
        if market_prices:
            ax1.set_ylim(min(min(market_prices), min(heston_prices)) * 0.95,
                       max(max(market_prices), max(heston_prices)) * 1.05)
        ax1.set_xlim(time_range[0], time_range[1])
        
        return market_line, heston_line, entry_scatter, exit_scatter
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=50, interval=200, blit=False)
    
    # Save animation
    output_path = "scripts/visualization_output/SOUN_animated_trading.gif"
    logger.info(f"Saving animation to: {output_path}")
    anim.save(output_path, writer='pillow', fps=3, dpi=100)
    
    plt.tight_layout()
    plt.show()
    
    logger.info("Animation created successfully!")

if __name__ == "__main__":
    create_soun_animation() 