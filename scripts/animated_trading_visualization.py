import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import openpyxl
from datetime import datetime, timedelta
import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnimatedTradingVisualization:
    """
    Creates animated visualizations of trading data showing market vs Heston prices
    with trade entry/exit points marked on the charts.
    """
    
    def __init__(self, data_dir: str = "scripts/realtime_output/multi_company_sep19"):
        self.data_dir = data_dir
        self.companies = ['AMD', 'NVDA', 'CRCL', 'AUR', 'AVGO', 'BBAI', 'SLDB', 'SOFI', 'SOUN', 'TSLA']
        self.trade_data = {}
        self.hourly_data = {}
        
    def load_trade_logs(self) -> Dict:
        """
        Parse trading activity logs to extract trade entry/exit information
        """
        log_file = os.path.join(self.data_dir, "logs", "trading_activities.log")
        trade_data = {}
        
        if not os.path.exists(log_file):
            logger.warning(f"Log file not found: {log_file}")
            return trade_data
            
        logger.info(f"Loading trade data from: {log_file}")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_trade = None
        
        for line in lines:
            # Look for trade entry patterns - enhanced pattern matching
            if "TRADE ENTERED:" in line or "🎯 TRADE ENTERED:" in line:
                # Extract trade information
                match = re.search(r'TRADE ENTERED: (\w+) (\S+)', line)
                if match:
                    trade_type = match.group(1)  # BUY or SELL
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
                    logger.info(f"Found trade entry: {trade_type} {option_id}")
                    
            # Extract entry details
            elif current_trade and "Entry Time:" in line:
                time_match = re.search(r'Entry Time: (.+)', line)
                if time_match:
                    current_trade['entry_time'] = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
                    logger.info(f"Entry time: {current_trade['entry_time']}")
                    
            elif current_trade and "Entry Price:" in line:
                price_match = re.search(r'Entry Price: \$([\d.]+)', line)
                if price_match:
                    current_trade['entry_price'] = float(price_match.group(1))
                    logger.info(f"Entry price: ${current_trade['entry_price']}")
                    
            elif current_trade and "Entry Heston:" in line:
                heston_match = re.search(r'Entry Heston: \$([\d.]+)', line)
                if heston_match:
                    current_trade['entry_heston'] = float(heston_match.group(1))
                    logger.info(f"Entry Heston: ${current_trade['entry_heston']}")
                    
            # Look for trade exit patterns - enhanced pattern matching
            elif "TRADE EXITED:" in line or "💰 TRADE EXITED:" in line:
                match = re.search(r'TRADE EXITED: (\w+) (\S+)', line)
                if match:
                    trade_type = match.group(1)
                    option_id = match.group(2)
                    company = option_id.split('_')[0]
                    
                    # Find the corresponding active trade
                    if current_trade and current_trade['option_id'] == option_id:
                        current_trade['status'] = 'exited'
                        logger.info(f"Found trade exit: {trade_type} {option_id}")
                        
            # Extract exit details
            elif current_trade and current_trade['status'] == 'exited' and "Exit Time:" in line:
                time_match = re.search(r'Exit Time: (.+)', line)
                if time_match:
                    current_trade['exit_time'] = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
                    logger.info(f"Exit time: {current_trade['exit_time']}")
                    
            elif current_trade and current_trade['status'] == 'exited' and "Exit Price:" in line:
                price_match = re.search(r'Exit Price: \$([\d.]+)', line)
                if price_match:
                    current_trade['exit_price'] = float(price_match.group(1))
                    logger.info(f"Exit price: ${current_trade['exit_price']}")
                    
            elif current_trade and current_trade['status'] == 'exited' and "Exit Heston:" in line:
                heston_match = re.search(r'Exit Heston: \$([\d.]+)', line)
                if heston_match:
                    current_trade['exit_heston'] = float(heston_match.group(1))
                    logger.info(f"Exit Heston: ${current_trade['exit_heston']}")
                    
            elif current_trade and current_trade['status'] == 'exited' and "PnL:" in line:
                pnl_match = re.search(r'PnL: \$([-\d.]+)', line)
                if pnl_match:
                    current_trade['pnl'] = float(pnl_match.group(1))
                    logger.info(f"PnL: ${current_trade['pnl']}")
                    
                    # Store the completed trade
                    if current_trade['company'] not in trade_data:
                        trade_data[current_trade['company']] = []
                    trade_data[current_trade['company']].append(current_trade.copy())
                    logger.info(f"Completed trade stored for {current_trade['company']}: {current_trade['option_id']}")
                    current_trade = None
        
        # Store any remaining active trade
        if current_trade and current_trade['status'] == 'active':
            if current_trade['company'] not in trade_data:
                trade_data[current_trade['company']] = []
            trade_data[current_trade['company']].append(current_trade)
            logger.info(f"Active trade stored for {current_trade['company']}: {current_trade['option_id']}")
        
        total_trades = sum(len(trades) for trades in trade_data.values())
        logger.info(f"Loaded {total_trades} trades from logs")
        
        # Log summary of trades by company
        for company, trades in trade_data.items():
            logger.info(f"{company}: {len(trades)} trades")
            for trade in trades:
                logger.info(f"  - {trade['option_id']}: {trade['trade_type']} | Entry: ${trade.get('entry_price', 'N/A')} | Exit: ${trade.get('exit_price', 'N/A')} | PnL: ${trade.get('pnl', 'N/A')}")
        
        return trade_data
    
    def load_hourly_data(self) -> Dict:
        """
        Load hourly data from Excel files for each company
        """
        hourly_data = {}
        
        for company in self.companies:
            excel_file = os.path.join(self.data_dir, f"{company}_hourly_data.xlsx")
            
            if not os.path.exists(excel_file):
                logger.warning(f"Excel file not found: {excel_file}")
                continue
                
            logger.info(f"Loading hourly data for {company} from: {excel_file}")
            
            try:
                # Read all sheets from the Excel file
                excel_data = pd.read_excel(excel_file, sheet_name=None)
                
                company_data = []
                for sheet_name, sheet_data in excel_data.items():
                    if sheet_name.startswith('Hour_'):
                        # Extract hour and date from sheet name
                        parts = sheet_name.split('_')
                        if len(parts) >= 3:
                            hour = int(parts[1])
                            date_str = '_'.join(parts[2:])  # Handle date format
                            
                            # Add hour and date columns
                            sheet_data['Hour'] = hour
                            sheet_data['Date'] = date_str
                            sheet_data['Company'] = company
                            
                            company_data.append(sheet_data)
                
                if company_data:
                    # Combine all hourly data
                    hourly_data[company] = pd.concat(company_data, ignore_index=True)
                    logger.info(f"Loaded {len(hourly_data[company])} records for {company}")
                else:
                    logger.warning(f"No hourly data found for {company}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {company}: {e}")
                
        return hourly_data
    
    def create_animated_chart(self, company: str, save_path: str = None):
        """
        Create an animated chart for a specific company showing market vs Heston prices
        with trade entry/exit points
        """
        if company not in self.hourly_data:
            logger.error(f"No hourly data available for {company}")
            return
            
        data = self.hourly_data[company]
        trades = self.trade_data.get(company, [])
        
        if data.empty:
            logger.warning(f"No data available for {company}")
            return
            
        # Sort data by date and hour
        data = data.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Animated Trading Visualization - {company}', fontsize=16, fontweight='bold')
        
        # Initialize empty lines
        market_line, = ax1.plot([], [], 'b-', label='Market Price', linewidth=2)
        heston_line, = ax1.plot([], [], 'r-', label='Heston Price', linewidth=2)
        
        # Initialize scatter plots for trade points
        entry_scatter = ax1.scatter([], [], c='green', s=150, marker='^', label='Entry', alpha=0.9, edgecolors='black', linewidth=2)
        exit_scatter = ax1.scatter([], [], c='red', s=150, marker='v', label='Exit', alpha=0.9, edgecolors='black', linewidth=2)
        
        # Initialize PnL bar chart
        pnl_bars = ax2.bar([], [], color='green', alpha=0.7)
        
        # Set up axes
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{company} - Market vs Heston Prices')
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
                # Create datetime from date and hour
                date_obj = datetime.strptime(row['Date'], '%Y-%m-%d')
                time_obj = date_obj.replace(hour=row['Hour'])
                all_times.append(time_obj)
            except:
                continue
        
        if not all_times:
            logger.error(f"Could not parse time data for {company}")
            return
            
        time_range = (min(all_times), max(all_times))
        
        # Animation function
        def animate(frame):
            # Calculate current time based on frame
            total_frames = 100
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
                return market_line, heston_line, entry_scatter, exit_scatter, pnl_bars
            
            # Update price lines
            times = [d['time'] for d in current_data]
            market_prices = [d['market_price'] for d in current_data]
            heston_prices = [d['heston_price'] for d in current_data]
            
            market_line.set_data(times, market_prices)
            heston_line.set_data(times, heston_prices)
            
            # Update trade markers
            entry_times = []
            entry_prices = []
            exit_times = []
            exit_prices = []
            
            # Clear previous annotations
            for artist in ax1.texts:
                artist.remove()
            
            for trade in trades:
                if trade.get('entry_time') and trade['entry_time'] <= current_time:
                    entry_times.append(trade['entry_time'])
                    entry_prices.append(trade.get('entry_price', 0))
                    
                    # Add annotation for entry
                    ax1.annotate(f"{trade['trade_type']}\n{trade['option_id']}\n${trade.get('entry_price', 0):.2f}", 
                               xy=(trade['entry_time'], trade.get('entry_price', 0)),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                               fontsize=8, color='white', fontweight='bold')
                    
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
                               fontsize=8, color='white', fontweight='bold')
            
            if entry_times:
                entry_scatter.set_offsets(list(zip(entry_times, entry_prices)))
            if exit_times:
                exit_scatter.set_offsets(list(zip(exit_times, exit_prices)))
            
            # Update PnL chart
            completed_trades = [t for t in trades if t.get('exit_time') and t['exit_time'] <= current_time]
            if completed_trades:
                trade_labels = [f"Trade {i+1}" for i in range(len(completed_trades))]
                pnl_values = [t.get('pnl', 0) for t in completed_trades]
                
                # Clear and redraw bars
                ax2.clear()
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
        anim = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=False)
        
        # Save animation
        if save_path:
            output_path = os.path.join(save_path, f"{company}_animated_trading.gif")
            logger.info(f"Saving animation to: {output_path}")
            anim.save(output_path, writer='pillow', fps=5, dpi=100)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def create_static_summary_chart(self, company: str, save_path: str = None):
        """
        Create a static summary chart showing all trades and performance
        """
        if company not in self.hourly_data:
            logger.error(f"No hourly data available for {company}")
            return
            
        data = self.hourly_data[company]
        trades = self.trade_data.get(company, [])
        
        if data.empty:
            logger.warning(f"No data available for {company}")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Trading Summary - {company}', fontsize=16, fontweight='bold')
        
        # Plot 1: Price comparison over time
        data_sorted = data.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        times = []
        market_prices = []
        heston_prices = []
        
        for _, row in data_sorted.iterrows():
            try:
                date_obj = datetime.strptime(row['Date'], '%Y-%m-%d')
                time_obj = date_obj.replace(hour=row['Hour'])
                times.append(time_obj)
                market_prices.append(row.get('PX_LAST', 0))
                heston_prices.append(row.get('Heston_Price', 0))
            except:
                continue
        
        if times:
            ax1.plot(times, market_prices, 'b-', label='Market Price', linewidth=2)
            ax1.plot(times, heston_prices, 'r-', label='Heston Price', linewidth=2)
            
            # Mark trade points with annotations
            for i, trade in enumerate(trades):
                if trade.get('entry_time'):
                    ax1.scatter(trade['entry_time'], trade.get('entry_price', 0), 
                               c='green', s=150, marker='^', alpha=0.9, edgecolors='black', linewidth=2, 
                               label='Entry' if i == 0 else "")
                    
                    # Add entry annotation
                    ax1.annotate(f"{trade['trade_type']}\n{trade['option_id']}\n${trade.get('entry_price', 0):.2f}", 
                               xy=(trade['entry_time'], trade.get('entry_price', 0)),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                               fontsize=9, color='white', fontweight='bold')
                    
                if trade.get('exit_time'):
                    ax1.scatter(trade['exit_time'], trade.get('exit_price', 0), 
                               c='red', s=150, marker='v', alpha=0.9, edgecolors='black', linewidth=2, 
                               label='Exit' if i == 0 else "")
                    
                    # Add exit annotation
                    pnl = trade.get('pnl', 0)
                    color = 'green' if pnl > 0 else 'red'
                    ax1.annotate(f"EXIT\n{trade['option_id']}\n${trade.get('exit_price', 0):.2f}\nPnL: ${pnl:.2f}", 
                               xy=(trade['exit_time'], trade.get('exit_price', 0)),
                               xytext=(10, -20), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                               fontsize=9, color='white', fontweight='bold')
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price ($)')
            ax1.set_title('Price Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trade PnL
        completed_trades = [t for t in trades if t.get('pnl') is not None]
        if completed_trades:
            trade_numbers = list(range(1, len(completed_trades) + 1))
            pnl_values = [t['pnl'] for t in completed_trades]
            colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
            
            bars = ax2.bar(trade_numbers, pnl_values, color=colors, alpha=0.7)
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('PnL ($)')
            ax2.set_title('Trade PnL')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, pnl in zip(bars, pnl_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'${pnl:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 3: Cumulative PnL
        if completed_trades:
            cumulative_pnl = np.cumsum(pnl_values)
            ax3.plot(trade_numbers, cumulative_pnl, 'b-', linewidth=2, marker='o')
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('Cumulative PnL ($)')
            ax3.set_title('Cumulative PnL')
            ax3.grid(True, alpha=0.3)
            
            # Add final PnL annotation
            final_pnl = cumulative_pnl[-1]
            ax3.annotate(f'Total: ${final_pnl:.2f}', 
                        xy=(len(trade_numbers), final_pnl),
                        xytext=(len(trade_numbers)*0.8, final_pnl*1.2),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=12, fontweight='bold')
        
        # Plot 4: Trade statistics
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
            
            stats_data = {
                'Total Trades': len(completed_trades),
                'Winning Trades': len(winning_trades),
                'Losing Trades': len(losing_trades),
                'Win Rate': f"{len(winning_trades)/len(completed_trades)*100:.1f}%",
                'Total PnL': f"${sum(t['pnl'] for t in completed_trades):.2f}",
                'Avg Win': f"${np.mean([t['pnl'] for t in winning_trades]):.2f}" if winning_trades else "$0.00",
                'Avg Loss': f"${np.mean([t['pnl'] for t in losing_trades]):.2f}" if losing_trades else "$0.00"
            }
            
            # Create text box
            stats_text = '\n'.join([f"{k}: {v}" for k, v in stats_data.items()])
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('Trading Statistics')
            ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            output_path = os.path.join(save_path, f"{company}_trading_summary.png")
            logger.info(f"Saving summary chart to: {output_path}")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def run_visualization(self, output_dir: str = "scripts/visualization_output"):
        """
        Run the complete visualization process for all companies
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        logger.info("Loading trade data from logs...")
        self.trade_data = self.load_trade_logs()
        
        logger.info("Loading hourly data from Excel files...")
        self.hourly_data = self.load_hourly_data()
        
        # Create visualizations for each company
        for company in self.companies:
            if company in self.hourly_data and self.hourly_data[company] is not None:
                logger.info(f"Creating visualizations for {company}...")
                
                try:
                    # Create static summary chart
                    self.create_static_summary_chart(company, output_dir)
                    
                    # Create animated chart
                    self.create_animated_chart(company, output_dir)
                    
                except Exception as e:
                    logger.error(f"Error creating visualization for {company}: {e}")
        
        logger.info(f"Visualization complete! Output saved to: {output_dir}")

def main():
    """
    Main function to run the visualization
    """
    # Create visualization object
    viz = AnimatedTradingVisualization()
    
    # Run visualization
    viz.run_visualization()
    
    print("🎨 Trading visualization complete!")
    print("📊 Check the 'scripts/visualization_output' directory for generated charts")

if __name__ == "__main__":
    main() 