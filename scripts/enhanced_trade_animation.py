import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import os
import re
import logging
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTradeAnimation:
    """
    Creates detailed animated visualizations with smooth curvy lines
    showing hourly price changes and precise trade entry/exit points
    """
    
    def __init__(self, data_dir: str = "scripts/realtime_output/multi_company_sep19"):
        self.data_dir = data_dir
        self.companies = ['AMD', 'NVDA', 'CRCL', 'AUR', 'AVGO', 'BBAI', 'SLDB', 'SOFI', 'SOUN', 'TSLA']
        
    def load_trades_from_company_logs(self):
        """Load trades from individual company log files"""
        all_trades = {}
        
        for company in self.companies:
            log_file = os.path.join(self.data_dir, f"{company}_trades.log")
            if not os.path.exists(log_file):
                continue
                
            company_trades = []
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                # Parse ENTER trades
                if "ENTER" in line:
                    match = re.search(r'\[(.+)\] ENTER - (\w+) - (\S+) - (\w+) - \$([\d.]+)', line)
                    if match:
                        timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                        company_name = match.group(2)
                        option_id = match.group(3)
                        trade_type = match.group(4)
                        position_size = float(match.group(5))
                        
                        trade = {
                            'company': company_name,
                            'option_id': option_id,
                            'trade_type': trade_type,
                            'entry_time': timestamp,
                            'position_size': position_size,
                            'entry_price': None,
                            'exit_time': None,
                            'exit_price': None,
                            'pnl': None,
                            'return_pct': None,
                            'status': 'active'
                        }
                        company_trades.append(trade)
                
                # Parse EXIT trades
                elif "EXIT" in line:
                    match = re.search(r'\[(.+)\] EXIT - (\w+) - (\S+) - (\w+) - \$([\d.]+) - PnL: \$([-\d.]+) - Return: ([-\d.]+)%', line)
                    if match:
                        timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                        company_name = match.group(2)
                        option_id = match.group(3)
                        trade_type = match.group(4)
                        position_size = float(match.group(5))
                        pnl = float(match.group(6)) if match.group(6) != 'nan' else 0.0
                        return_pct = float(match.group(7)) if match.group(7) != 'nan' else 0.0
                        
                        # Find the corresponding active trade
                        for trade in company_trades:
                            if (trade['option_id'] == option_id and 
                                trade['trade_type'] == trade_type and 
                                trade['status'] == 'active'):
                                trade['exit_time'] = timestamp
                                trade['pnl'] = pnl
                                trade['return_pct'] = return_pct
                                trade['status'] = 'completed'
                                break
            
            if company_trades:
                all_trades[company] = company_trades
                logger.info(f"Loaded {len(company_trades)} trades for {company}")
        
        return all_trades
    
    def load_hourly_data(self, company):
        """Load hourly data for a specific company"""
        excel_file = os.path.join(self.data_dir, f"{company}_hourly_data.xlsx")
        
        if not os.path.exists(excel_file):
            logger.warning(f"Excel file not found: {excel_file}")
            return None
            
        try:
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
            
            if company_data:
                data = pd.concat(company_data, ignore_index=True)
                data = data.sort_values(['Date', 'Hour']).reset_index(drop=True)
                return data
            else:
                logger.warning(f"No hourly data found for {company}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data for {company}: {e}")
            return None
    
    def create_smooth_curves(self, times, prices, smoothness=0.1):
        """Create smooth curves using spline interpolation"""
        if len(times) < 2:
            return times, prices
        
        # Remove duplicates by keeping the first occurrence
        unique_data = {}
        for i, t in enumerate(times):
            if t not in unique_data:
                unique_data[t] = prices[i]
        
        unique_times = list(unique_data.keys())
        unique_prices = list(unique_data.values())
        
        if len(unique_times) < 2:
            return times, prices
            
        # Convert times to numeric for interpolation
        time_nums = [(t - unique_times[0]).total_seconds() for t in unique_times]
        
        # Create smooth curve using spline interpolation
        if len(time_nums) > 3:
            try:
                # Use cubic spline for smooth curves
                f = interp1d(time_nums, unique_prices, kind='cubic', bounds_error=False, fill_value='extrapolate')
                
                # Create more points for smoother curves
                num_points = max(100, len(unique_times) * 10)
                smooth_time_nums = np.linspace(min(time_nums), max(time_nums), num_points)
                smooth_prices = f(smooth_time_nums)
                
                # Convert back to datetime
                smooth_times = [unique_times[0] + timedelta(seconds=t) for t in smooth_time_nums]
            except:
                # Fallback to linear interpolation
                f = interp1d(time_nums, unique_prices, kind='linear', bounds_error=False, fill_value='extrapolate')
                num_points = max(50, len(unique_times) * 5)
                smooth_time_nums = np.linspace(min(time_nums), max(time_nums), num_points)
                smooth_prices = f(smooth_time_nums)
                smooth_times = [unique_times[0] + timedelta(seconds=t) for t in smooth_time_nums]
        else:
            # Use linear interpolation for few points
            f = interp1d(time_nums, unique_prices, kind='linear', bounds_error=False, fill_value='extrapolate')
            num_points = max(50, len(unique_times) * 5)
            smooth_time_nums = np.linspace(min(time_nums), max(time_nums), num_points)
            smooth_prices = f(smooth_time_nums)
            smooth_times = [unique_times[0] + timedelta(seconds=t) for t in smooth_time_nums]
        
        return smooth_times, smooth_prices
    
    def create_detailed_animation(self, company, trades, save_path=None):
        """Create detailed animated visualization with smooth curves"""
        # Load data
        data = self.load_hourly_data(company)
        
        if data is None or data.empty:
            logger.error(f"No data available for {company}")
            return
        
        # Prepare time series data
        times = []
        market_prices = []
        heston_prices = []
        
        for _, row in data.iterrows():
            try:
                date_obj = datetime.strptime(row['Date'], '%Y-%m-%d')
                time_obj = date_obj.replace(hour=row['Hour'])
                times.append(time_obj)
                market_prices.append(row.get('PX_LAST', 0))
                heston_prices.append(row.get('Heston_Price', 0))
            except:
                continue
        
        if not times:
            logger.error(f"Could not parse time data for {company}")
            return
        
        # Create smooth curves
        smooth_times, smooth_market_prices = self.create_smooth_curves(times, market_prices)
        smooth_times, smooth_heston_prices = self.create_smooth_curves(times, heston_prices)
        
        # Create figure with enhanced styling
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        fig.patch.set_facecolor('#1a1a1a')
        
        # Set up main price chart
        ax1.set_facecolor('#2a2a2a')
        ax1.grid(True, alpha=0.2, color='white')
        ax1.set_xlabel('Time', fontsize=12, color='white')
        ax1.set_ylabel('Price ($)', fontsize=12, color='white')
        ax1.set_title(f'{company} - Detailed Price Animation with Trade Points', 
                     fontsize=16, fontweight='bold', color='white', pad=20)
        
        # Set up PnL chart
        ax2.set_facecolor('#2a2a2a')
        ax2.grid(True, alpha=0.2, color='white')
        ax2.set_xlabel('Time', fontsize=12, color='white')
        ax2.set_ylabel('PnL ($)', fontsize=12, color='white')
        ax2.set_title('Trade PnL Over Time', fontsize=14, fontweight='bold', color='white')
        
        # Initialize lines with enhanced styling
        market_line, = ax1.plot([], [], color='#00ff88', linewidth=3, 
                               label='Market Price', alpha=0.9)
        heston_line, = ax1.plot([], [], color='#ff6b6b', linewidth=3, 
                               label='Heston Price', alpha=0.9)
        
        # Initialize trade markers
        entry_scatter = ax1.scatter([], [], c='#00ff00', s=200, marker='^', 
                                   alpha=0.9, edgecolors='white', linewidth=3, 
                                   label='Entry', zorder=10)
        exit_scatter = ax1.scatter([], [], c='#ff0000', s=200, marker='v', 
                                  alpha=0.9, edgecolors='white', linewidth=3, 
                                  label='Exit', zorder=10)
        
        # Initialize PnL line
        pnl_line, = ax2.plot([], [], color='#ffd700', linewidth=2, marker='o', 
                            markersize=6, alpha=0.8)
        
        # Add legends
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.8)
        
        # Set time range
        time_range = (min(times), max(times))
        
        # Animation function
        def animate(frame):
            total_frames = 200  # More frames for smoother animation
            progress = frame / total_frames
            current_time = time_range[0] + (time_range[1] - time_range[0]) * progress
            
            # Filter data up to current time
            current_smooth_times = []
            current_market_prices = []
            current_heston_prices = []
            
            for i, t in enumerate(smooth_times):
                if t <= current_time:
                    current_smooth_times.append(t)
                    current_market_prices.append(smooth_market_prices[i])
                    current_heston_prices.append(smooth_heston_prices[i])
            
            # Update price lines
            if current_smooth_times:
                market_line.set_data(current_smooth_times, current_market_prices)
                heston_line.set_data(current_smooth_times, current_heston_prices)
                
                # Update axis limits dynamically
                all_prices = current_market_prices + current_heston_prices
                if all_prices:
                    price_min = min(all_prices) * 0.98
                    price_max = max(all_prices) * 1.02
                    ax1.set_ylim(price_min, price_max)
            
            ax1.set_xlim(time_range[0], time_range[1])
            
            # Clear previous annotations
            for artist in ax1.texts:
                artist.remove()
            
            # Update trade markers with enhanced styling
            entry_times = []
            entry_prices = []
            exit_times = []
            exit_prices = []
            pnl_times = []
            pnl_values = []
            
            for trade in trades:
                if trade.get('entry_time') and trade['entry_time'] <= current_time:
                    entry_times.append(trade['entry_time'])
                    entry_prices.append(trade.get('entry_price', 0))
                    
                    # Add detailed entry annotation
                    ax1.annotate(f"🎯 ENTRY\n{trade['trade_type']}\n{trade['option_id']}\n"
                               f"Size: ${trade.get('position_size', 0):.2f}", 
                               xy=(trade['entry_time'], trade.get('entry_price', 0)),
                               xytext=(15, 15), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='#00ff00', 
                                       alpha=0.9, edgecolor='white', linewidth=2),
                               fontsize=9, color='black', fontweight='bold',
                               ha='center', va='bottom')
                    
                    # Add entry pulse effect
                    circle = Circle((trade['entry_time'], trade.get('entry_price', 0)), 
                                  0.5, fill=False, color='#00ff00', linewidth=2, alpha=0.7)
                    ax1.add_patch(circle)
                
                if trade.get('exit_time') and trade['exit_time'] <= current_time:
                    exit_times.append(trade['exit_time'])
                    exit_prices.append(trade.get('exit_price', 0))
                    
                    # Add detailed exit annotation
                    pnl = trade.get('pnl', 0)
                    color = '#00ff00' if pnl > 0 else '#ff0000'
                    pnl_times.append(trade['exit_time'])
                    pnl_values.append(pnl)
                    
                    ax1.annotate(f"💰 EXIT\n{trade['option_id']}\n"
                               f"PnL: ${pnl:.2f}\n"
                               f"Return: {trade.get('return_pct', 0):.1f}%", 
                               xy=(trade['exit_time'], trade.get('exit_price', 0)),
                               xytext=(15, -25), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, 
                                       alpha=0.9, edgecolor='white', linewidth=2),
                               fontsize=9, color='black', fontweight='bold',
                               ha='center', va='top')
                    
                    # Add exit pulse effect
                    circle = Circle((trade['exit_time'], trade.get('exit_price', 0)), 
                                  0.5, fill=False, color=color, linewidth=2, alpha=0.7)
                    ax1.add_patch(circle)
            
            # Update scatter plots
            if entry_times:
                entry_scatter.set_offsets(list(zip(entry_times, entry_prices)))
            if exit_times:
                exit_scatter.set_offsets(list(zip(exit_times, exit_prices)))
            
            # Update PnL chart
            if pnl_times:
                pnl_line.set_data(pnl_times, pnl_values)
                ax2.set_xlim(time_range[0], time_range[1])
                
                if pnl_values:
                    pnl_min = min(pnl_values) * 1.1 if min(pnl_values) < 0 else min(pnl_values) * 0.9
                    pnl_max = max(pnl_values) * 1.1 if max(pnl_values) > 0 else max(pnl_values) * 0.9
                    ax2.set_ylim(pnl_min, pnl_max)
                    
                    # Add PnL annotations
                    for i, (t, pnl) in enumerate(zip(pnl_times, pnl_values)):
                        color = '#00ff00' if pnl > 0 else '#ff0000'
                        ax2.annotate(f'${pnl:.2f}', xy=(t, pnl), xytext=(0, 10),
                                   textcoords='offset points', ha='center', va='bottom',
                                   fontsize=8, color=color, fontweight='bold')
            
            # Add time indicator
            time_text = ax1.text(0.02, 0.98, f'Time: {current_time.strftime("%Y-%m-%d %H:%M")}', 
                               transform=ax1.transAxes, fontsize=10, color='white',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8),
                               verticalalignment='top')
            
            return market_line, heston_line, entry_scatter, exit_scatter, pnl_line
        
        # Create animation with more frames and slower speed
        anim = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=False, repeat=True)
        
        # Save animation
        if save_path:
            output_path = os.path.join(save_path, f"{company}_enhanced_curvy_animation.gif")
            logger.info(f"Saving enhanced animation to: {output_path}")
            anim.save(output_path, writer='pillow', fps=10, dpi=150)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def run_for_all_companies(self, output_dir="scripts/visualization_output"):
        """Run enhanced animation for all companies with trade data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all trades
        all_trades = self.load_trades_from_company_logs()
        
        print("=" * 80)
        print("ENHANCED TRADE ANIMATION SUMMARY")
        print("=" * 80)
        
        for company, trades in all_trades.items():
            completed_trades = [t for t in trades if t.get('status') == 'completed']
            active_trades = [t for t in trades if t.get('status') == 'active']
            
            print(f"\n📊 {company}: {len(trades)} total trades")
            print(f"   ✅ Completed: {len(completed_trades)}")
            print(f"   🔄 Active: {len(active_trades)}")
            
            if completed_trades:
                total_pnl = sum(t.get('pnl', 0) for t in completed_trades)
                print(f"   💰 Total PnL: ${total_pnl:.2f}")
                
                # Create animation for companies with completed trades
                logger.info(f"Creating enhanced animation for {company}...")
                try:
                    self.create_detailed_animation(company, trades, output_dir)
                    logger.info(f"✅ Enhanced animation completed for {company}")
                except Exception as e:
                    logger.error(f"❌ Error creating animation for {company}: {e}")
        
        print("\n" + "=" * 80)
        print("🎨 Enhanced curvy animations complete!")
        print("📊 Check the 'scripts/visualization_output' directory for animated GIFs")
        print("=" * 80)

def main():
    """Main function to run the enhanced animation"""
    animator = EnhancedTradeAnimation()
    animator.run_for_all_companies()

if __name__ == "__main__":
    main() 