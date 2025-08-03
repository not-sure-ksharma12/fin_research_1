import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Optional
import QuantLib as ql

# Add scripts directory to path
sys.path.append(r"C:\Users\ksharma12\fin_research\scripts")

# Import the existing modules
from fetch_options_to_excel import (
    connect_to_bloomberg, get_option_chain, parse_option_ticker, 
    get_current_price, fetch_option_data, save_with_formatting
)
from heston_calculator import calibrate_heston, heston_price_row

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NVDARealTimeTrading:
    def __init__(self, initial_capital: float = 100.0):
        """
        Initialize NVDA real-time Heston mispricing trading system
        
        Args:
            initial_capital: Initial capital allocation
        """
        self.company = 'NVDA'
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.active_trades = {}  # {option_id: trade_info}
        self.trade_history = []
        self.session = None
        self.last_processed_time = None
        self.is_running = False
        
        # Trade tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        
        # Create output directories
        self.output_dir = f"scripts/realtime_output/{self.company}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize trade log file
        self.trade_log_file = f"{self.output_dir}/{self.company}_realtime_trades.xlsx"
        self.summary_file = f"{self.output_dir}/{self.company}_realtime_summary.xlsx"
        
        logger.info(f"Real-time trading system initialized for {self.company} with ${initial_capital} capital")
    
    def connect_bloomberg(self):
        """Connect to Bloomberg Terminal"""
        try:
            self.session = connect_to_bloomberg()
            logger.info("Successfully connected to Bloomberg Terminal")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Bloomberg: {e}")
            return False
    
    def fetch_live_data(self, expiry_date: str = "2025-12-19") -> pd.DataFrame:
        """
        Fetch live NVDA options data from Bloomberg
        
        Args:
            expiry_date: Options expiration date
            
        Returns:
            DataFrame with current options data including Heston prices
        """
        try:
            logger.info(f"Fetching live {self.company} options data for expiry {expiry_date}")
            
            # Get current option chain
            chain = get_option_chain(self.session, self.company)
            
            # Select options for the given expiry
            all_options = []
            for option in chain:
                parsed = parse_option_ticker(option)
                if parsed and parsed["Expiration"] == expiry_date:
                    all_options.append((option, parsed["Strike"], parsed["Option Type"]))
            
            logger.info(f"Found {len(all_options)} live options for {self.company}")
            
            # Get current stock price
            underlying_ticker = f"{self.company} US Equity"
            current_price = get_current_price(self.session, underlying_ticker)
            
            if not current_price:
                logger.warning(f"Could not get current price for {self.company}")
                return pd.DataFrame()
            
            # Fetch live option data
            option_data = fetch_option_data(self.session, all_options, current_price=current_price)
            df = pd.DataFrame(option_data)
            
            if df.empty:
                logger.warning(f"No live option data retrieved for {self.company}")
                return df
            
            # Ensure current price is filled
            df["Current Price"] = current_price
            
            # Sort by strike price
            df_calls = df[df["Option Type"] == "Call"].sort_values(by=["Strike"], ascending=True)
            df_puts = df[df["Option Type"] == "Put"].sort_values(by=["Strike"], ascending=True)
            df_sorted = pd.concat([df_calls, df_puts], ignore_index=True)
            
            # Calculate live Heston prices
            df_with_heston = self.calculate_live_heston_prices(df_sorted)
            
            # Add current timestamp
            current_time = datetime.now()
            df_with_heston['Timestamp'] = current_time
            df_with_heston['Hour'] = current_time.hour
            
            return df_with_heston
            
        except Exception as e:
            logger.error(f"Error fetching live {self.company} data: {e}")
            return pd.DataFrame()
    
    def calculate_live_heston_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate live Heston model prices for options
        
        Args:
            df: DataFrame with live options data
            
        Returns:
            DataFrame with live Heston prices added
        """
        try:
            if df.empty:
                return df
            
            # Prepare data for Heston calculation
            valuation_date = datetime.today()
            risk_free_rate = 0.0453  # 4.53% from Bloomberg
            
            # Ensure required columns exist
            if "Implied Volatility" not in df.columns or df["Implied Volatility"].isna().all():
                logger.warning(f"No implied volatility data for {self.company}, using default")
                df["Implied Volatility"] = 0.3 * 100  # 30% default
            
            # Calibrate Heston model with live data
            params, heston_model, helpers, model_prices, market_prices = calibrate_heston(
                df, valuation_date, risk_free_rate=risk_free_rate
            )
            
            logger.info(f"Live Heston params for {self.company}: v0={params[0]:.4f}, kappa={params[1]:.4f}, theta={params[2]:.4f}, sigma={params[3]:.4f}, rho={params[4]:.4f}")
            
            # Calculate live Heston prices for all options
            heston_results = df.apply(
                lambda row: heston_price_row(row, heston_model, valuation_date, risk_free_rate), 
                axis=1
            )
            
            # Add Heston prices to DataFrame
            df_with_heston = pd.concat([df, heston_results], axis=1)
            
            # Calculate live mispricing metrics
            df_with_heston['Market_vs_Heston'] = df_with_heston['PX_LAST'] - df_with_heston['Heston_Price']
            df_with_heston['Heston_vs_Market'] = df_with_heston['Heston_Price'] - df_with_heston['PX_LAST']
            
            # Add unique option identifier
            df_with_heston['Option_ID'] = df_with_heston.apply(
                lambda row: f"{self.company}_{row['Strike']}_{row['Option Type']}", 
                axis=1
            )
            
            return df_with_heston
            
        except Exception as e:
            logger.error(f"Error calculating live Heston prices for {self.company}: {e}")
            return df
    
    def select_live_trading_opportunities(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Select the 2 most undervalued and 2 most overvalued call options from live data
        
        Args:
            data: Current live options data
            
        Returns:
            Tuple of (undervalued_option_ids, overvalued_option_ids)
        """
        # Filter for call options only
        call_data = data[data['Option Type'] == 'Call'].copy()
        
        # Filter for valid data (non-null prices)
        valid_data = call_data.dropna(subset=['PX_LAST', 'Heston_Price'])
        
        if len(valid_data) < 4:
            logger.warning(f"Insufficient live data for {self.company}: only {len(valid_data)} valid call options")
            return [], []
        
        # Find undervalued options (Market < Heston)
        undervalued = valid_data[valid_data['Market_vs_Heston'] < 0].copy()
        undervalued = undervalued.sort_values('Heston_vs_Market', ascending=False)
        
        # Find overvalued options (Market > Heston)
        overvalued = valid_data[valid_data['Market_vs_Heston'] > 0].copy()
        overvalued = overvalued.sort_values('Market_vs_Heston', ascending=False)
        
        # Select top 2 from each category
        undervalued_ids = undervalued['Option_ID'].head(2).tolist()
        overvalued_ids = overvalued['Option_ID'].head(2).tolist()
        
        logger.info(f"Live selection for {self.company}: {len(undervalued_ids)} undervalued, {len(overvalued_ids)} overvalued options")
        
        return undervalued_ids, overvalued_ids
    
    def enter_live_trade(self, option_id: str, trade_type: str, data: pd.DataFrame) -> bool:
        """
        Enter a new live trade if not already in position
        
        Args:
            option_id: Unique option identifier
            trade_type: 'BUY' for undervalued, 'SELL' for overvalued
            data: Current live data
            
        Returns:
            True if trade entered, False otherwise
        """
        if option_id in self.active_trades:
            logger.debug(f"Already in live trade for {option_id}")
            return False
        
        option_data = data[data['Option_ID'] == option_id].iloc[0]
        
        # Calculate position size (equal allocation among active trades)
        num_active_trades = len(self.active_trades) + 1
        position_size = self.current_capital / num_active_trades
        
        # Calculate number of contracts (assuming $100 per contract)
        contracts = int(position_size / 100)
        if contracts == 0:
            contracts = 1  # Minimum 1 contract
        
        trade_info = {
            'option_id': option_id,
            'trade_type': trade_type,
            'entry_time': option_data['Timestamp'],
            'entry_hour': option_data['Hour'],
            'entry_market_price': option_data['PX_LAST'],
            'entry_heston_price': option_data['Heston_Price'],
            'strike_price': option_data['Strike'],
            'stock_price': option_data['Current Price'],
            'contracts': contracts,
            'position_size': position_size,
            'entry_mispricing': option_data['Market_vs_Heston']
        }
        
        self.active_trades[option_id] = trade_info
        self.total_trades += 1
        
        # Enhanced logging for trade entry
        logger.info("=" * 80)
        logger.info(f"🎯 TRADE ENTERED: {trade_type} {option_id}")
        logger.info(f"   Entry Time: {option_data['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Entry Price: ${option_data['PX_LAST']:.2f}")
        logger.info(f"   Heston Price: ${option_data['Heston_Price']:.2f}")
        logger.info(f"   Strike Price: ${option_data['Strike']:.2f}")
        logger.info(f"   Stock Price: ${option_data['Current Price']:.2f}")
        logger.info(f"   Contracts: {contracts}")
        logger.info(f"   Position Size: ${position_size:.2f}")
        logger.info(f"   Mispricing: ${option_data['Market_vs_Heston']:.2f}")
        logger.info(f"   Active Trades: {len(self.active_trades)}")
        logger.info(f"   Current Capital: ${self.current_capital:.2f}")
        logger.info("=" * 80)
        
        return True
    
    def check_live_exit_conditions(self, option_id: str, data: pd.DataFrame) -> bool:
        """
        Check if exit conditions are met for a live trade
        
        Args:
            option_id: Option identifier
            data: Current live data
            
        Returns:
            True if should exit, False otherwise
        """
        if option_id not in self.active_trades:
            return False
        
        trade = self.active_trades[option_id]
        option_data = data[data['Option_ID'] == option_id]
        
        if len(option_data) == 0:
            return False
        
        current_data = option_data.iloc[0]
        current_market_price = current_data['PX_LAST']
        current_heston_price = current_data['Heston_Price']
        
        # Exit conditions
        if trade['trade_type'] == 'BUY':
            # Exit when Market >= Heston (no longer undervalued)
            should_exit = current_market_price >= current_heston_price
        else:  # SELL
            # Exit when Market <= Heston (no longer overvalued)
            should_exit = current_market_price <= current_heston_price
        
        return should_exit
    
    def exit_live_trade(self, option_id: str, data: pd.DataFrame) -> bool:
        """
        Exit a live trade and calculate PnL
        
        Args:
            option_id: Option identifier
            data: Current live data
            
        Returns:
            True if trade exited, False otherwise
        """
        if option_id not in self.active_trades:
            return False
        
        trade = self.active_trades[option_id]
        option_data = data[data['Option_ID'] == option_id]
        
        if len(option_data) == 0:
            return False
        
        current_data = option_data.iloc[0]
        current_market_price = current_data['PX_LAST']
        
        # Calculate PnL
        if trade['trade_type'] == 'BUY':
            # Long position: profit = (exit_price - entry_price) * contracts
            pnl = (current_market_price - trade['entry_market_price']) * trade['contracts']
        else:  # SELL
            # Short position: profit = (entry_price - exit_price) * contracts
            pnl = (trade['entry_market_price'] - current_market_price) * trade['contracts']
        
        # Update trade info
        trade.update({
            'exit_time': current_data['Timestamp'],
            'exit_hour': current_data['Hour'],
            'exit_market_price': current_market_price,
            'exit_heston_price': current_data['Heston_Price'],
            'pnl': pnl,
            'return_pct': (pnl / trade['position_size']) * 100,
            'duration_hours': current_data['Hour'] - trade['entry_hour']
        })
        
        # Update portfolio statistics
        self.total_pnl += pnl
        if pnl > 0:
            self.profitable_trades += 1
        
        # Update capital
        self.current_capital += pnl
        
        # Move to trade history
        self.trade_history.append(trade)
        del self.active_trades[option_id]
        
        # Enhanced logging for trade exit
        logger.info("=" * 80)
        logger.info(f"💰 TRADE EXITED: {trade['trade_type']} {option_id}")
        logger.info(f"   Exit Time: {current_data['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Entry Time: {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Duration: {trade['duration_hours']} hours")
        logger.info(f"   Entry Price: ${trade['entry_market_price']:.2f}")
        logger.info(f"   Exit Price: ${current_market_price:.2f}")
        logger.info(f"   Entry Heston: ${trade['entry_heston_price']:.2f}")
        logger.info(f"   Exit Heston: ${current_data['Heston_Price']:.2f}")
        logger.info(f"   PnL: ${pnl:.2f}")
        logger.info(f"   Return: {trade['return_pct']:.2f}%")
        logger.info(f"   Contracts: {trade['contracts']}")
        logger.info(f"   Active Trades: {len(self.active_trades)}")
        logger.info(f"   Current Capital: ${self.current_capital:.2f}")
        logger.info(f"   Total PnL: ${self.total_pnl:.2f}")
        logger.info("=" * 80)
        
        return True
    
    def process_live_hour(self):
        """
        Process one live trading hour
        """
        current_time = datetime.now()
        logger.info(f"Processing live hour for {self.company} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Fetch fresh live data
        live_data = self.fetch_live_data()
        if live_data.empty:
            logger.error("No live data available, skipping this hour")
            return
        
        # Check exit conditions for existing trades
        options_to_exit = []
        for option_id in list(self.active_trades.keys()):
            if self.check_live_exit_conditions(option_id, live_data):
                options_to_exit.append(option_id)
        
        # Exit trades
        for option_id in options_to_exit:
            self.exit_live_trade(option_id, live_data)
        
        # Select new trading opportunities
        undervalued_ids, overvalued_ids = self.select_live_trading_opportunities(live_data)
        
        # Enter new trades
        for option_id in undervalued_ids:
            self.enter_live_trade(option_id, 'BUY', live_data)
        
        for option_id in overvalued_ids:
            self.enter_live_trade(option_id, 'SELL', live_data)
        
        # Log current portfolio status
        active_trades_count = len(self.active_trades)
        logger.info(f"Live hour complete: {active_trades_count} active trades, Capital: ${self.current_capital:.2f}")
        
        # Save incremental results
        self.save_incremental_results()
        
        # Update last processed time
        self.last_processed_time = current_time
    
    def save_incremental_results(self):
        """Save incremental results after each hour"""
        
        # Save trade log
        if self.trade_history:
            df_trades = pd.DataFrame(self.trade_history)
            df_trades.to_excel(self.trade_log_file, index=False)
        
        # Save summary
        summary_data = {
            'Metric': [
                'Company', 'Initial Capital', 'Current Capital', 'Total PnL', 'Total Return %',
                'Total Trades', 'Profitable Trades', 'Win Rate %', 'Active Trades',
                'Last Updated'
            ],
            'Value': [
                self.company,
                f"${self.initial_capital:.2f}",
                f"${self.current_capital:.2f}",
                f"${self.total_pnl:.2f}",
                f"{((self.current_capital - self.initial_capital) / self.initial_capital * 100):.2f}%",
                self.total_trades,
                self.profitable_trades,
                f"{(self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0:.2f}%",
                len(self.active_trades),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(self.summary_file, index=False)
        
        logger.info(f"Incremental results saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def save_comprehensive_results(self):
        """Save comprehensive results with detailed analysis"""
        
        logger.info("=" * 80)
        logger.info("📊 SAVING COMPREHENSIVE RESULTS...")
        logger.info("=" * 80)
        
        # Save detailed trade log
        if self.trade_history:
            df_trades = pd.DataFrame(self.trade_history)
            
            # Add additional analysis columns
            df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
            df_trades['cumulative_return'] = (df_trades['cumulative_pnl'] / self.initial_capital) * 100
            df_trades['trade_number'] = range(1, len(df_trades) + 1)
            
            # Save detailed trade log
            detailed_trade_file = f"{self.output_dir}/{self.company}_detailed_trades.xlsx"
            df_trades.to_excel(detailed_trade_file, index=False)
            logger.info(f"Detailed trade log saved: {detailed_trade_file}")
            
            # Save active trades summary
            if self.active_trades:
                active_trades_data = []
                for option_id, trade in self.active_trades.items():
                    active_trades_data.append({
                        'option_id': option_id,
                        'trade_type': trade['trade_type'],
                        'entry_time': trade['entry_time'],
                        'entry_price': trade['entry_market_price'],
                        'strike_price': trade['strike_price'],
                        'contracts': trade['contracts'],
                        'position_size': trade['position_size'],
                        'duration_hours': (datetime.now() - trade['entry_time']).total_seconds() / 3600
                    })
                
                df_active = pd.DataFrame(active_trades_data)
                active_trades_file = f"{self.output_dir}/{self.company}_active_trades.xlsx"
                df_active.to_excel(active_trades_file, index=False)
                logger.info(f"Active trades saved: {active_trades_file}")
        
        # Save comprehensive summary
        summary_data = {
            'Metric': [
                'Company', 'Initial Capital', 'Final Capital', 'Total PnL', 'Total Return %',
                'Total Trades', 'Profitable Trades', 'Win Rate %', 'Active Trades',
                'Average Return per Trade %', 'Average Duration (hours)', 'Max Profit', 'Max Loss',
                'Best Trade', 'Worst Trade', 'Start Time', 'End Time', 'Total Runtime (hours)',
                'Last Updated'
            ],
            'Value': [
                self.company,
                f"${self.initial_capital:.2f}",
                f"${self.current_capital:.2f}",
                f"${self.total_pnl:.2f}",
                f"{((self.current_capital - self.initial_capital) / self.initial_capital * 100):.2f}%",
                self.total_trades,
                self.profitable_trades,
                f"{(self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0:.2f}%",
                len(self.active_trades),
                f"{self.trade_history[-1]['return_pct']:.2f}%" if self.trade_history else "0.00%",
                f"{self.trade_history[-1]['duration_hours']:.1f}" if self.trade_history else "0.0",
                f"${max([t['pnl'] for t in self.trade_history]):.2f}" if self.trade_history else "$0.00",
                f"${min([t['pnl'] for t in self.trade_history]):.2f}" if self.trade_history else "$0.00",
                f"{max(self.trade_history, key=lambda x: x['pnl'])['option_id']}" if self.trade_history else "N/A",
                f"{min(self.trade_history, key=lambda x: x['pnl'])['option_id']}" if self.trade_history else "N/A",
                self.last_processed_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_processed_time else "N/A",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f"{((datetime.now() - self.last_processed_time).total_seconds() / 3600):.1f}" if self.last_processed_time else "0.0",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        comprehensive_summary_file = f"{self.output_dir}/{self.company}_comprehensive_summary.xlsx"
        df_summary.to_excel(comprehensive_summary_file, index=False)
        logger.info(f"Comprehensive summary saved: {comprehensive_summary_file}")
        
        logger.info("=" * 80)
        logger.info("✅ COMPREHENSIVE RESULTS SAVED")
        logger.info("=" * 80)
    
    def generate_detailed_analysis(self):
        """Generate detailed analysis with plots and insights"""
        
        logger.info("=" * 80)
        logger.info("📈 GENERATING DETAILED ANALYSIS...")
        logger.info("=" * 80)
        
        if not self.trade_history:
            logger.warning("No trade history to analyze")
            return
        
        df_trades = pd.DataFrame(self.trade_history)
        
        # Create comprehensive analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.company} Real-Time Trading Analysis', fontsize=16)
        
        # Plot 1: Cumulative PnL over time
        cumulative_pnl = df_trades['pnl'].cumsum()
        ax1.plot(range(len(cumulative_pnl)), cumulative_pnl, 'b-', linewidth=2, marker='o')
        ax1.set_title('Cumulative PnL Over Time')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative PnL ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 2: Individual Trade PnL
        colors = ['green' if pnl > 0 else 'red' for pnl in df_trades['pnl']]
        bars = ax2.bar(range(len(df_trades)), df_trades['pnl'], color=colors, alpha=0.7)
        ax2.set_title('Individual Trade PnL')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('PnL ($)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 3: Trade Duration vs Return
        scatter = ax3.scatter(df_trades['duration_hours'], df_trades['return_pct'], 
                             c=df_trades['pnl'], cmap='RdYlGn', alpha=0.7, s=100)
        ax3.set_title('Trade Duration vs Return %')
        ax3.set_xlabel('Duration (hours)')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='PnL ($)')
        
        # Plot 4: Entry Mispricing vs PnL
        scatter2 = ax4.scatter(df_trades['entry_mispricing'], df_trades['pnl'], 
                              c=df_trades['return_pct'], cmap='RdYlGn', alpha=0.7, s=100)
        ax4.set_title('Entry Mispricing vs PnL')
        ax4.set_xlabel('Entry Mispricing ($)')
        ax4.set_ylabel('PnL ($)')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax4, label='Return (%)')
        
        plt.tight_layout()
        
        # Save analysis plot
        analysis_plot_file = f"{self.output_dir}/{self.company}_detailed_analysis.png"
        plt.savefig(analysis_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detailed analysis plot saved: {analysis_plot_file}")
        
        # Generate insights
        logger.info("=" * 80)
        logger.info("📊 TRADING INSIGHTS:")
        logger.info("=" * 80)
        
        # Calculate insights
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_return = df_trades['return_pct'].mean()
        avg_duration = df_trades['duration_hours'].mean()
        max_profit = df_trades['pnl'].max()
        max_loss = df_trades['pnl'].min()
        best_trade = df_trades.loc[df_trades['pnl'].idxmax()]
        worst_trade = df_trades.loc[df_trades['pnl'].idxmin()]
        
        logger.info(f"💰 Total Return: {total_return:.2f}%")
        logger.info(f"🎯 Win Rate: {win_rate:.2f}%")
        logger.info(f"📈 Average Return per Trade: {avg_return:.2f}%")
        logger.info(f"⏱️  Average Duration: {avg_duration:.1f} hours")
        logger.info(f"🚀 Best Trade: {best_trade['option_id']} (${max_profit:.2f})")
        logger.info(f"📉 Worst Trade: {worst_trade['option_id']} (${max_loss:.2f})")
        logger.info(f"📊 Total Trades: {self.total_trades}")
        logger.info(f"✅ Profitable Trades: {self.profitable_trades}")
        logger.info(f"💼 Active Positions: {len(self.active_trades)}")
        
        logger.info("=" * 80)
        logger.info("✅ DETAILED ANALYSIS COMPLETE")
        logger.info("=" * 80)
    
    def run_realtime_trading(self):
        """
        Run the real-time trading system continuously
        """
        logger.info(f"Starting real-time trading system for {self.company}")
        
        # Connect to Bloomberg
        if not self.connect_bloomberg():
            logger.error("Failed to connect to Bloomberg, cannot start real-time trading")
            return
        
        self.is_running = True
        logger.info("Real-time trading system is now running. Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Process the current hour
                self.process_live_hour()
                
                # Wait for the next hour (3600 seconds = 1 hour)
                logger.info(f"Waiting for next hour... (will process at {(current_time + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')})")
                time.sleep(3600)  # Wait 1 hour
                
        except KeyboardInterrupt:
            logger.info("=" * 80)
            logger.info("🛑 REAL-TIME TRADING STOPPED BY USER (Ctrl+C)")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Error in real-time trading: {e}")
        finally:
            # Close any remaining positions
            logger.info("=" * 80)
            logger.info("🔚 CLOSING REMAINING POSITIONS...")
            logger.info("=" * 80)
            
            try:
                live_data = self.fetch_live_data()
                if not live_data.empty:
                    for option_id in list(self.active_trades.keys()):
                        self.exit_live_trade(option_id, live_data)
                else:
                    logger.warning("Could not fetch live data for position closure")
            except Exception as e:
                logger.error(f"Error closing positions: {e}")
            
            # Close Bloomberg session
            if self.session:
                self.session.stop()
                logger.info("Bloomberg session closed")
            
            # Save comprehensive final results
            self.save_comprehensive_results()
            
            # Generate detailed final analysis
            self.generate_detailed_analysis()
            
            logger.info("=" * 80)
            logger.info("✅ REAL-TIME TRADING SYSTEM STOPPED")
            logger.info("=" * 80)
    
    def generate_final_summary(self):
        """Generate final summary report"""
        
        if not self.trade_history:
            logger.warning("No trade history to summarize")
            return
        
        df_trades = pd.DataFrame(self.trade_history)
        
        # Calculate final statistics
        win_rate = (self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        avg_return = df_trades['return_pct'].mean() if len(df_trades) > 0 else 0
        avg_duration = df_trades['duration_hours'].mean() if len(df_trades) > 0 else 0
        max_profit = df_trades['pnl'].max() if len(df_trades) > 0 else 0
        max_loss = df_trades['pnl'].min() if len(df_trades) > 0 else 0
        
        logger.info("=== FINAL REAL-TIME TRADING SUMMARY ===")
        logger.info(f"Company: {self.company}")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Final Capital: ${self.current_capital:.2f}")
        logger.info(f"Total PnL: ${self.total_pnl:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Profitable Trades: {self.profitable_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Average Return per Trade: {avg_return:.2f}%")
        logger.info(f"Average Duration: {avg_duration:.1f} hours")
        logger.info(f"Max Profit: ${max_profit:.2f}")
        logger.info(f"Max Loss: ${max_loss:.2f}")
        logger.info("========================================")


def main():
    """Main function to run the real-time NVDA trading system"""
    
    # Initialize real-time trading system
    trading_system = NVDARealTimeTrading(initial_capital=100.0)
    
    # Start real-time trading
    trading_system.run_realtime_trading()


if __name__ == "__main__":
    main() 