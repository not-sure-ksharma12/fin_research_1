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

class CompanyRealTimeTrading:
    """Individual company real-time trading system"""
    
    def __init__(self, company: str, initial_capital: float = 100.0):
        """
        Initialize real-time trading system for a single company
        
        Args:
            company: Company ticker symbol
            initial_capital: Initial capital allocation
        """
        self.company = company
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.active_trades = {}  # {option_id: trade_info}
        self.trade_history = []
        self.last_processed_time = None
        
        # Trade tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        
        logger.info(f"Real-time trading system initialized for {self.company} with ${initial_capital} capital")
    
    def fetch_live_data(self, session, expiry_date: str = "2025-12-19") -> pd.DataFrame:
        """
        Fetch live options data from Bloomberg for this company
        
        Args:
            session: Bloomberg session
            expiry_date: Options expiration date
            
        Returns:
            DataFrame with current options data including Heston prices
        """
        try:
            logger.info(f"Fetching live {self.company} options data for expiry {expiry_date}")
            
            # Get current option chain
            chain = get_option_chain(session, self.company)
            
            # Select options for the given expiry
            all_options = []
            for option in chain:
                parsed = parse_option_ticker(option)
                if parsed and parsed["Expiration"] == expiry_date:
                    all_options.append((option, parsed["Strike"], parsed["Option Type"]))
            
            logger.info(f"Found {len(all_options)} live options for {self.company}")
            
            # Get current stock price
            underlying_ticker = f"{self.company} US Equity"
            current_price = get_current_price(session, underlying_ticker)
            
            if not current_price:
                logger.warning(f"Could not get current price for {self.company}")
                return pd.DataFrame()
            
            # Fetch live option data
            option_data = fetch_option_data(session, all_options, current_price=current_price)
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
        logger.info(f"   Company: {self.company}")
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
        logger.info(f"   Company: {self.company}")
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
    
    def process_live_hour(self, session):
        """
        Process one live trading hour for this company
        """
        current_time = datetime.now()
        logger.info(f"Processing live hour for {self.company} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Fetch fresh live data
        live_data = self.fetch_live_data(session)
        if live_data.empty:
            logger.error(f"No live data available for {self.company}, skipping this hour")
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
        logger.info(f"Live hour complete for {self.company}: {active_trades_count} active trades, Capital: ${self.current_capital:.2f}")
        
        # Update last processed time
        self.last_processed_time = current_time


class MultiCompanyRealTimeTrading:
    """Multi-company real-time trading system"""
    
    def __init__(self, companies: List[str], capital_per_company: float = 100.0):
        """
        Initialize multi-company real-time trading system
        
        Args:
            companies: List of company ticker symbols
            capital_per_company: Capital allocation per company
        """
        self.companies = companies
        self.capital_per_company = capital_per_company
        self.session = None
        self.is_running = False
        
        # Initialize individual company trading systems
        self.company_systems = {}
        for company in companies:
            self.company_systems[company] = CompanyRealTimeTrading(company, capital_per_company)
        
        # Create output directories
        self.output_dir = "scripts/realtime_output/multi_company_aug15"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Portfolio tracking
        self.total_portfolio_capital = capital_per_company * len(companies)
        self.current_portfolio_capital = self.total_portfolio_capital
        self.portfolio_trade_history = []
        
        logger.info(f"Multi-company real-time trading system initialized for {len(companies)} companies")
        logger.info(f"Companies: {', '.join(companies)}")
        logger.info(f"Capital per company: ${capital_per_company}")
        logger.info(f"Total portfolio capital: ${self.total_portfolio_capital}")
    
    def connect_bloomberg(self):
        """Connect to Bloomberg Terminal"""
        try:
            self.session = connect_to_bloomberg()
            logger.info("Successfully connected to Bloomberg Terminal")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Bloomberg: {e}")
            return False
    
    def process_live_hour_all_companies(self):
        """Process one live trading hour for all companies"""
        current_time = datetime.now()
        logger.info("=" * 80)
        logger.info(f"🔄 PROCESSING LIVE HOUR FOR ALL COMPANIES at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Process each company
        for company in self.companies:
            try:
                self.company_systems[company].process_live_hour(self.session)
            except Exception as e:
                logger.error(f"Error processing {company}: {e}")
        
        # Update portfolio capital
        self.current_portfolio_capital = sum(
            system.current_capital for system in self.company_systems.values()
        )
        
        # Log portfolio status
        total_active_trades = sum(
            len(system.active_trades) for system in self.company_systems.values()
        )
        total_trades = sum(
            system.total_trades for system in self.company_systems.values()
        )
        total_pnl = sum(
            system.total_pnl for system in self.company_systems.values()
        )
        
        logger.info("=" * 80)
        logger.info(f"📊 PORTFOLIO STATUS:")
        logger.info(f"   Total Active Trades: {total_active_trades}")
        logger.info(f"   Total Trades Executed: {total_trades}")
        logger.info(f"   Total Portfolio PnL: ${total_pnl:.2f}")
        logger.info(f"   Current Portfolio Capital: ${self.current_portfolio_capital:.2f}")
        logger.info(f"   Portfolio Return: {((self.current_portfolio_capital - self.total_portfolio_capital) / self.total_portfolio_capital * 100):.2f}%")
        logger.info("=" * 80)
        
        # Save incremental results
        self.save_incremental_results()
    
    def save_incremental_results(self):
        """Save incremental results after each hour"""
        
        # Save individual company results
        for company, system in self.company_systems.items():
            if system.trade_history:
                df_trades = pd.DataFrame(system.trade_history)
                trade_file = f"{self.output_dir}/{company}_realtime_trades.xlsx"
                df_trades.to_excel(trade_file, index=False)
        
        # Save portfolio summary
        portfolio_summary = {
            'Company': [],
            'Initial Capital': [],
            'Current Capital': [],
            'Total PnL': [],
            'Total Return %': [],
            'Total Trades': [],
            'Profitable Trades': [],
            'Win Rate %': [],
            'Active Trades': [],
            'Last Updated': []
        }
        
        for company, system in self.company_systems.items():
            win_rate = (system.profitable_trades / system.total_trades * 100) if system.total_trades > 0 else 0
            total_return = ((system.current_capital - system.initial_capital) / system.initial_capital * 100)
            
            portfolio_summary['Company'].append(company)
            portfolio_summary['Initial Capital'].append(f"${system.initial_capital:.2f}")
            portfolio_summary['Current Capital'].append(f"${system.current_capital:.2f}")
            portfolio_summary['Total PnL'].append(f"${system.total_pnl:.2f}")
            portfolio_summary['Total Return %'].append(f"{total_return:.2f}%")
            portfolio_summary['Total Trades'].append(system.total_trades)
            portfolio_summary['Profitable Trades'].append(system.profitable_trades)
            portfolio_summary['Win Rate %'].append(f"{win_rate:.2f}%")
            portfolio_summary['Active Trades'].append(len(system.active_trades))
            portfolio_summary['Last Updated'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        df_portfolio = pd.DataFrame(portfolio_summary)
        portfolio_file = f"{self.output_dir}/portfolio_summary.xlsx"
        df_portfolio.to_excel(portfolio_file, index=False)
        
        logger.info(f"Incremental portfolio results saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def save_comprehensive_results(self):
        """Save comprehensive results with detailed analysis"""
        
        logger.info("=" * 80)
        logger.info("📊 SAVING COMPREHENSIVE PORTFOLIO RESULTS...")
        logger.info("=" * 80)
        
        # Save detailed trade logs for each company
        for company, system in self.company_systems.items():
            if system.trade_history:
                df_trades = pd.DataFrame(system.trade_history)
                
                # Add additional analysis columns
                df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
                df_trades['cumulative_return'] = (df_trades['cumulative_pnl'] / system.initial_capital) * 100
                df_trades['trade_number'] = range(1, len(df_trades) + 1)
                
                # Save detailed trade log
                detailed_trade_file = f"{self.output_dir}/{company}_detailed_trades.xlsx"
                df_trades.to_excel(detailed_trade_file, index=False)
                logger.info(f"Detailed trade log saved for {company}: {detailed_trade_file}")
                
                # Save active trades summary
                if system.active_trades:
                    active_trades_data = []
                    for option_id, trade in system.active_trades.items():
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
                    active_trades_file = f"{self.output_dir}/{company}_active_trades.xlsx"
                    df_active.to_excel(active_trades_file, index=False)
                    logger.info(f"Active trades saved for {company}: {active_trades_file}")
        
        # Save comprehensive portfolio summary
        portfolio_summary_data = {
            'Metric': [
                'Total Companies', 'Initial Portfolio Capital', 'Final Portfolio Capital', 'Total Portfolio PnL', 'Total Portfolio Return %',
                'Total Trades', 'Total Profitable Trades', 'Overall Win Rate %', 'Total Active Trades',
                'Best Performing Company', 'Worst Performing Company', 'Start Time', 'End Time', 'Total Runtime (hours)',
                'Last Updated'
            ],
            'Value': [
                len(self.companies),
                f"${self.total_portfolio_capital:.2f}",
                f"${self.current_portfolio_capital:.2f}",
                f"${self.current_portfolio_capital - self.total_portfolio_capital:.2f}",
                f"{((self.current_portfolio_capital - self.total_portfolio_capital) / self.total_portfolio_capital * 100):.2f}%",
                sum(system.total_trades for system in self.company_systems.values()),
                sum(system.profitable_trades for system in self.company_systems.values()),
                f"{(sum(system.profitable_trades for system in self.company_systems.values()) / sum(system.total_trades for system in self.company_systems.values()) * 100) if sum(system.total_trades for system in self.company_systems.values()) > 0 else 0:.2f}%",
                sum(len(system.active_trades) for system in self.company_systems.values()),
                max(self.company_systems.items(), key=lambda x: x[1].total_pnl)[0] if any(system.total_pnl > 0 for system in self.company_systems.values()) else "N/A",
                min(self.company_systems.items(), key=lambda x: x[1].total_pnl)[0] if any(system.total_pnl < 0 for system in self.company_systems.values()) else "N/A",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "0.0",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        df_portfolio_summary = pd.DataFrame(portfolio_summary_data)
        comprehensive_portfolio_file = f"{self.output_dir}/comprehensive_portfolio_summary.xlsx"
        df_portfolio_summary.to_excel(comprehensive_portfolio_file, index=False)
        logger.info(f"Comprehensive portfolio summary saved: {comprehensive_portfolio_file}")
        
        logger.info("=" * 80)
        logger.info("✅ COMPREHENSIVE PORTFOLIO RESULTS SAVED")
        logger.info("=" * 80)
    
    def generate_detailed_analysis(self):
        """Generate detailed analysis with plots and insights"""
        
        logger.info("=" * 80)
        logger.info("📈 GENERATING DETAILED PORTFOLIO ANALYSIS...")
        logger.info("=" * 80)
        
        # Create comprehensive analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Company Real-Time Trading Analysis', fontsize=16)
        
        # Plot 1: Company Performance Comparison
        companies = list(self.company_systems.keys())
        returns = []
        pnls = []
        
        for company in companies:
            system = self.company_systems[company]
            total_return = ((system.current_capital - system.initial_capital) / system.initial_capital * 100)
            returns.append(total_return)
            pnls.append(system.total_pnl)
        
        colors = ['green' if ret > 0 else 'red' for ret in returns]
        bars1 = ax1.bar(companies, returns, color=colors, alpha=0.7)
        ax1.set_title('Company Performance Comparison')
        ax1.set_xlabel('Company')
        ax1.set_ylabel('Total Return (%)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 2: Total PnL by Company
        bars2 = ax2.bar(companies, pnls, color=colors, alpha=0.7)
        ax2.set_title('Total PnL by Company')
        ax2.set_xlabel('Company')
        ax2.set_ylabel('Total PnL ($)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 3: Trade Count by Company
        trade_counts = [system.total_trades for system in self.company_systems.values()]
        bars3 = ax3.bar(companies, trade_counts, alpha=0.7)
        ax3.set_title('Total Trades by Company')
        ax3.set_xlabel('Company')
        ax3.set_ylabel('Number of Trades')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Win Rate by Company
        win_rates = []
        for system in self.company_systems.values():
            win_rate = (system.profitable_trades / system.total_trades * 100) if system.total_trades > 0 else 0
            win_rates.append(win_rate)
        
        bars4 = ax4.bar(companies, win_rates, alpha=0.7)
        ax4.set_title('Win Rate by Company')
        ax4.set_xlabel('Company')
        ax4.set_ylabel('Win Rate (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save analysis plot
        analysis_plot_file = f"{self.output_dir}/portfolio_detailed_analysis.png"
        plt.savefig(analysis_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detailed portfolio analysis plot saved: {analysis_plot_file}")
        
        # Generate insights
        logger.info("=" * 80)
        logger.info("📊 PORTFOLIO TRADING INSIGHTS:")
        logger.info("=" * 80)
        
        # Calculate portfolio insights
        total_portfolio_return = ((self.current_portfolio_capital - self.total_portfolio_capital) / self.total_portfolio_capital) * 100
        total_trades = sum(system.total_trades for system in self.company_systems.values())
        total_profitable_trades = sum(system.profitable_trades for system in self.company_systems.values())
        overall_win_rate = (total_profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        best_company = max(self.company_systems.items(), key=lambda x: x[1].total_pnl)[0]
        worst_company = min(self.company_systems.items(), key=lambda x: x[1].total_pnl)[0]
        
        logger.info(f"💰 Total Portfolio Return: {total_portfolio_return:.2f}%")
        logger.info(f"🎯 Overall Win Rate: {overall_win_rate:.2f}%")
        logger.info(f"📊 Total Trades: {total_trades}")
        logger.info(f"✅ Total Profitable Trades: {total_profitable_trades}")
        logger.info(f"🚀 Best Performing Company: {best_company}")
        logger.info(f"📉 Worst Performing Company: {worst_company}")
        logger.info(f"💼 Total Active Positions: {sum(len(system.active_trades) for system in self.company_systems.values())}")
        
        logger.info("=" * 80)
        logger.info("✅ DETAILED PORTFOLIO ANALYSIS COMPLETE")
        logger.info("=" * 80)
    
    def run_realtime_trading(self):
        """
        Run the multi-company real-time trading system continuously
        """
        logger.info(f"Starting multi-company real-time trading system for {len(self.companies)} companies")
        
        # Connect to Bloomberg
        if not self.connect_bloomberg():
            logger.error("Failed to connect to Bloomberg, cannot start real-time trading")
            return
        
        self.is_running = True
        logger.info("Multi-company real-time trading system is now running. Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Process the current hour for all companies
                self.process_live_hour_all_companies()
                
                # Wait for the next hour (3600 seconds = 1 hour)
                logger.info(f"Waiting for next hour... (will process at {(current_time + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')})")
                time.sleep(3600)  # Wait 1 hour
                
        except KeyboardInterrupt:
            logger.info("=" * 80)
            logger.info("🛑 MULTI-COMPANY REAL-TIME TRADING STOPPED BY USER (Ctrl+C)")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Error in multi-company real-time trading: {e}")
        finally:
            # Close any remaining positions for all companies
            logger.info("=" * 80)
            logger.info("🔚 CLOSING REMAINING POSITIONS FOR ALL COMPANIES...")
            logger.info("=" * 80)
            
            try:
                for company, system in self.company_systems.items():
                    logger.info(f"Closing positions for {company}...")
                    live_data = system.fetch_live_data(self.session)
                    if not live_data.empty:
                        for option_id in list(system.active_trades.keys()):
                            system.exit_live_trade(option_id, live_data)
                    else:
                        logger.warning(f"Could not fetch live data for {company} position closure")
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
            logger.info("✅ MULTI-COMPANY REAL-TIME TRADING SYSTEM STOPPED")
            logger.info("=" * 80)


def main():
    """Main function to run the multi-company real-time trading system"""
    
    # Define the 7 companies
    companies = ['AVGO', 'AMD', 'TSLA', 'BBAI', 'CRCL', 'NVDA', 'AUR']
    
    # Initialize multi-company real-time trading system
    trading_system = MultiCompanyRealTimeTrading(companies, capital_per_company=100.0)
    
    # Start multi-company real-time trading
    trading_system.run_realtime_trading()


if __name__ == "__main__":
    main() 