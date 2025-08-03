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
from heston_calculator import calibrate_heston, heston_price_row

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyOfflineTrading:
    """Individual company offline trading system using existing Excel data"""
    
    def __init__(self, company: str, initial_capital: float = 100.0):
        """
        Initialize offline trading system for a single company
        
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
        
        logger.info(f"Offline trading system initialized for {self.company} with ${initial_capital} capital")
    
    def load_existing_data(self) -> pd.DataFrame:
        """
        Load existing options data from Excel file
        
        Returns:
            DataFrame with options data including Heston prices
        """
        try:
            # Try to load the Heston data file
            heston_file = f"scripts/excels/{self.company.lower()}_options_2025-08-15_heston.xlsx"
            
            if not os.path.exists(heston_file):
                logger.error(f"Data file not found: {heston_file}")
                return pd.DataFrame()
            
            logger.info(f"Loading existing data from {heston_file}")
            df = pd.read_excel(heston_file)
            
            if df.empty:
                logger.warning(f"No data found in {heston_file}")
                return df
            
            # Add current timestamp
            current_time = datetime.now()
            df['Timestamp'] = current_time
            df['Hour'] = current_time.hour
            
            # Add unique option identifier
            df['Option_ID'] = df.apply(
                lambda row: f"{self.company}_{row['Strike']}_{row['Option Type']}", 
                axis=1
            )
            
            logger.info(f"Loaded {len(df)} options for {self.company}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading existing data for {self.company}: {e}")
            return pd.DataFrame()
    
    def select_trading_opportunities(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Select the 2 most undervalued and 2 most overvalued call options
        
        Args:
            data: Current options data
            
        Returns:
            Tuple of (undervalued_option_ids, overvalued_option_ids)
        """
        # Filter for call options only
        call_data = data[data['Option Type'] == 'Call'].copy()
        
        # Filter for valid data (non-null prices)
        valid_data = call_data.dropna(subset=['PX_LAST', 'Heston_Price'])
        
        if len(valid_data) < 4:
            logger.warning(f"Insufficient data for {self.company}: only {len(valid_data)} valid call options")
            return [], []
        
        # Calculate mispricing
        valid_data['Market_vs_Heston'] = valid_data['PX_LAST'] - valid_data['Heston_Price']
        valid_data['Heston_vs_Market'] = valid_data['Heston_Price'] - valid_data['PX_LAST']
        
        # Find undervalued options (Market < Heston)
        undervalued = valid_data[valid_data['Market_vs_Heston'] < 0].copy()
        undervalued = undervalued.sort_values('Heston_vs_Market', ascending=False)
        
        # Find overvalued options (Market > Heston)
        overvalued = valid_data[valid_data['Market_vs_Heston'] > 0].copy()
        overvalued = overvalued.sort_values('Market_vs_Heston', ascending=False)
        
        # Select top 2 from each category
        undervalued_ids = undervalued['Option_ID'].head(2).tolist()
        overvalued_ids = overvalued['Option_ID'].head(2).tolist()
        
        logger.info(f"Selection for {self.company}: {len(undervalued_ids)} undervalued, {len(overvalued_ids)} overvalued options")
        
        return undervalued_ids, overvalued_ids
    
    def enter_trade(self, option_id: str, trade_type: str, data: pd.DataFrame) -> bool:
        """
        Enter a new trade if not already in position
        
        Args:
            option_id: Unique option identifier
            trade_type: 'BUY' for undervalued, 'SELL' for overvalued
            data: Current data
            
        Returns:
            True if trade entered, False otherwise
        """
        if option_id in self.active_trades:
            logger.debug(f"Already in trade for {option_id}")
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
            'entry_mispricing': option_data['PX_LAST'] - option_data['Heston_Price']
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
        logger.info(f"   Mispricing: ${trade_info['entry_mispricing']:.2f}")
        logger.info(f"   Active Trades: {len(self.active_trades)}")
        logger.info(f"   Current Capital: ${self.current_capital:.2f}")
        logger.info("=" * 80)
        
        return True
    
    def check_exit_conditions(self, option_id: str, data: pd.DataFrame) -> bool:
        """
        Check if exit conditions are met for a trade
        
        Args:
            option_id: Option identifier
            data: Current data
            
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
    
    def exit_trade(self, option_id: str, data: pd.DataFrame) -> bool:
        """
        Exit a trade and calculate PnL
        
        Args:
            option_id: Option identifier
            data: Current data
            
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
    
    def process_hour(self):
        """
        Process one trading hour for this company
        """
        current_time = datetime.now()
        logger.info(f"Processing hour for {self.company} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load existing data
        data = self.load_existing_data()
        if data.empty:
            logger.error(f"No data available for {self.company}, skipping this hour")
            return
        
        # Check exit conditions for existing trades
        options_to_exit = []
        for option_id in list(self.active_trades.keys()):
            if self.check_exit_conditions(option_id, data):
                options_to_exit.append(option_id)
        
        # Exit trades
        for option_id in options_to_exit:
            self.exit_trade(option_id, data)
        
        # Select new trading opportunities
        undervalued_ids, overvalued_ids = self.select_trading_opportunities(data)
        
        # Enter new trades
        for option_id in undervalued_ids:
            self.enter_trade(option_id, 'BUY', data)
        
        for option_id in overvalued_ids:
            self.enter_trade(option_id, 'SELL', data)
        
        # Log current portfolio status
        active_trades_count = len(self.active_trades)
        logger.info(f"Hour complete for {self.company}: {active_trades_count} active trades, Capital: ${self.current_capital:.2f}")
        
        # Update last processed time
        self.last_processed_time = current_time


class MultiCompanyOfflineTrading:
    """Multi-company offline trading system using existing Excel data"""
    
    def __init__(self, companies: List[str], capital_per_company: float = 100.0):
        """
        Initialize multi-company offline trading system
        
        Args:
            companies: List of company ticker symbols
            capital_per_company: Capital allocation per company
        """
        self.companies = companies
        self.capital_per_company = capital_per_company
        self.is_running = False
        
        # Initialize individual company trading systems
        self.company_systems = {}
        for company in companies:
            self.company_systems[company] = CompanyOfflineTrading(company, capital_per_company)
        
        # Create output directories
        self.output_dir = "scripts/realtime_output/offline_multi_company"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Portfolio tracking
        self.total_portfolio_capital = capital_per_company * len(companies)
        self.current_portfolio_capital = self.total_portfolio_capital
        self.portfolio_trade_history = []
        
        logger.info(f"Multi-company offline trading system initialized for {len(companies)} companies")
        logger.info(f"Companies: {', '.join(companies)}")
        logger.info(f"Capital per company: ${capital_per_company}")
        logger.info(f"Total portfolio capital: ${self.total_portfolio_capital}")
    
    def process_hour_all_companies(self):
        """Process one trading hour for all companies"""
        current_time = datetime.now()
        logger.info("=" * 80)
        logger.info(f"🔄 PROCESSING HOUR FOR ALL COMPANIES at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Process each company
        for company in self.companies:
            try:
                self.company_systems[company].process_hour()
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
                trade_file = f"{self.output_dir}/{company}_offline_trades.xlsx"
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
        portfolio_file = f"{self.output_dir}/offline_portfolio_summary.xlsx"
        df_portfolio.to_excel(portfolio_file, index=False)
        
        logger.info(f"Offline incremental portfolio results saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_offline_trading(self, num_hours: int = 24):
        """
        Run the multi-company offline trading system for specified hours
        
        Args:
            num_hours: Number of hours to simulate
        """
        logger.info(f"Starting multi-company offline trading system for {len(self.companies)} companies")
        logger.info(f"Will simulate {num_hours} hours of trading")
        
        self.is_running = True
        logger.info("Multi-company offline trading system is now running. Press Ctrl+C to stop.")
        
        try:
            for hour in range(num_hours):
                current_time = datetime.now()
                logger.info(f"Processing hour {hour + 1}/{num_hours}")
                
                # Process the current hour for all companies
                self.process_hour_all_companies()
                
                # Wait for the next hour (simulate real-time)
                if hour < num_hours - 1:  # Don't wait after the last hour
                    logger.info(f"Waiting for next hour... (will process at {(current_time + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')})")
                    time.sleep(3600)  # Wait 1 hour
                
        except KeyboardInterrupt:
            logger.info("=" * 80)
            logger.info("🛑 MULTI-COMPANY OFFLINE TRADING STOPPED BY USER (Ctrl+C)")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Error in multi-company offline trading: {e}")
        finally:
            logger.info("=" * 80)
            logger.info("✅ MULTI-COMPANY OFFLINE TRADING SYSTEM STOPPED")
            logger.info("=" * 80)


def main():
    """Main function to run the multi-company offline trading system"""
    
    # Define the 7 companies
    companies = ['AVGO', 'AMD', 'TSLA', 'BBAI', 'CRCL', 'NVDA', 'AUR']
    
    # Initialize multi-company offline trading system
    trading_system = MultiCompanyOfflineTrading(companies, capital_per_company=100.0)
    
    # Start multi-company offline trading (simulate 24 hours)
    trading_system.run_offline_trading(num_hours=24)


if __name__ == "__main__":
    main() 