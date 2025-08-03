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
import blpapi
from blpapi.event import Event

# Add scripts directory to path
sys.path.append(r"C:\Users\ksharma12\fin_research\scripts")

# Import the existing modules
from fetch_options_to_excel import (
    connect_to_bloomberg, get_option_chain, parse_option_ticker, 
    select_top_25_each, get_current_price, fetch_option_data, save_with_formatting
)
from heston_calculator import calibrate_heston, heston_price_row

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteHestonStrategy:
    def __init__(self, companies: List[str], capital_per_company: float = 100.0):
        """
        Initialize the complete Heston mispricing trading strategy
        
        Args:
            companies: List of company tickers
            capital_per_company: Capital allocation per company
        """
        self.companies = companies
        self.capital_per_company = capital_per_company
        self.session = None
        self.strategies = {}
        self.results = {}
        
        # Create output directories
        self.output_dir = "scripts/strategy_output/complete_strategy"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize strategies for each company
        for company in companies:
            self.strategies[company] = CompanyStrategy(
                company=company, 
                initial_capital=capital_per_company
            )
    
    def connect_bloomberg(self):
        """Connect to Bloomberg Terminal"""
        try:
            self.session = connect_to_bloomberg()
            logger.info("Successfully connected to Bloomberg Terminal")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Bloomberg: {e}")
            return False
    
    def fetch_company_data(self, company: str, expiry_date: str = "2025-12-19") -> pd.DataFrame:
        """
        Fetch and process options data for a company
        
        Args:
            company: Company ticker
            expiry_date: Options expiration date
            
        Returns:
            DataFrame with options data including Heston prices
        """
        try:
            logger.info(f"Fetching options data for {company}")
            
            # Get option chain
            chain = get_option_chain(self.session, company)
            
            # Select options for the given expiry
            all_options = []
            for option in chain:
                parsed = parse_option_ticker(option)
                if parsed and parsed["Expiration"] == expiry_date:
                    all_options.append((option, parsed["Strike"], parsed["Option Type"]))
            
            logger.info(f"Found {len(all_options)} options for {company}")
            
            # Get current stock price
            underlying_ticker = f"{company} US Equity"
            current_price = get_current_price(self.session, underlying_ticker)
            
            if not current_price:
                logger.warning(f"Could not get current price for {company}")
                return pd.DataFrame()
            
            # Fetch option data
            option_data = fetch_option_data(self.session, all_options, current_price=current_price)
            df = pd.DataFrame(option_data)
            
            if df.empty:
                logger.warning(f"No option data retrieved for {company}")
                return df
            
            # Ensure current price is filled
            df["Current Price"] = current_price
            
            # Sort by strike price
            df_calls = df[df["Option Type"] == "Call"].sort_values(by=["Strike"], ascending=True)
            df_puts = df[df["Option Type"] == "Put"].sort_values(by=["Strike"], ascending=True)
            df_sorted = pd.concat([df_calls, df_puts], ignore_index=True)
            
            # Calculate Heston prices
            df_with_heston = self.calculate_heston_prices(df_sorted, company)
            
            return df_with_heston
            
        except Exception as e:
            logger.error(f"Error fetching data for {company}: {e}")
            return pd.DataFrame()
    
    def calculate_heston_prices(self, df: pd.DataFrame, company: str) -> pd.DataFrame:
        """
        Calculate Heston model prices for options
        
        Args:
            df: DataFrame with options data
            company: Company ticker
            
        Returns:
            DataFrame with Heston prices added
        """
        try:
            if df.empty:
                return df
            
            # Prepare data for Heston calculation
            valuation_date = datetime.today()
            risk_free_rate = 0.0453  # 4.53% from Bloomberg
            
            # Ensure required columns exist
            if "Implied Volatility" not in df.columns or df["Implied Volatility"].isna().all():
                logger.warning(f"No implied volatility data for {company}, using default")
                df["Implied Volatility"] = 0.3 * 100  # 30% default
            
            # Calibrate Heston model
            params, heston_model, helpers, model_prices, market_prices = calibrate_heston(
                df, valuation_date, risk_free_rate=risk_free_rate
            )
            
            logger.info(f"Calibrated Heston params for {company}: v0={params[0]:.4f}, kappa={params[1]:.4f}, theta={params[2]:.4f}, sigma={params[3]:.4f}, rho={params[4]:.4f}")
            
            # Calculate Heston prices for all options
            heston_results = df.apply(
                lambda row: heston_price_row(row, heston_model, valuation_date, risk_free_rate), 
                axis=1
            )
            
            # Add Heston prices to DataFrame
            df_with_heston = pd.concat([df, heston_results], axis=1)
            
            # Calculate mispricing metrics
            df_with_heston['Market_vs_Heston'] = df_with_heston['PX_LAST'] - df_with_heston['Heston_Price']
            df_with_heston['Heston_vs_Market'] = df_with_heston['Heston_Price'] - df_with_heston['PX_LAST']
            
            # Add unique option identifier
            df_with_heston['Option_ID'] = df_with_heston.apply(
                lambda row: f"{company}_{row['Strike']}_{row['Option Type']}", 
                axis=1
            )
            
            return df_with_heston
            
        except Exception as e:
            logger.error(f"Error calculating Heston prices for {company}: {e}")
            return df
    
    def simulate_hourly_data(self, base_data: pd.DataFrame, num_hours: int = 24) -> List[pd.DataFrame]:
        """
        Simulate hourly data by adding random price movements
        
        Args:
            base_data: Base options data
            num_hours: Number of hours to simulate
            
        Returns:
            List of hourly DataFrames
        """
        hourly_data = []
        
        for hour in range(num_hours):
            # Create a copy of base data for this hour
            hour_data = base_data.copy()
            
            # Simulate market price movements (random walk)
            price_change_factor = np.random.normal(1.0, 0.02)  # 2% standard deviation
            
            # Update option prices
            hour_data['PX_LAST'] = hour_data['PX_LAST'] * price_change_factor
            hour_data['Current Price'] = hour_data['Current Price'] * price_change_factor
            
            # Recalculate mispricing metrics
            hour_data['Market_vs_Heston'] = hour_data['PX_LAST'] - hour_data['Heston_Price']
            hour_data['Heston_vs_Market'] = hour_data['Heston_Price'] - hour_data['PX_LAST']
            
            # Add timestamp
            hour_data['Timestamp'] = datetime.now() + timedelta(hours=hour)
            hour_data['Hour'] = hour
            
            hourly_data.append(hour_data)
            
        return hourly_data
    
    def run_simulation_for_company(self, company: str, num_hours: int = 24) -> Dict:
        """Run complete simulation for a single company"""
        logger.info(f"Running complete simulation for {company}")
        
        # Fetch fresh data from Bloomberg
        df = self.fetch_company_data(company)
        
        if df.empty:
            logger.warning(f"No data available for {company}")
            return {}
        
        # Run strategy simulation
        strategy = self.strategies[company]
        results = strategy.run_simulation_with_data(df, num_hours)
        
        return results
    
    def run_multi_company_simulation(self, num_hours: int = 24) -> Dict:
        """Run simulation for all companies"""
        logger.info(f"Starting multi-company simulation for {len(self.companies)} companies")
        
        # Connect to Bloomberg
        if not self.connect_bloomberg():
            logger.error("Failed to connect to Bloomberg, simulation cannot proceed")
            return {}
        
        all_results = {}
        
        for company in self.companies:
            try:
                results = self.run_simulation_for_company(company, num_hours)
                if results:
                    all_results[company] = results
                    logger.info(f"Completed simulation for {company}")
                else:
                    logger.warning(f"Failed to get results for {company}")
            except Exception as e:
                logger.error(f"Error running simulation for {company}: {e}")
        
        # Close Bloomberg session
        if self.session:
            self.session.stop()
        
        # Generate portfolio-level analysis
        portfolio_results = self.analyze_portfolio(all_results)
        
        # Save combined results
        self.save_portfolio_results(portfolio_results, all_results)
        
        return portfolio_results
    
    def analyze_portfolio(self, company_results: Dict) -> Dict:
        """Analyze portfolio-level performance across all companies"""
        
        if not company_results:
            return {}
        
        # Calculate portfolio statistics
        total_initial_capital = len(company_results) * self.capital_per_company
        total_final_capital = sum(results['final_capital'] for results in company_results.values())
        total_pnl = sum(results['total_pnl'] for results in company_results.values())
        total_trades = sum(results['total_trades'] for results in company_results.values())
        total_profitable_trades = sum(results['profitable_trades'] for results in company_results.values())
        
        # Calculate averages
        avg_return = np.mean([results['total_return_pct'] for results in company_results.values()])
        avg_win_rate = np.mean([results['win_rate'] for results in company_results.values()])
        avg_trades_per_company = total_trades / len(company_results)
        
        # Find best and worst performers
        returns_by_company = {company: results['total_return_pct'] for company, results in company_results.items()}
        best_company = max(returns_by_company, key=returns_by_company.get)
        worst_company = min(returns_by_company, key=returns_by_company.get)
        
        portfolio_results = {
            'total_companies': len(company_results),
            'total_initial_capital': total_initial_capital,
            'total_final_capital': total_final_capital,
            'total_pnl': total_pnl,
            'total_return_pct': ((total_final_capital - total_initial_capital) / total_initial_capital) * 100,
            'total_trades': total_trades,
            'total_profitable_trades': total_profitable_trades,
            'portfolio_win_rate': (total_profitable_trades / total_trades * 100) if total_trades > 0 else 0,
            'avg_return_per_company': avg_return,
            'avg_win_rate_per_company': avg_win_rate,
            'avg_trades_per_company': avg_trades_per_company,
            'best_performer': best_company,
            'worst_performer': worst_company,
            'best_return': returns_by_company[best_company],
            'worst_return': returns_by_company[worst_company],
            'company_results': company_results
        }
        
        return portfolio_results
    
    def save_portfolio_results(self, portfolio_results: Dict, company_results: Dict):
        """Save portfolio-level results and analysis"""
        
        if not portfolio_results:
            logger.warning("No portfolio results to save")
            return
        
        # Create portfolio summary
        summary_data = {
            'Metric': [
                'Total Companies', 'Total Initial Capital', 'Total Final Capital', 'Total PnL', 'Total Return %',
                'Total Trades', 'Total Profitable Trades', 'Portfolio Win Rate %', 'Avg Return per Company %',
                'Avg Win Rate per Company %', 'Avg Trades per Company', 'Best Performer', 'Worst Performer',
                'Best Return %', 'Worst Return %'
            ],
            'Value': [
                portfolio_results['total_companies'],
                f"${portfolio_results['total_initial_capital']:.2f}",
                f"${portfolio_results['total_final_capital']:.2f}",
                f"${portfolio_results['total_pnl']:.2f}",
                f"{portfolio_results['total_return_pct']:.2f}%",
                portfolio_results['total_trades'],
                portfolio_results['total_profitable_trades'],
                f"{portfolio_results['portfolio_win_rate']:.2f}%",
                f"{portfolio_results['avg_return_per_company']:.2f}%",
                f"{portfolio_results['avg_win_rate_per_company']:.2f}%",
                f"{portfolio_results['avg_trades_per_company']:.1f}",
                portfolio_results['best_performer'],
                portfolio_results['worst_performer'],
                f"{portfolio_results['best_return']:.2f}%",
                f"{portfolio_results['worst_return']:.2f}%"
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        summary_file = f"{self.output_dir}/portfolio_summary_report.xlsx"
        df_summary.to_excel(summary_file, index=False)
        logger.info(f"Portfolio summary saved to {summary_file}")
        
        # Create company comparison table
        company_comparison = []
        for company, results in company_results.items():
            company_comparison.append({
                'Company': company,
                'Initial Capital': results['initial_capital'],
                'Final Capital': results['final_capital'],
                'Total PnL': results['total_pnl'],
                'Return %': results['total_return_pct'],
                'Total Trades': results['total_trades'],
                'Profitable Trades': results['profitable_trades'],
                'Win Rate %': results['win_rate'],
                'Avg Return %': results['avg_return_pct'],
                'Avg Duration (hours)': results['avg_duration_hours'],
                'Max Profit': results['max_profit'],
                'Max Loss': results['max_loss']
            })
        
        df_comparison = pd.DataFrame(company_comparison)
        comparison_file = f"{self.output_dir}/company_comparison.xlsx"
        df_comparison.to_excel(comparison_file, index=False)
        logger.info(f"Company comparison saved to {comparison_file}")
        
        # Create portfolio visualization
        self.create_portfolio_plots(portfolio_results, company_results)
    
    def create_portfolio_plots(self, portfolio_results: Dict, company_results: Dict):
        """Create portfolio-level visualizations"""
        
        if not company_results:
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Complete Heston Mispricing Strategy Results', fontsize=16)
        
        # Plot 1: Company Returns Comparison
        companies = list(company_results.keys())
        returns = [company_results[company]['total_return_pct'] for company in companies]
        colors = ['green' if ret > 0 else 'red' for ret in returns]
        
        bars = ax1.bar(companies, returns, color=colors, alpha=0.7)
        ax1.set_title('Company Returns Comparison')
        ax1.set_ylabel('Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 2: Win Rate Comparison
        win_rates = [company_results[company]['win_rate'] for company in companies]
        ax2.bar(companies, win_rates, color='skyblue', alpha=0.7)
        ax2.set_title('Company Win Rates')
        ax2.set_ylabel('Win Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Number of Trades per Company
        trade_counts = [company_results[company]['total_trades'] for company in companies]
        ax3.bar(companies, trade_counts, color='orange', alpha=0.7)
        ax3.set_title('Number of Trades per Company')
        ax3.set_ylabel('Number of Trades')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: PnL Distribution
        pnls = [company_results[company]['total_pnl'] for company in companies]
        colors_pnl = ['green' if pnl > 0 else 'red' for pnl in pnls]
        ax4.bar(companies, pnls, color=colors_pnl, alpha=0.7)
        ax4.set_title('Total PnL per Company')
        ax4.set_ylabel('PnL ($)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"{self.output_dir}/portfolio_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Portfolio analysis plot saved to {plot_file}")


class CompanyStrategy:
    def __init__(self, company: str, initial_capital: float = 100.0):
        """
        Initialize company-specific trading strategy
        
        Args:
            company: Company ticker
            initial_capital: Initial capital allocation
        """
        self.company = company
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.active_trades = {}  # {option_id: trade_info}
        self.trade_history = []
        self.hourly_data = []
        self.current_hour = 0
        
        # Trade tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        
        # Create output directories
        self.output_dir = f"scripts/strategy_output/{company}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def select_trading_opportunities(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Select the 2 most undervalued and 2 most overvalued call options
        
        Args:
            data: Current hour's options data
            
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
        
        # Find undervalued options (Market < Heston)
        undervalued = valid_data[valid_data['Market_vs_Heston'] < 0].copy()
        undervalued = undervalued.sort_values('Heston_vs_Market', ascending=False)
        
        # Find overvalued options (Market > Heston)
        overvalued = valid_data[valid_data['Market_vs_Heston'] > 0].copy()
        overvalued = overvalued.sort_values('Market_vs_Heston', ascending=False)
        
        # Select top 2 from each category
        undervalued_ids = undervalued['Option_ID'].head(2).tolist()
        overvalued_ids = overvalued['Option_ID'].head(2).tolist()
        
        logger.info(f"Selected for {self.company}: {len(undervalued_ids)} undervalued, {len(overvalued_ids)} overvalued options")
        
        return undervalued_ids, overvalued_ids
    
    def enter_trade(self, option_id: str, trade_type: str, data: pd.DataFrame) -> bool:
        """
        Enter a new trade if not already in position
        
        Args:
            option_id: Unique option identifier
            trade_type: 'BUY' for undervalued, 'SELL' for overvalued
            data: Current hour's data
            
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
            'entry_mispricing': option_data['Market_vs_Heston']
        }
        
        self.active_trades[option_id] = trade_info
        self.total_trades += 1
        
        logger.info(f"Entered {trade_type} trade for {option_id} at ${option_data['PX_LAST']:.2f}")
        return True
    
    def check_exit_conditions(self, option_id: str, data: pd.DataFrame) -> bool:
        """
        Check if exit conditions are met for a trade
        
        Args:
            option_id: Option identifier
            data: Current hour's data
            
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
            data: Current hour's data
            
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
        
        logger.info(f"Exited {trade['trade_type']} trade for {option_id}: PnL=${pnl:.2f} ({trade['return_pct']:.2f}%)")
        return True
    
    def run_simulation_with_data(self, base_data: pd.DataFrame, num_hours: int = 24) -> Dict:
        """
        Run the complete trading simulation with provided data
        
        Args:
            base_data: Base options data with Heston prices
            num_hours: Number of hours to simulate
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting simulation for {self.company} with ${self.initial_capital} capital")
        
        if base_data.empty:
            logger.error("No data provided, simulation failed")
            return {}
        
        # Simulate hourly data
        self.hourly_data = self.simulate_hourly_data(base_data, num_hours)
        
        # Run simulation hour by hour
        for hour, hour_data in enumerate(self.hourly_data):
            self.current_hour = hour
            logger.info(f"Processing hour {hour} for {self.company}")
            
            # Check exit conditions for existing trades
            options_to_exit = []
            for option_id in list(self.active_trades.keys()):
                if self.check_exit_conditions(option_id, hour_data):
                    options_to_exit.append(option_id)
            
            # Exit trades
            for option_id in options_to_exit:
                self.exit_trade(option_id, hour_data)
            
            # Select new trading opportunities
            undervalued_ids, overvalued_ids = self.select_trading_opportunities(hour_data)
            
            # Enter new trades
            for option_id in undervalued_ids:
                self.enter_trade(option_id, 'BUY', hour_data)
            
            for option_id in overvalued_ids:
                self.enter_trade(option_id, 'SELL', hour_data)
        
        # Close any remaining positions at the end
        final_data = self.hourly_data[-1]
        for option_id in list(self.active_trades.keys()):
            self.exit_trade(option_id, final_data)
        
        # Generate results
        results = self.generate_results()
        self.save_results(results)
        
        return results
    
    def simulate_hourly_data(self, base_data: pd.DataFrame, num_hours: int = 24) -> List[pd.DataFrame]:
        """
        Simulate hourly data by adding random price movements
        
        Args:
            base_data: Base options data
            num_hours: Number of hours to simulate
            
        Returns:
            List of hourly DataFrames
        """
        hourly_data = []
        
        for hour in range(num_hours):
            # Create a copy of base data for this hour
            hour_data = base_data.copy()
            
            # Simulate market price movements (random walk)
            price_change_factor = np.random.normal(1.0, 0.02)  # 2% standard deviation
            
            # Update option prices
            hour_data['PX_LAST'] = hour_data['PX_LAST'] * price_change_factor
            hour_data['Current Price'] = hour_data['Current Price'] * price_change_factor
            
            # Recalculate mispricing metrics
            hour_data['Market_vs_Heston'] = hour_data['PX_LAST'] - hour_data['Heston_Price']
            hour_data['Heston_vs_Market'] = hour_data['Heston_Price'] - hour_data['PX_LAST']
            
            # Add timestamp
            hour_data['Timestamp'] = datetime.now() + timedelta(hours=hour)
            hour_data['Hour'] = hour
            
            hourly_data.append(hour_data)
            
        return hourly_data
    
    def generate_results(self) -> Dict:
        """Generate comprehensive simulation results"""
        
        # Calculate statistics
        win_rate = (self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Trade analysis
        if self.trade_history:
            df_trades = pd.DataFrame(self.trade_history)
            avg_return = df_trades['return_pct'].mean()
            avg_duration = df_trades['duration_hours'].mean()
            max_profit = df_trades['pnl'].max()
            max_loss = df_trades['pnl'].min()
        else:
            avg_return = avg_duration = max_profit = max_loss = 0
        
        results = {
            'company': self.company,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'total_return_pct': total_return,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': win_rate,
            'avg_return_pct': avg_return,
            'avg_duration_hours': avg_duration,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'trade_history': self.trade_history,
            'hourly_data': self.hourly_data
        }
        
        return results
    
    def save_results(self, results: Dict):
        """Save simulation results to files"""
        
        # Save trade log
        if results['trade_history']:
            df_trades = pd.DataFrame(results['trade_history'])
            trade_log_file = f"{self.output_dir}/{self.company}_trade_log.xlsx"
            df_trades.to_excel(trade_log_file, index=False)
            logger.info(f"Trade log saved to {trade_log_file}")
        
        # Save summary report
        summary_data = {
            'Metric': [
                'Company', 'Initial Capital', 'Final Capital', 'Total PnL', 'Total Return %',
                'Total Trades', 'Profitable Trades', 'Win Rate %', 'Avg Return %',
                'Avg Duration (hours)', 'Max Profit', 'Max Loss'
            ],
            'Value': [
                results['company'],
                f"${results['initial_capital']:.2f}",
                f"${results['final_capital']:.2f}",
                f"${results['total_pnl']:.2f}",
                f"{results['total_return_pct']:.2f}%",
                results['total_trades'],
                results['profitable_trades'],
                f"{results['win_rate']:.2f}%",
                f"{results['avg_return_pct']:.2f}%",
                f"{results['avg_duration_hours']:.1f}",
                f"${results['max_profit']:.2f}",
                f"${results['max_loss']:.2f}"
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        summary_file = f"{self.output_dir}/{self.company}_summary_report.xlsx"
        df_summary.to_excel(summary_file, index=False)
        logger.info(f"Summary report saved to {summary_file}")


def main():
    """Main function to run the complete Heston strategy simulation"""
    
    # Define the 9 companies as mentioned in the plan
    companies = [
        'NVDA', 'AMD', 'AVGO', 'BBAI', 'CRCL', 
        'SOUN', 'TSLA', 'AAPL', 'MSFT'
    ]
    
    # Initialize complete strategy
    complete_strategy = CompleteHestonStrategy(
        companies=companies,
        capital_per_company=100.0
    )
    
    # Run simulation
    logger.info("Starting complete Heston mispricing strategy simulation")
    portfolio_results = complete_strategy.run_multi_company_simulation(num_hours=24)
    
    if portfolio_results:
        logger.info("Complete simulation finished successfully!")
        logger.info("Portfolio Summary:")
        logger.info(f"  Total Companies: {portfolio_results['total_companies']}")
        logger.info(f"  Total Initial Capital: ${portfolio_results['total_initial_capital']:.2f}")
        logger.info(f"  Total Final Capital: ${portfolio_results['total_final_capital']:.2f}")
        logger.info(f"  Total PnL: ${portfolio_results['total_pnl']:.2f}")
        logger.info(f"  Total Return: {portfolio_results['total_return_pct']:.2f}%")
        logger.info(f"  Total Trades: {portfolio_results['total_trades']}")
        logger.info(f"  Portfolio Win Rate: {portfolio_results['portfolio_win_rate']:.2f}%")
        logger.info(f"  Best Performer: {portfolio_results['best_performer']} ({portfolio_results['best_return']:.2f}%)")
        logger.info(f"  Worst Performer: {portfolio_results['worst_performer']} ({portfolio_results['worst_return']:.2f}%)")
    else:
        logger.error("Complete simulation failed!")


if __name__ == "__main__":
    main() 