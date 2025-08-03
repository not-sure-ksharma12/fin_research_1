import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HestonTradingStrategy:
    def __init__(self, company: str, initial_capital: float = 100.0):
        """
        Initialize the Heston mispricing trading strategy
        
        Args:
            company: Company ticker (e.g., 'NVDA')
            initial_capital: Initial capital allocation per company
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
        
    def load_historical_data(self, data_file: str) -> pd.DataFrame:
        """
        Load historical options data from Excel file
        
        Args:
            data_file: Path to the Excel file with options data
            
        Returns:
            DataFrame with options data
        """
        try:
            logger.info(f"Loading data from {data_file}")
            df = pd.read_excel(data_file)
            
            # Ensure required columns exist
            required_columns = [
                'Ticker', 'Stock Price', 'Strike Price', 'Option Price (BBG)', 
                'Heston Price', 'Option Type', 'Time to Expiration'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Filter for call options only
            df = df[df['Option Type'] == 'Call'].copy()
            
            # Add unique option identifier
            df['Option_ID'] = df.apply(
                lambda row: f"{row['Ticker']}_{row['Strike Price']}_{row['Option Type']}", 
                axis=1
            )
            
            # Calculate mispricing metrics
            df['Market_vs_Heston'] = df['Option Price (BBG)'] - df['Heston Price']
            df['Heston_vs_Market'] = df['Heston Price'] - df['Option Price (BBG)']
            
            logger.info(f"Loaded {len(df)} call options for {self.company}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
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
            hour_data['Option Price (BBG)'] = hour_data['Option Price (BBG)'] * price_change_factor
            hour_data['Stock Price'] = hour_data['Stock Price'] * price_change_factor
            
            # Recalculate mispricing metrics
            hour_data['Market_vs_Heston'] = hour_data['Option Price (BBG)'] - hour_data['Heston Price']
            hour_data['Heston_vs_Market'] = hour_data['Heston Price'] - hour_data['Option Price (BBG)']
            
            # Add timestamp
            hour_data['Timestamp'] = datetime.now() + timedelta(hours=hour)
            hour_data['Hour'] = hour
            
            hourly_data.append(hour_data)
            
        return hourly_data
    
    def select_trading_opportunities(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Select the 2 most undervalued and 2 most overvalued call options
        
        Args:
            data: Current hour's options data
            
        Returns:
            Tuple of (undervalued_option_ids, overvalued_option_ids)
        """
        # Filter for valid data (non-null prices)
        valid_data = data.dropna(subset=['Option Price (BBG)', 'Heston Price'])
        
        if len(valid_data) < 4:
            logger.warning(f"Insufficient data for {self.company}: only {len(valid_data)} valid options")
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
            'entry_market_price': option_data['Option Price (BBG)'],
            'entry_heston_price': option_data['Heston Price'],
            'strike_price': option_data['Strike Price'],
            'stock_price': option_data['Stock Price'],
            'contracts': contracts,
            'position_size': position_size,
            'entry_mispricing': option_data['Market_vs_Heston']
        }
        
        self.active_trades[option_id] = trade_info
        self.total_trades += 1
        
        logger.info(f"Entered {trade_type} trade for {option_id} at ${option_data['Option Price (BBG)']:.2f}")
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
        current_market_price = current_data['Option Price (BBG)']
        current_heston_price = current_data['Heston Price']
        current_mispricing = current_data['Market_vs_Heston']
        
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
        current_market_price = current_data['Option Price (BBG)']
        
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
            'exit_heston_price': current_data['Heston Price'],
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
    
    def run_simulation(self, data_file: str, num_hours: int = 24) -> Dict:
        """
        Run the complete trading simulation
        
        Args:
            data_file: Path to options data file
            num_hours: Number of hours to simulate
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting simulation for {self.company} with ${self.initial_capital} capital")
        
        # Load and prepare data
        base_data = self.load_historical_data(data_file)
        if base_data.empty:
            logger.error("No data loaded, simulation failed")
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
        
        # Create PnL plot
        self.create_pnl_plot(results)
    
    def create_pnl_plot(self, results: Dict):
        """Create PnL visualization"""
        
        if not results['trade_history']:
            logger.warning("No trade history to plot")
            return
        
        df_trades = pd.DataFrame(results['trade_history'])
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.company} Heston Mispricing Strategy Results', fontsize=16)
        
        # Plot 1: Cumulative PnL
        cumulative_pnl = df_trades['pnl'].cumsum()
        ax1.plot(range(len(cumulative_pnl)), cumulative_pnl, 'b-', linewidth=2)
        ax1.set_title('Cumulative PnL')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative PnL ($)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual Trade PnL
        colors = ['green' if pnl > 0 else 'red' for pnl in df_trades['pnl']]
        ax2.bar(range(len(df_trades)), df_trades['pnl'], color=colors, alpha=0.7)
        ax2.set_title('Individual Trade PnL')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('PnL ($)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trade Duration vs Return
        ax3.scatter(df_trades['duration_hours'], df_trades['return_pct'], 
                   c=df_trades['pnl'], cmap='RdYlGn', alpha=0.7)
        ax3.set_title('Trade Duration vs Return %')
        ax3.set_xlabel('Duration (hours)')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Entry vs Exit Mispricing
        ax4.scatter(df_trades['entry_mispricing'], df_trades['pnl'], 
                   c=df_trades['return_pct'], cmap='RdYlGn', alpha=0.7)
        ax4.set_title('Entry Mispricing vs PnL')
        ax4.set_xlabel('Entry Mispricing ($)')
        ax4.set_ylabel('PnL ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"{self.output_dir}/{self.company}_pnl_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PnL analysis plot saved to {plot_file}")


def main():
    """Main function to run the NVDA trading strategy simulation"""
    
    # Initialize strategy for NVDA
    strategy = HestonTradingStrategy(company='NVDA', initial_capital=100.0)
    
    # Run simulation using the most recent NVDA data
    data_file = "analysis/nvda_options_2025-12-19_heston.xlsx"
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    # Run simulation
    results = strategy.run_simulation(data_file, num_hours=24)
    
    if results:
        logger.info("Simulation completed successfully!")
        logger.info(f"Final Results for {results['company']}:")
        logger.info(f"  Initial Capital: ${results['initial_capital']:.2f}")
        logger.info(f"  Final Capital: ${results['final_capital']:.2f}")
        logger.info(f"  Total PnL: ${results['total_pnl']:.2f}")
        logger.info(f"  Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"  Total Trades: {results['total_trades']}")
        logger.info(f"  Win Rate: {results['win_rate']:.2f}%")
    else:
        logger.error("Simulation failed!")


if __name__ == "__main__":
    main() 