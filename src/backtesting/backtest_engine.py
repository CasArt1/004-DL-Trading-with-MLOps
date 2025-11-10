"""Backtesting engine with realistic trading costs."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class TradingStrategy:
    """Base trading strategy with ML predictions."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% per trade
        slippage_rate: float = 0.0005,   # 0.05% slippage
        position_size: float = 1.0        # Fraction of capital to use
    ):
        """
        Initialize trading strategy.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
            position_size: Position size as fraction of capital
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.position_size = position_size
        
    def calculate_position_size(self, capital: float, price: float) -> float:
        """
        Calculate position size in shares.
        
        Args:
            capital: Available capital
            price: Current price
        
        Returns:
            Number of shares to trade
        """
        position_value = capital * self.position_size
        shares = position_value / price
        return shares
    
    def apply_costs(
        self,
        price: float,
        shares: float,
        side: str
    ) -> Tuple[float, float]:
        """
        Apply trading costs (commission and slippage).
        
        Args:
            price: Execution price
            shares: Number of shares
            side: 'buy' or 'sell'
        
        Returns:
            Tuple of (adjusted_price, total_cost)
        """
        # Apply slippage
        if side == 'buy':
            adjusted_price = price * (1 + self.slippage_rate)
        else:
            adjusted_price = price * (1 - self.slippage_rate)
        
        # Calculate commission
        trade_value = adjusted_price * shares
        commission = trade_value * self.commission_rate
        
        return adjusted_price, commission


class BacktestEngine:
    """Backtesting engine for trading strategies with ML predictions."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        strategy: TradingStrategy
    ):
        """
        Initialize backtest engine.
        
        Args:
            data: Historical OHLCV data
            predictions: Trading signals (BUY=1, HOLD=0, SELL=-1 or 2)
            strategy: Trading strategy instance
        """
        self.data = data.copy()
        self.predictions = predictions.copy()
        self.strategy = strategy
        self.trades = []
        self.portfolio_history = []
        
    def run_backtest(self) -> Dict:
        """
        Run the backtest simulation.
        
        Returns:
            Dictionary with backtest results
        """
        capital = self.strategy.initial_capital
        position = 0  # Current position in shares
        entry_price = 0
        
        for i in range(len(self.data)):
            timestamp = self.data.index[i]
            price = self.data['Close'].iloc[i]
            
            # Get prediction signal
            if i < len(self.predictions):
                signal = self.predictions.iloc[i]
            else:
                signal = 0
            
            # Convert signal to standard format
            if signal == 2:  # SELL class
                signal = -1
            elif signal == 1:  # BUY class
                signal = 1
            else:  # HOLD
                signal = 0
            
            # Execute trades based on signal
            if signal == 1 and position == 0:  # BUY signal and no position
                shares = self.strategy.calculate_position_size(capital, price)
                adjusted_price, commission = self.strategy.apply_costs(price, shares, 'buy')
                
                cost = adjusted_price * shares + commission
                if cost <= capital:
                    capital -= cost
                    position = shares
                    entry_price = adjusted_price
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'type': 'BUY',
                        'price': price,
                        'adjusted_price': adjusted_price,
                        'shares': shares,
                        'commission': commission,
                        'capital': capital,
                        'position_value': position * price
                    })
            
            elif signal == -1 and position > 0:  # SELL signal and holding position
                adjusted_price, commission = self.strategy.apply_costs(price, position, 'sell')
                
                proceeds = adjusted_price * position - commission
                capital += proceeds
                
                pnl = (adjusted_price - entry_price) * position - commission
                pnl_pct = (adjusted_price - entry_price) / entry_price * 100
                
                self.trades.append({
                    'timestamp': timestamp,
                    'type': 'SELL',
                    'price': price,
                    'adjusted_price': adjusted_price,
                    'shares': position,
                    'commission': commission,
                    'capital': capital,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                position = 0
                entry_price = 0
            
            # Record portfolio value
            portfolio_value = capital + (position * price if position > 0 else 0)
            self.portfolio_history.append({
                'timestamp': timestamp,
                'capital': capital,
                'position': position,
                'price': price,
                'portfolio_value': portfolio_value
            })
        
        # Close any open position at the end
        if position > 0:
            price = self.data['Close'].iloc[-1]
            adjusted_price, commission = self.strategy.apply_costs(price, position, 'sell')
            proceeds = adjusted_price * position - commission
            capital += proceeds
            
            pnl = (adjusted_price - entry_price) * position - commission
            pnl_pct = (adjusted_price - entry_price) / entry_price * 100
            
            self.trades.append({
                'timestamp': self.data.index[-1],
                'type': 'SELL',
                'price': price,
                'adjusted_price': adjusted_price,
                'shares': position,
                'commission': commission,
                'capital': capital,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        # Calculate performance metrics
        results = self.calculate_metrics()
        return results
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trades)
        
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.strategy.initial_capital) / self.strategy.initial_capital * 100
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        num_trades = len(trades_df)
        buy_trades = trades_df[trades_df['type'] == 'BUY']
        sell_trades = trades_df[trades_df['type'] == 'SELL']
        
        winning_trades = sell_trades[sell_trades['pnl'] > 0] if 'pnl' in sell_trades.columns else pd.DataFrame()
        losing_trades = sell_trades[sell_trades['pnl'] <= 0] if 'pnl' in sell_trades.columns else pd.DataFrame()
        
        win_rate = len(winning_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Sharpe ratio (assuming 252 trading days per year)
        returns = portfolio_df['returns'].dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Total commissions
        total_commissions = trades_df['commission'].sum()
        
        metrics = {
            'initial_capital': self.strategy.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return,
            'num_trades': num_trades,
            'num_buy_trades': len(buy_trades),
            'num_sell_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_commissions': total_commissions
        }
        
        return metrics
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return pd.DataFrame(self.trades)
    
    def get_portfolio_df(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        return pd.DataFrame(self.portfolio_history)
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        portfolio_df = self.get_portfolio_df()
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Portfolio value over time
        axes[0].plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], label='Portfolio Value')
        axes[0].axhline(y=self.strategy.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Value ($)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Price with buy/sell signals
        axes[1].plot(self.data.index, self.data['Close'], label='Price', alpha=0.7)
        
        trades_df = self.get_trades_df()
        if len(trades_df) > 0:
            buy_trades = trades_df[trades_df['type'] == 'BUY']
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            
            axes[1].scatter(buy_trades['timestamp'], buy_trades['price'], 
                          color='green', marker='^', s=100, label='Buy', zorder=5)
            axes[1].scatter(sell_trades['timestamp'], sell_trades['price'], 
                          color='red', marker='v', s=100, label='Sell', zorder=5)
        
        axes[1].set_title('Price with Trading Signals')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Drawdown
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        cumulative_returns = (1 + portfolio_df['returns'].fillna(0)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        axes[2].fill_between(portfolio_df['timestamp'], drawdown, 0, alpha=0.3, color='red')
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.close()
    
    def generate_report(self, filepath: str = "reports/backtest_report.txt"):
        """
        Generate a text report of backtest results.
        
        Args:
            filepath: Path to save the report
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        metrics = self.calculate_metrics()
        
        report = f"""
=================================================
        BACKTEST REPORT
=================================================

Initial Capital:        ${metrics['initial_capital']:,.2f}
Final Portfolio Value:  ${metrics['final_value']:,.2f}
Total Return:           ${metrics['final_value'] - metrics['initial_capital']:,.2f} ({metrics['total_return_pct']:.2f}%)

-------------------------------------------------
TRADING STATISTICS
-------------------------------------------------
Total Trades:           {metrics['num_trades']}
Buy Trades:             {metrics['num_buy_trades']}
Sell Trades:            {metrics['num_sell_trades']}
Winning Trades:         {metrics['winning_trades']}
Losing Trades:          {metrics['losing_trades']}
Win Rate:               {metrics['win_rate']:.2f}%

-------------------------------------------------
PERFORMANCE METRICS
-------------------------------------------------
Average Win:            ${metrics['avg_win']:,.2f}
Average Loss:           ${metrics['avg_loss']:,.2f}
Profit Factor:          {metrics['profit_factor']:.2f}
Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}
Max Drawdown:           {metrics['max_drawdown']:.2f}%

-------------------------------------------------
COSTS
-------------------------------------------------
Total Commissions:      ${metrics['total_commissions']:,.2f}
Commission Rate:        {self.strategy.commission_rate * 100:.2f}%
Slippage Rate:          {self.strategy.slippage_rate * 100:.3f}%

=================================================
"""
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {filepath}")
        print(report)
