"""
Backtesting Framework

Simulate trading strategy based on CNN model predictions with realistic
transaction costs, position sizing, and risk management.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Tuple
import pickle
from pathlib import Path
import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingBacktest:
    """
    Backtest trading strategy with CNN model predictions.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize backtester.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Backtesting parameters
        self.initial_capital = self.config['backtesting']['initial_capital']
        self.position_size = self.config['backtesting']['position_size']
        self.stop_loss = self.config['backtesting']['stop_loss']
        self.take_profit = self.config['backtesting']['take_profit']
        self.commission_rate = self.config['backtesting']['commission_rate']
        self.borrow_rate = self.config['backtesting']['borrow_rate']  # Annualized
        self.confidence_threshold = self.config['backtesting'].get('confidence_threshold', 0.0)  # Default: no filter
        
        self.model = None
        self.scaler = None
        
    def load_model(self, model_path: str = "models/best_model.h5"):
        """Load trained model."""
        logger.info(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        
    def load_scaler(self, scaler_path: str = "models/feature_scaler.pkl"):
        """Load feature scaler."""
        logger.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
        
    def load_test_data(self):
        """Load test data with labels and prices."""
        logger.info("Loading test data...")
        
        # Load features
        test_features = pd.read_csv("data/processed/test_features.csv", index_col=0, parse_dates=True)
        
        # Load labels
        test_labels = pd.read_csv("data/processed/test_labeled.csv", index_col=0, parse_dates=True)
        
        # Load raw prices for calculating returns
        test_raw = pd.read_csv("data/raw/test_raw.csv", index_col=0, parse_dates=True)
        
        # Load feature names
        with open("data/processed/feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Loaded test data: {len(test_features)} samples, {len(feature_names)} features")
        
        return test_features, test_labels, test_raw, feature_names
    
    def create_sequences(self, X, lookback=60):
        """
        Create sequences for CNN input.
        
        Args:
            X: Feature array
            lookback: Lookback window size
        
        Returns:
            Sequences array
        """
        sequences = []
        for i in range(lookback, len(X)):
            sequences.append(X[i-lookback:i])
        
        return np.array(sequences)
    
    def generate_signals(self, test_features, feature_names, lookback=60):
        """
        Generate trading signals from model predictions.
        
        Args:
            test_features: Test feature DataFrame
            feature_names: List of feature names
            lookback: Lookback window
        
        Returns:
            DataFrame with dates, predictions, and probabilities
        """
        logger.info("Generating trading signals...")
        
        # Prepare features
        X_test = test_features[feature_names].values
        
        # Create sequences
        X_test_seq = self.create_sequences(X_test, lookback)
        
        # Get predictions
        predictions_proba = self.model.predict(X_test_seq, verbose=0)
        predictions = np.argmax(predictions_proba, axis=1)
        
        # Create signals DataFrame
        # Align with dates (skip first 'lookback' rows)
        dates = test_features.index[lookback:]
        
        signals_df = pd.DataFrame({
            'date': dates,
            'signal': predictions,  # 0=SELL, 1=HOLD, 2=BUY
            'prob_sell': predictions_proba[:, 0],
            'prob_hold': predictions_proba[:, 1],
            'prob_buy': predictions_proba[:, 2],
            'confidence': np.max(predictions_proba, axis=1)
        })
        
        signals_df.set_index('date', inplace=True)
        
        logger.info(f"Generated {len(signals_df)} signals")
        logger.info(f"Signal distribution - SELL: {(predictions==0).sum()}, "
                   f"HOLD: {(predictions==1).sum()}, BUY: {(predictions==2).sum()}")
        
        return signals_df
    
    def simulate_trading(self, signals_df, prices_df):
        """
        Simulate trading based on signals with transaction costs.
        
        Args:
            signals_df: DataFrame with trading signals
            prices_df: DataFrame with OHLC prices
        
        Returns:
            DataFrame with trade history and portfolio value
        """
        logger.info("Simulating trading strategy...")
        
        # Initialize portfolio
        cash = self.initial_capital
        position = 0  # Number of shares held (can be negative for short)
        entry_price = 0
        portfolio_values = []
        trades = []
        
        # Daily borrow rate for short positions
        daily_borrow_rate = self.borrow_rate / 252
        
        for date, row in signals_df.iterrows():
            if date not in prices_df.index:
                continue
                
            current_price = prices_df.loc[date, 'Close']
            signal = row['signal']
            confidence = row['confidence']
            
            # Calculate current portfolio value
            position_value = position * current_price
            
            # Borrowing cost for short positions
            borrow_cost = 0
            if position < 0:
                borrow_cost = abs(position_value) * daily_borrow_rate
                cash -= borrow_cost
            
            portfolio_value = cash + position_value
            
            # Trading logic (with confidence threshold filtering)
            trade_occurred = False
            
            # Only trade if confidence exceeds threshold
            if confidence < self.confidence_threshold:
                # Skip this signal - insufficient confidence
                pass
            elif signal == 2 and position <= 0:  # BUY signal and no long position
                # Close short position if any
                if position < 0:
                    shares_to_cover = abs(position)
                    cover_cost = shares_to_cover * current_price
                    commission = cover_cost * self.commission_rate
                    cash -= (cover_cost + commission)
                    
                    trades.append({
                        'date': date,
                        'action': 'COVER_SHORT',
                        'shares': shares_to_cover,
                        'price': current_price,
                        'commission': commission,
                        'portfolio_value': cash
                    })
                    position = 0
                
                # Open long position
                capital_to_use = portfolio_value * self.position_size
                shares_to_buy = int(capital_to_use / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    commission = cost * self.commission_rate
                    
                    if cash >= (cost + commission):
                        cash -= (cost + commission)
                        position = shares_to_buy
                        entry_price = current_price
                        trade_occurred = True
                        
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'commission': commission,
                            'portfolio_value': portfolio_value
                        })
            
            elif signal == 0 and position >= 0:  # SELL signal and no short position
                # Close long position if any
                if position > 0:
                    proceeds = position * current_price
                    commission = proceeds * self.commission_rate
                    cash += (proceeds - commission)
                    
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'shares': position,
                        'price': current_price,
                        'commission': commission,
                        'portfolio_value': cash
                    })
                    position = 0
                
                # Open short position
                capital_to_use = portfolio_value * self.position_size
                shares_to_short = int(capital_to_use / current_price)
                
                if shares_to_short > 0:
                    proceeds = shares_to_short * current_price
                    commission = proceeds * self.commission_rate
                    cash += (proceeds - commission)
                    position = -shares_to_short
                    entry_price = current_price
                    trade_occurred = True
                    
                    trades.append({
                        'date': date,
                        'action': 'SHORT',
                        'shares': shares_to_short,
                        'price': current_price,
                        'commission': commission,
                        'portfolio_value': portfolio_value
                    })
            
            # Check stop loss and take profit
            if position != 0 and entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                
                if position < 0:  # Short position (inverse logic)
                    pnl_pct = -pnl_pct
                
                # Stop loss
                if pnl_pct <= -self.stop_loss:
                    if position > 0:
                        proceeds = position * current_price
                        commission = proceeds * self.commission_rate
                        cash += (proceeds - commission)
                        
                        trades.append({
                            'date': date,
                            'action': 'STOP_LOSS',
                            'shares': position,
                            'price': current_price,
                            'commission': commission,
                            'portfolio_value': cash
                        })
                    else:  # Short position
                        shares_to_cover = abs(position)
                        cover_cost = shares_to_cover * current_price
                        commission = cover_cost * self.commission_rate
                        cash -= (cover_cost + commission)
                        
                        trades.append({
                            'date': date,
                            'action': 'STOP_LOSS_COVER',
                            'shares': shares_to_cover,
                            'price': current_price,
                            'commission': commission,
                            'portfolio_value': cash
                        })
                    
                    position = 0
                    entry_price = 0
                
                # Take profit
                elif pnl_pct >= self.take_profit:
                    if position > 0:
                        proceeds = position * current_price
                        commission = proceeds * self.commission_rate
                        cash += (proceeds - commission)
                        
                        trades.append({
                            'date': date,
                            'action': 'TAKE_PROFIT',
                            'shares': position,
                            'price': current_price,
                            'commission': commission,
                            'portfolio_value': cash
                        })
                    else:  # Short position
                        shares_to_cover = abs(position)
                        cover_cost = shares_to_cover * current_price
                        commission = cover_cost * self.commission_rate
                        cash -= (cover_cost + commission)
                        
                        trades.append({
                            'date': date,
                            'action': 'TAKE_PROFIT_COVER',
                            'shares': shares_to_cover,
                            'price': current_price,
                            'commission': commission,
                            'portfolio_value': cash
                        })
                    
                    position = 0
                    entry_price = 0
            
            # Record portfolio value
            portfolio_values.append({
                'date': date,
                'cash': cash,
                'position': position,
                'position_value': position_value,
                'portfolio_value': portfolio_value,
                'borrow_cost': borrow_cost
            })
        
        # Close any remaining position at the end
        if position != 0:
            final_date = signals_df.index[-1]
            final_price = prices_df.loc[final_date, 'Close']
            
            if position > 0:
                proceeds = position * final_price
                commission = proceeds * self.commission_rate
                cash += (proceeds - commission)
                
                trades.append({
                    'date': final_date,
                    'action': 'CLOSE_LONG',
                    'shares': position,
                    'price': final_price,
                    'commission': commission,
                    'portfolio_value': cash
                })
            else:
                shares_to_cover = abs(position)
                cover_cost = shares_to_cover * final_price
                commission = cover_cost * self.commission_rate
                cash -= (cover_cost + commission)
                
                trades.append({
                    'date': final_date,
                    'action': 'CLOSE_SHORT',
                    'shares': shares_to_cover,
                    'price': final_price,
                    'commission': commission,
                    'portfolio_value': cash
                })
        
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            trades_df.set_index('date', inplace=True)
        
        logger.info(f"Simulation completed. Total trades: {len(trades_df)}")
        
        return portfolio_df, trades_df
    
    def calculate_metrics(self, portfolio_df, trades_df):
        """
        Calculate performance metrics.
        
        Args:
            portfolio_df: Portfolio value over time
            trades_df: Trade history
        
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Calculating performance metrics...")
        
        # Total return
        initial_value = self.initial_capital
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        mean_daily_return = portfolio_df['daily_return'].mean()
        std_daily_return = portfolio_df['daily_return'].std()
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        
        # Maximum drawdown
        cumulative_max = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Win rate
        if len(trades_df) > 0:
            trades_df['pnl'] = trades_df['portfolio_value'].diff()
            winning_trades = (trades_df['pnl'] > 0).sum()
            total_trades = len(trades_df)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0
            total_trades = 0
        
        # Total commissions paid
        total_commissions = trades_df['commission'].sum() if len(trades_df) > 0 else 0
        
        # Total borrow costs
        total_borrow_costs = portfolio_df['borrow_cost'].sum()
        
        # Calmar ratio
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        
        # Trading days
        trading_days = len(portfolio_df)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        
        metrics = {
            'initial_capital': float(initial_value),
            'final_value': float(final_value),
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'total_commissions': float(total_commissions),
            'total_borrow_costs': float(total_borrow_costs),
            'trading_days': int(trading_days)
        }
        
        logger.info("Performance Metrics:")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Annualized Return: {annualized_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Total Trades: {total_trades}")
        
        return metrics
    
    def run_backtest(self):
        """Run complete backtest pipeline."""
        logger.info("Starting backtest...")
        
        # Load model and scaler
        self.load_model()
        self.load_scaler()
        
        # Load test data
        test_features, test_labels, test_raw, feature_names = self.load_test_data()
        
        # Generate signals
        lookback = self.config['features']['lookback_window']
        signals_df = self.generate_signals(test_features, feature_names, lookback)
        
        # Simulate trading
        portfolio_df, trades_df = self.simulate_trading(signals_df, test_raw)
        
        # Calculate metrics
        metrics = self.calculate_metrics(portfolio_df, trades_df)
        
        # Save results
        portfolio_df.to_csv("models/backtest_portfolio.csv")
        if len(trades_df) > 0:
            trades_df.to_csv("models/backtest_trades.csv")
        
        # Save metrics
        import json
        with open("models/backtest_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Backtest completed successfully!")
        logger.info(f"Results saved to models/backtest_*.csv")
        
        return portfolio_df, trades_df, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trading backtest')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Run backtest
    backtester = TradingBacktest(args.config)
    portfolio_df, trades_df, metrics = backtester.run_backtest()
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
    print(f"Final Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Total Commissions: ${metrics['total_commissions']:,.2f}")
    print(f"Total Borrow Costs: ${metrics['total_borrow_costs']:,.2f}")
    print("="*50)
