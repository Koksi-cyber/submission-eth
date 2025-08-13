"""
eval_eth.py
This script evaluates a trained ETHUSDT futures signal model on a hold‑out
dataset.  It performs a realistic backtest by sequentially stepping through
minute‑level candlestick data, generating features, computing model
probabilities and opening trades whenever the predicted probability exceeds
the configured threshold.  Each open trade is then simulated forward (no
future leakage) until the take‑profit or liquidation price is hit.  The script
produces detailed trade logs and summary statistics including monthly and
yearly accuracies, profit factor and maximum drawdown.

Usage:
    python eval_eth.py --set test
    python eval_eth.py --set backtest

The ``--set`` flag determines which one‑year period to evaluate: ``test``
corresponds to the second year of data (2023‑08→2024‑08) and ``backtest`` to
the third year (2024‑08→2025‑08).  The script expects ``eth_model.pkl`` and
``config_eth.json`` to be present in the working directory.
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import sys
import pandas as pd

# Ensure package imports work when running as a script
sys.path.append(str(Path(__file__).resolve().parent))

from features import generate_features
from calc import TradeParams
from labeler import backtest_trade


def load_data_for_set(which: str) -> pd.DataFrame:
    """Load and concatenate candlestick JSON files for the specified set.

    Parameters
    ----------
    which : {'test', 'backtest'}
        Identifier for the evaluation period.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted DataFrame of candlesticks.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    if which == 'test':
        files = [
            data_dir / 'ETHUSDT_1m_2023-08_to_2024-02.json',
            data_dir / 'ETHUSDT_1m_2024-02_to_2024-08.json',
        ]
    elif which == 'backtest':
        files = [
            data_dir / 'ETHUSDT_1m_H1.json',
            data_dir / 'ETHUSDT_1m_H2.json',
        ]
    else:
        raise ValueError(f"unknown set: {which}")
    dfs = []
    for fp in files:
        with open(fp, 'r') as f:
            data = json.load(f)
        dfs.append(pd.DataFrame(data))
    df = pd.concat(dfs, ignore_index=True)
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'])
    else:
        df['open_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df.sort_values('open_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def evaluate(set_name: str):
    # Load configuration and model
    with open('config_eth.json', 'r') as f:
        cfg = json.load(f)
    with open('eth_model.pkl', 'rb') as f:
        model = pickle.load(f)
    feature_cols = cfg['feature_columns']
    prob_threshold = cfg['prob_threshold']
    prob_power = cfg.get('prob_power', 1.0)
    min_raw_prob = cfg.get('min_raw_prob', 0.0)
    # Trade parameters
    params = TradeParams(
        margin=cfg['margin'],
        leverage=cfg['leverage'],
        taker_fee=cfg['taker_fee'],
        mmr=cfg['mmr'],
        profit_target=cfg['profit_target'],
    )
    # Load data for the selected set
    print(f"Loading {set_name} data…")
    df = load_data_for_set(set_name)
    # Generate features for the entire period
    print("Generating features…")
    df_features = generate_features(df)
    # Align features order
    X_all = df_features[feature_cols].values
    # Compute probabilities for every candle and apply optional
    # power transformation to calibrate the distribution.  Raising
    # probabilities to a power < 1 inflates small values and increases
    # the number of signals crossing the threshold while preserving the
    # ranking order (monotonic transformation).  This knob is stored
    # in the config under ``prob_power``.
    print("Predicting probabilities…")
    raw_prob_all = model.predict_proba(X_all)[:, 1]
    prob_all = raw_prob_all ** prob_power

    trades = []  # list of dicts capturing trade details
    loss_streak = 0
    pause_until_index = -1  # index until which trades are paused
    PAUSE_DURATION_MINUTES = 60  # number of minutes to pause after 5 losses
    i = 0
    # Step through every candle.  We allow overlapping trades: multiple
    # trades can be opened in successive minutes without waiting for the
    # previous trade to exit.  This increases the number of signals
    # executed and better reflects a high‑frequency strategy.  A pause
    # period is still enforced after a streak of losses.
    while i < len(df_features):
        # Skip if we are in pause period
        if i < pause_until_index:
            i += 1
            continue
        raw_prob = raw_prob_all[i]
        prob = prob_all[i]
        # Skip if the raw model probability is below the minimum required
        if raw_prob < min_raw_prob:
            i += 1
            continue
        if prob >= prob_threshold:
            # Open a trade at this candle
            entry_time = df_features.iloc[i]['open_time']
            entry_price = float(df_features.iloc[i]['close'])
            # Perform an unbounded forward scan to determine outcome
            label, exit_price, pnl, p_tp, exit_idx = backtest_trade(df, i, params)
            trades.append({
                'entry_time': entry_time.isoformat(),
                'entry_index': i,
                'entry_price': entry_price,
                'prob': prob,
                'p_tp': p_tp,
                'p_liq': params.compute_all(entry_price)[1],
                'exit_time': df_features.iloc[exit_idx]['open_time'].isoformat(),
                'exit_index': exit_idx,
                'exit_price': exit_price,
                'pnl': pnl,
                'win': label,
            })
            # Update loss streak and handle pause
            if label == 1:
                loss_streak = 0
            else:
                loss_streak += 1
            if loss_streak >= 5:
                pause_until_index = i + PAUSE_DURATION_MINUTES
                loss_streak = 0
            # Advance to next candle (do not skip to exit_idx)
            i += 1
        else:
            i += 1

    # Create DataFrame of trades
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("No trades were executed.  Check your threshold or model.")
        return

    # Save trades to CSV
    trades_df.to_csv('eth_trades.csv', index=False)
    print(f"Saved trade log to eth_trades.csv with {len(trades_df)} trades.")

    # Compute monthly and yearly accuracies
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['year'] = trades_df['entry_time'].dt.year
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly_stats = trades_df.groupby('month')['win'].agg(['count', 'sum'])
    monthly_stats['accuracy'] = monthly_stats['sum'] / monthly_stats['count']
    monthly_stats = monthly_stats.reset_index()
    monthly_stats.to_csv('eth_monthly.csv', index=False)
    # Yearly stats
    yearly_stats = trades_df.groupby('year')['win'].agg(['count', 'sum'])
    yearly_stats['accuracy'] = yearly_stats['sum'] / yearly_stats['count']
    yearly_stats = yearly_stats.reset_index()
    yearly_stats.to_csv('eth_yearly.csv', index=False)
    print("Saved monthly and yearly statistics.")

    # Compute overall performance metrics
    total_wins = trades_df['win'].sum()
    total_trades = len(trades_df)
    yearly_accuracy = total_wins / total_trades if total_trades else 0
    # Rolling last 3 months accuracy
    trades_df = trades_df.sort_values('entry_time')
    last3cut = trades_df['entry_time'].max() - pd.DateOffset(months=3)
    last3 = trades_df[trades_df['entry_time'] >= last3cut]
    last3_accuracy = last3['win'].sum() / len(last3) if len(last3) else 0
    # Profit factor: sum of profits / sum of losses (absolute)
    profits = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    losses = -trades_df[trades_df['pnl'] < 0]['pnl'].sum()
    profit_factor = profits / losses if losses > 0 else float('inf')
    # Max drawdown
    equity = 100.0
    peak = equity
    drawdowns = []
    for pnl in trades_df['pnl']:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        drawdowns.append(dd)
    max_drawdown = max(drawdowns) if drawdowns else 0

    # Write summary markdown
    summary_lines = [
        f"# ETHUSDT Futures Backtest Summary ({set_name})",
        "",
        f"Total trades: {total_trades}",
        f"Total wins: {total_wins}",
        f"Yearly accuracy: {yearly_accuracy:.3f}",
        f"Last 3 months accuracy: {last3_accuracy:.3f}",
        f"Profit factor: {profit_factor:.3f}",
        f"Max drawdown: {max_drawdown:.3f}",
        "",
        "## Monthly Performance",
    ]
    for _, row in monthly_stats.iterrows():
        summary_lines.append(
            f"- {row['month']}: {row['sum']} wins / {row['count']} trades (acc {row['accuracy']:.3f})"
        )
    summary_text = "\n".join(summary_lines)
    with open('eth_summary.md', 'w') as f:
        f.write(summary_text)
    print("Saved summary to eth_summary.md")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ETH futures model on hold‑out set')
    parser.add_argument('--set', choices=['test', 'backtest'], default='test', help='which one‑year period to evaluate')
    args = parser.parse_args()
    evaluate(args.set)