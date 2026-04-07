#!/usr/bin/env python3
"""
Analyze backtest results to identify improvement opportunities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ledger_path = PROJECT_ROOT / 'data' / 'backtest_ledger.csv'

df = pd.read_csv(ledger_path)

print("="*60)
print("BACKTEST ANALYSIS")
print("="*60)

# Exit reason breakdown
print("\n1. EXIT REASONS:")
print(df['reason'].value_counts())
print(f"\nTime stops: {(df['reason'].str.contains('Time Stop')).sum()} ({(df['reason'].str.contains('Time Stop')).sum()/len(df)*100:.1f}%)")

# PnL by exit reason
print("\n2. PnL BY EXIT REASON:")
for reason in df['reason'].unique():
    subset = df[df['reason'] == reason]
    avg_pnl = subset['pnl_rupees'].mean()
    win_rate = (subset['pnl_rupees'] > 0).sum() / len(subset) * 100
    print(f"{reason:20s}: Avg PnL: Rs.{avg_pnl:7.2f}, Win Rate: {win_rate:5.1f}%")

# Hold days analysis
print("\n3. HOLD DAYS ANALYSIS:")
print(df['hold_days'].describe())
print(f"\nTrades by hold days:")
print(df['hold_days'].value_counts().sort_index())

# Time stop analysis
time_stops = df[df['reason'].str.contains('Time Stop')]
print(f"\n4. TIME STOP TRADES (n={len(time_stops)}):")
print(f"Avg PnL: Rs.{time_stops['pnl_rupees'].mean():.2f}")
print(f"Win Rate: {(time_stops['pnl_rupees'] > 0).sum() / len(time_stops) * 100:.1f}%")
print(f"Median PnL: Rs.{time_stops['pnl_rupees'].median():.2f}")

# What if we exited time stops earlier?
print("\n5. WHAT IF WE REDUCED MAX_HOLD_DAYS?")
for days in [3, 5, 7]:
    # Simulate: if hold_days >= days and reason is time stop, assume we exit at median PnL
    simulated = df.copy()
    mask = (simulated['reason'].str.contains('Time Stop')) & (simulated['hold_days'] >= days)
    # Rough estimate: scale PnL proportionally to hold days
    simulated.loc[mask, 'pnl_rupees'] = simulated.loc[mask, 'pnl_rupees'] * (days / simulated.loc[mask, 'hold_days'])
    
    total_pnl = simulated['pnl_rupees'].sum()
    win_rate = (simulated['pnl_rupees'] > 0).sum() / len(simulated) * 100
    print(f"  {days} days: Net PnL: Rs.{total_pnl:,.2f}, Win Rate: {win_rate:.1f}%")

# Probability threshold analysis
print("\n6. RECOMMENDATION:")
print("Issue: 89% of trades hit time stop, suggesting:")
print("  - Model predictions not materializing within 10 days")
print("  - Need shorter holding period (5-7 days)")
print("  - Or higher probability threshold (0.65-0.70)")
print("  - Or tighter take profit targets")
