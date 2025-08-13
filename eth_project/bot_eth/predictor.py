"""
predictor.py
This module implements a simple prediction API for the live ETH futures bot.  It
loads the pre‑trained model and configuration at initialization time and
provides a ``predict`` function that accepts a DataFrame of recent candles and
returns a probability score indicating the likelihood of a profitable trade.
The caller is responsible for fetching and preparing the most recent 1‑minute
candlestick data (at least 240 rows are recommended to avoid NaNs in all
features).  No backtesting or trade execution is performed here – this module
is solely concerned with feature generation and probability inference.
"""

import json
import pickle
from pathlib import Path
from typing import Union

import sys
import pandas as pd

# Ensure parent directory is on sys.path so that features can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

from features import generate_features


class ETHPredictor:
    def __init__(self, model_path: Union[str, Path] = 'eth_model.pkl', config_path: Union[str, Path] = 'config_eth.json'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.feature_cols = cfg['feature_columns']
        self.prob_threshold = cfg['prob_threshold']

    def predict(self, candles: pd.DataFrame) -> float:
        """Given a DataFrame of recent 1‑minute candles, return probability of a win.

        The DataFrame must contain at least ``open``, ``high``, ``low``, ``close`` and
        ``volume`` columns and be sorted in ascending chronological order.
        """
        # Generate features for all rows
        feats = generate_features(candles)
        # Extract the most recent row and select feature columns
        X = feats[self.feature_cols].iloc[-1:].values
        prob = self.model.predict_proba(X)[0, 1]
        return float(prob)

    def should_trade(self, candles: pd.DataFrame) -> bool:
        """Return True if the model indicates a trade should be opened."""
        return self.predict(candles) >= self.prob_threshold


def load_predictor() -> ETHPredictor:
    """Utility function to instantiate a predictor with default paths."""
    return ETHPredictor()