"""
train_eth.py
This script trains a binary classification model to predict profitable ETHUSDT
futures trades based on 1‑minute candlestick data.  The training pipeline
performs the following steps:

1. Load raw JSON candlestick data for the designated training period.
2. Combine multiple files into a single chronologically ordered DataFrame.
3. Generate technical features using the ``features`` module.
4. Label each candle as a win or loss using the ``labeler`` module with a
   finite forward horizon.
5. Sample a subset of rows to manage memory and class imbalance.
6. Train an XGBoost classifier to predict the win label and evaluate on a
   hold‑out validation split.
7. Persist the trained model and configuration to disk for later use.

Usage:
    python train_eth.py

The script writes ``eth_model.pkl`` and ``config_eth.json`` into the current
working directory.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

# Ensure the current directory (eth_project) is on sys.path so that
# relative imports work when running as a script
import sys
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb

from calc import TradeParams, DEFAULT_MARGIN, DEFAULT_LEVERAGE, TAKER_FEE, MAINTENANCE_MARGIN_RATE
from features import generate_features
from labeler import label_dataset


def load_json_files(file_paths):
    """Load and concatenate JSON candlestick lists into a single DataFrame."""
    dfs = []
    for fp in file_paths:
        with open(fp, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    # Convert time columns to datetime
    if 'open_time' in df_all.columns:
        df_all['open_time'] = pd.to_datetime(df_all['open_time'])
    else:
        # Some JSON may not use open_time; fallback to close_time in ms
        if 'close_time' in df_all.columns:
            df_all['open_time'] = pd.to_datetime(df_all['close_time'], unit='ms')
    return df_all


def main():
    # Define training period by specifying which files constitute the first year
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    # Use the first two half‑year files (2022‑08→2023‑02 and 2023‑02→2023‑08)
    train_files = [
        data_dir / 'ETHUSDT_1m_2022-08_to_2023-02.json',
        data_dir / 'ETHUSDT_1m_2023-02_to_2023-08.json',
    ]
    print(f"Loading training data from {len(train_files)} files…")
    df_train = load_json_files(train_files)
    # Ensure chronological order
    df_train.sort_values('open_time', inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    # Generate features
    print("Generating features…")
    df_features = generate_features(df_train)

    # Label dataset using a short horizon (e.g. 60 minutes).  A smaller
    # horizon increases the fraction of candles that hit the take‑profit
    # threshold and therefore yields more positive examples.  This helps
    # produce a model that emits higher probabilities for a larger subset
    # of the data, which in turn increases the number of trades executed.
    params = TradeParams()
    label_horizon = 60
    print(f"Labelling dataset using a {label_horizon}‑minute horizon (this may take a while)…")
    df_labeled = label_dataset(df_features, horizon=label_horizon, params=params)

    # Drop rows with NaN values in feature columns or labels
    feature_cols = [c for c in df_labeled.columns if c not in ['open_time', 'label', 'p_liq', 'p_tp', 'quantity']]
    df_labeled_clean = df_labeled.dropna(subset=feature_cols + ['label'])
    print(f"Total labeled rows: {len(df_labeled_clean)}")

    # Oversample the positive class to encourage the classifier to assign
    # higher probabilities on winning setups.  We replicate all positive
    # examples several times and sample negatives to keep the overall
    # dataset size manageable.  This heuristic increases the frequency
    # of positive labels seen by the model and thus raises predicted
    # probabilities during evaluation.
    pos_df = df_labeled_clean[df_labeled_clean['label'] == 1]
    neg_df = df_labeled_clean[df_labeled_clean['label'] == 0]
    # Replicate positives many times to make them more prevalent in the
    # training set.  A higher repeat factor leads the classifier to assign
    # higher probabilities on a wider range of situations.  Keep the
    # dataset size within limits by adjusting the repeat factor.
    repeat_factor = 20
    pos_rep = pd.concat([pos_df] * repeat_factor, ignore_index=True)
    # Downsample negatives to achieve approx 1:3 positive:negative ratio
    neg_sample_size = min(len(pos_rep) * 3, len(neg_df))
    neg_sample = neg_df.sample(n=neg_sample_size, random_state=42)
    df_balanced = pd.concat([pos_rep, neg_sample], ignore_index=True)
    # Shuffle the combined data
    df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # If the balanced dataset is still too large, sample down to 300k rows
    sample_size = min(300_000, len(df_balanced))
    df_sampled = df_balanced.sample(n=sample_size, random_state=42)
    X = df_sampled[feature_cols].values
    y = df_sampled['label'].values

    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training XGBoost model…")
    # Handle class imbalance by weighting positive class higher.  Compute
    # scale_pos_weight = (negatives / positives).  We set a minimum of 1.0
    # to avoid under‑weighting the positive class.  A modest positive weight
    # encourages the model to predict higher probabilities on rare events.
    # For this experiment we do not upweight the positive class any further.
    # Setting ``scale_pos_weight`` to 1.0 encourages the model to consider
    # positive and negative examples with equal importance, which tends
    # to yield higher predicted probabilities on many candles.
    scale_pos_weight = 1.0
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=4,
        verbosity=1,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    print("Validation report:")
    print(classification_report(y_val, y_pred, digits=3))
    print(f"Validation accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Validation AUC: {roc_auc_score(y_val, y_pred_prob):.4f}")

    # Persist model
    model_path = Path('eth_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Saved trained model to {model_path}")

    # Persist configuration
    config = {
        'margin': params.margin,
        'leverage': params.leverage,
        'taker_fee': params.taker_fee,
        'mmr': params.mmr,
        'profit_target': params.profit_target,
        # Probability threshold used for filtering trades during evaluation
        'prob_threshold': 0.55,
        # Horizon used during labelling; saved here for reference
        'label_horizon': label_horizon,
        'feature_columns': feature_cols,
        # Exponent applied to raw probabilities during evaluation to
        # calibrate the distribution (values <1 amplify probabilities).
        'prob_power': 0.2,
        # Minimum raw model probability required to consider a trade.  Raw
        # probabilities below this threshold are ignored even after the
        # power transformation.  This helps filter out very low‑confidence
        # predictions that would otherwise be amplified by the exponent.
        'min_raw_prob': 0.15,
    }
    config_path = Path('config_eth.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {config_path}")


if __name__ == '__main__':
    main()