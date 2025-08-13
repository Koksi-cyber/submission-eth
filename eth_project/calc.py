"""
calc.py
This module defines functions for calculating trade quantities, notional values, maintenance
margins, liquidation prices, take‑profit prices and profit and loss (PnL) for long
ETHUSDT futures contracts on Binance.  The formulas implemented here are taken
directly from the user provided specification.  Any values not supplied when
calling these functions will fall back to sensible defaults defined in the
module level constants below.  These functions are pure and side‑effect free
and can therefore be imported from multiple places (training, evaluation and
live code) without any hidden state.

All monetary values are denominated in USDT and prices are denominated in
USDT per ETH.  Quantities represent how many ETH are purchased for a given
margin and leverage.
"""

from dataclasses import dataclass
from typing import Tuple

# Default parameters as described in the specification
DEFAULT_MARGIN: float = 10.0  # USDT used per trade
DEFAULT_LEVERAGE: float = 55.0  # 55× leverage
TAKER_FEE: float = 0.00045  # taker fee per transaction
MAKER_FEE: float = 0.00018  # maker fee per transaction (unused in market orders)
MAINTENANCE_MARGIN_RATE: float = 0.004  # mmr for ETHUSDT on Binance (approximate)


def calc_quantity(entry_price: float, margin: float = DEFAULT_MARGIN, leverage: float = DEFAULT_LEVERAGE) -> float:
    """Return the quantity of ETH purchased for a given entry price.

    Quantity (Q) = (margin × leverage) ÷ entry_price.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    return (margin * leverage) / entry_price


def calc_notional(entry_price: float, quantity: float) -> float:
    """Return the notional value of the position (N = entry_price × Q)."""
    return entry_price * quantity


def calc_maintenance_margin(notional: float, mmr: float = MAINTENANCE_MARGIN_RATE) -> float:
    """Return the maintenance margin (MM = notional × mmr)."""
    return notional * mmr


def calc_liquidation_price(
    entry_price: float,
    quantity: float,
    margin: float = DEFAULT_MARGIN,
    leverage: float = DEFAULT_LEVERAGE,
    taker_fee: float = TAKER_FEE,
    mmr: float = MAINTENANCE_MARGIN_RATE,
) -> float:
    """Return the liquidation price for a long position.

    From the spec:
        P_liq = Entry + (MM − margin + N × F_t) / Q

    where N = entry_price × Q and MM = N × mmr.
    """
    notional = calc_notional(entry_price, quantity)
    mm = calc_maintenance_margin(notional, mmr)
    return entry_price + (mm - margin + notional * taker_fee) / quantity


def calc_take_profit_price(
    entry_price: float,
    quantity: float,
    taker_fee: float = TAKER_FEE,
    profit_target: float = 15.0,
    ) -> float:
    """Return the take‑profit price that yields a net profit_target in USDT.

    P_tp = Entry + (profit_target + N × F_t × 2) / Q

    The spec states a take profit of +15 USDT net on a 10 USDT margin (i.e. 150%).
    We account for trading fees on both entry and exit.
    """
    notional = calc_notional(entry_price, quantity)
    return entry_price + (profit_target + notional * taker_fee * 2) / quantity


def calc_pnl(entry_price: float, exit_price: float, quantity: float, taker_fee: float = TAKER_FEE) -> float:
    """Return the profit or loss in USDT for a trade.

    PnL = Q × (exit_price − entry_price) − N × F_t × 2

    where N = entry_price × Q.  Note that we subtract the taker fee twice (once
    on entry and once on exit) per the spec.  If exit_price is below entry_price
    the result will be negative (a loss).
    """
    notional = calc_notional(entry_price, quantity)
    return quantity * (exit_price - entry_price) - notional * taker_fee * 2


@dataclass
class TradeParams:
    """Convenience dataclass to encapsulate trade parameters used throughout the
    project.  The default values mirror those defined globally.
    """
    margin: float = DEFAULT_MARGIN
    leverage: float = DEFAULT_LEVERAGE
    taker_fee: float = TAKER_FEE
    maker_fee: float = MAKER_FEE
    mmr: float = MAINTENANCE_MARGIN_RATE
    profit_target: float = 15.0

    def compute_all(self, entry_price: float) -> Tuple[float, float, float, float]:
        """Given an entry price, compute quantity, liquidation price and take profit price.

        Returns a tuple of (quantity, p_liq, p_tp, notional).
        """
        q = calc_quantity(entry_price, self.margin, self.leverage)
        n = calc_notional(entry_price, q)
        mm = calc_maintenance_margin(n, self.mmr)
        p_liq = entry_price + (mm - self.margin + n * self.taker_fee) / q
        p_tp = entry_price + (self.profit_target + n * self.taker_fee * 2) / q
        return q, p_liq, p_tp, n