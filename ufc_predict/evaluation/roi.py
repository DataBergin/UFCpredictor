"""ROI simulation: the only metric that matters for practical use."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class ROISimulator:
    """Simulate betting returns using model predictions vs closing lines."""

    def __init__(self, bet_size: float = 100.0, edge_threshold: float = 0.05):
        self.bet_size = bet_size
        self.edge_threshold = edge_threshold

    def simulate(self, y_true: np.ndarray, model_prob: np.ndarray,
                 closing_prob: np.ndarray, closing_odds_american: np.ndarray | None = None
                 ) -> dict[str, any]:
        """
        Simulate flat-bet strategy: bet when model edge > threshold vs closing line.

        Args:
            y_true: actual outcomes (1 = fighter A won)
            model_prob: model's predicted probability for fighter A
            closing_prob: closing line implied probability (devigged)
            closing_odds_american: American odds for payout calculation
        """
        results = []

        for i in range(len(y_true)):
            model_p = model_prob[i]
            closing_p = closing_prob[i]
            actual = y_true[i]

            edge_a = model_p - closing_p
            edge_b = (1 - model_p) - (1 - closing_p)

            bet_side = None
            edge = 0.0

            if edge_a > self.edge_threshold:
                bet_side = "A"
                edge = edge_a
            elif edge_b > self.edge_threshold:
                bet_side = "B"
                edge = edge_b

            if bet_side is None:
                continue

            # Calculate payout using implied odds
            if bet_side == "A":
                fair_odds = closing_p
                won = actual == 1
            else:
                fair_odds = 1 - closing_p
                won = actual == 0

            # Payout from fair odds (assuming we get closing line odds)
            if fair_odds > 0 and fair_odds < 1:
                decimal_odds = 1.0 / fair_odds
                pnl = self.bet_size * (decimal_odds - 1) if won else -self.bet_size
            else:
                pnl = 0

            results.append({
                "fight_idx": i,
                "bet_side": bet_side,
                "edge": edge,
                "model_prob": model_p,
                "closing_prob": closing_p,
                "won": won,
                "pnl": pnl,
            })

        if not results:
            return {
                "total_bets": 0,
                "total_pnl": 0.0,
                "roi_pct": 0.0,
                "win_rate": 0.0,
                "avg_edge": 0.0,
                "results_df": pd.DataFrame(),
            }

        df = pd.DataFrame(results)
        total_pnl = df["pnl"].sum()
        total_wagered = len(df) * self.bet_size

        return {
            "total_bets": len(df),
            "total_wagered": total_wagered,
            "total_pnl": total_pnl,
            "roi_pct": (total_pnl / total_wagered) * 100 if total_wagered > 0 else 0,
            "win_rate": df["won"].mean(),
            "avg_edge": df["edge"].mean(),
            "max_drawdown": self._max_drawdown(df["pnl"].values),
            "sharpe": self._sharpe_ratio(df["pnl"].values),
            "results_df": df,
        }

    def simulate_kelly(self, y_true: np.ndarray, model_prob: np.ndarray,
                       closing_prob: np.ndarray, bankroll: float = 10000.0,
                       kelly_fraction: float = 0.25) -> dict[str, any]:
        """Simulate Kelly criterion betting (fractional Kelly for safety)."""
        results = []
        current_bankroll = bankroll

        for i in range(len(y_true)):
            model_p = model_prob[i]
            closing_p = closing_prob[i]
            actual = y_true[i]

            # Kelly for fighter A
            b = (1 / closing_p) - 1  # decimal odds minus 1
            p = model_p
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0
            kelly *= kelly_fraction  # fractional kelly

            bet_side = None
            bet_fraction = 0.0

            if kelly > 0.01:
                bet_side = "A"
                bet_fraction = kelly
            else:
                # Check fighter B
                b_b = (1 / (1 - closing_p)) - 1
                p_b = 1 - model_p
                q_b = model_p
                kelly_b = (b_b * p_b - q_b) / b_b if b_b > 0 else 0
                kelly_b *= kelly_fraction
                if kelly_b > 0.01:
                    bet_side = "B"
                    bet_fraction = kelly_b

            if bet_side is None:
                results.append({"bankroll": current_bankroll, "pnl": 0})
                continue

            bet_amount = current_bankroll * min(bet_fraction, 0.1)  # cap at 10%

            if bet_side == "A":
                won = actual == 1
                decimal_odds = 1 / closing_p
            else:
                won = actual == 0
                decimal_odds = 1 / (1 - closing_p)

            pnl = bet_amount * (decimal_odds - 1) if won else -bet_amount
            current_bankroll += pnl

            results.append({
                "bankroll": current_bankroll,
                "bet_amount": bet_amount,
                "bet_side": bet_side,
                "won": won,
                "pnl": pnl,
                "kelly_fraction": bet_fraction,
            })

        df = pd.DataFrame(results)
        return {
            "final_bankroll": current_bankroll,
            "total_return_pct": ((current_bankroll - bankroll) / bankroll) * 100,
            "max_drawdown_pct": self._max_drawdown_pct(df["bankroll"].values, bankroll),
            "results_df": df,
        }

    def _max_drawdown(self, pnl_series: np.ndarray) -> float:
        cumulative = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        return float(drawdown.min()) if len(drawdown) > 0 else 0.0

    def _max_drawdown_pct(self, bankroll_series: np.ndarray, initial: float) -> float:
        peak = np.maximum.accumulate(bankroll_series)
        drawdown_pct = (bankroll_series - peak) / peak * 100
        return float(drawdown_pct.min()) if len(drawdown_pct) > 0 else 0.0

    def _sharpe_ratio(self, pnl_series: np.ndarray) -> float:
        if len(pnl_series) == 0 or pnl_series.std() == 0:
            return 0.0
        return float(pnl_series.mean() / pnl_series.std() * np.sqrt(len(pnl_series)))

    def plot_equity_curve(self, results: dict, save_path: str | Path | None = None) -> plt.Figure:
        """Plot cumulative P&L over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        df = results["results_df"]
        if df.empty:
            return fig

        cumulative_pnl = df["pnl"].cumsum()
        ax1.plot(cumulative_pnl.values, "b-", linewidth=1.5)
        ax1.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax1.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, 0,
                         where=cumulative_pnl.values >= 0, alpha=0.3, color="green")
        ax1.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, 0,
                         where=cumulative_pnl.values < 0, alpha=0.3, color="red")
        ax1.set_xlabel("Bet Number")
        ax1.set_ylabel("Cumulative P&L ($)")
        ax1.set_title(f"Equity Curve | ROI: {results['roi_pct']:.1f}% | "
                      f"Bets: {results['total_bets']} | Win Rate: {results['win_rate']:.1%}")
        ax1.grid(True, alpha=0.3)

        # Edge distribution for bets taken
        ax2.hist(df["edge"].values, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
        ax2.axvline(self.edge_threshold, color="red", linestyle="--",
                    label=f"Min edge: {self.edge_threshold:.0%}")
        ax2.set_xlabel("Model Edge vs Closing Line")
        ax2.set_ylabel("Count")
        ax2.set_title("Edge Distribution (Bets Taken)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
