#!/usr/bin/env python3
"""Plot exact selected-book performance versus the test-universe equal weight."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "TGD20": ROOT
    / "research/reports/tgd_v1/portfolio/long_book_excess_daily_size_industry.csv",
    "FlowDensity20": ROOT
    / "research/reports/l2_flow_density_v1/validation_v1/long_book_excess_daily_size_industry.csv",
}


def compound(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod() - 1.0


def main() -> None:
    for factor_id, source in SOURCES.items():
        df = pd.read_csv(source, index_col=0, parse_dates=True)
        out = ROOT / "research_delivery/selected_factors" / factor_id / "plots"
        out.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            compound(df["long_book_return"]),
            label="Selected long book (EW)",
            lw=1.4,
        )
        ax.plot(
            compound(df["test_universe_ew_return"]),
            label="Valid test universe (EW)",
            lw=1.2,
        )
        ax.set_title(f"{factor_id}: selected book vs test-universe EW")
        ax.set_ylabel("Cumulative return")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "long_book_vs_universe.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            compound(df["long_book_excess_return"]),
            color="darkgreen",
            lw=1.4,
        )
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{factor_id}: cumulative exact universe-excess return")
        ax.set_ylabel("Cumulative excess return")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "long_book_excess_curve.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
