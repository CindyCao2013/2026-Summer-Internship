"""Unified data access for factor research (no parquet cache)."""

from core.data.panel_reader import get_daily_panel, get_minute_panel

__all__ = ["get_daily_panel", "get_minute_panel"]
