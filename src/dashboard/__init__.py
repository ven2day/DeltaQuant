"""
Dashboard Module

Shared trading-state schema for the live loop and the web UI.
"""

from .stats import TradingDashboard, TradingStats

__all__ = ["TradingDashboard", "TradingStats"]
