"""
Notifications module for ₹DeltaQuant.
"""

from .telegram import TelegramNotifier, get_notifier

__all__ = ["TelegramNotifier", "get_notifier"]
