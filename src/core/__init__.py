"""Broker-neutral DeltaQuant intelligence and infrastructure boundaries.

Market domains may import from :mod:`src.core`; shared core code must never import
from :mod:`src.markets.nse`, :mod:`src.markets.forex`, or :mod:`src.markets.crypto`.
Legacy modules remain as compatibility locations while imports migrate gradually.
"""

