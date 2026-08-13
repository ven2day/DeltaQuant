# Forex strategy profile

DeltaQuant uses the same production strategy implementations for every market. A
Forex worker loads its own `PRODUCTION_STRATEGY_*_JSON` values from the selected
`.env.forex.*` profile; NSE validation and parameters are never inherited as Forex
approval.

Initially enabled in SHADOW only:

- `ema_adx_trend`
- `donchian_breakout`
- `time_series_momentum`
- `trend_pullback`
- `supertrend_adx_ema`
- `macd_trend_continuation`
- `bollinger_rsi_mean_reversion`

Initially disabled:

- `vwap_mean_reversion`: OANDA volume is tick/update count, not centralized traded volume.
- `opening_range_breakout`: requires Forex-session opening ranges and independent OOS validation.
- `relative_strength_momentum`: no simplistic equity-style benchmark is substituted for a currency-strength model.

Every enabled `market + strategy + timeframe + model_version` grain starts SHADOW or
UNVALIDATED. Promotion requires Forex-only walk-forward/OOS evidence net of spread,
slippage, and holding costs.
