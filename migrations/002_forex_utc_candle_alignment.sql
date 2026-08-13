-- Quarantine pre-cutover OANDA H4 rows that used the provider's legacy daily
-- alignment instead of DeltaQuant's explicit UTC alignment. Public legacy data and
-- the migration backup remain untouched. This migration is idempotent.

BEGIN;

CREATE TABLE IF NOT EXISTS forex.market_candles_quarantine
(LIKE forex.market_candles INCLUDING DEFAULTS);

ALTER TABLE forex.market_candles_quarantine
    ADD COLUMN IF NOT EXISTS quarantine_reason TEXT NOT NULL DEFAULT 'UNSPECIFIED',
    ADD COLUMN IF NOT EXISTS quarantined_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE UNIQUE INDEX IF NOT EXISTS uq_forex_candle_quarantine_grain
ON forex.market_candles_quarantine
    (market, provider, symbol, timeframe, candle_time);

INSERT INTO forex.market_candles_quarantine
    (market, provider, symbol, timeframe, candle_time, open, high, low, close,
     volume, source, volume_type, is_complete, ingested_at, quarantine_reason)
SELECT market, provider, symbol, timeframe, candle_time, open, high, low, close,
       volume, source, volume_type, is_complete, ingested_at,
       'PRE_UTC_ALIGNMENT_OANDA_H4'
FROM forex.market_candles
WHERE market = 'FOREX'
  AND provider = 'OANDA'
  AND timeframe = '4h'
  AND mod(extract(epoch FROM candle_time)::bigint, 14400) <> 0
ON CONFLICT (market, provider, symbol, timeframe, candle_time) DO NOTHING;

DELETE FROM forex.market_candles AS active
USING forex.market_candles_quarantine AS quarantine
WHERE active.market = quarantine.market
  AND active.provider = quarantine.provider
  AND active.symbol = quarantine.symbol
  AND active.timeframe = quarantine.timeframe
  AND active.candle_time = quarantine.candle_time
  AND quarantine.quarantine_reason = 'PRE_UTC_ALIGNMENT_OANDA_H4';

COMMIT;
