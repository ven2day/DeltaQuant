-- Non-destructive logical market separation for DeltaQuant.
-- Run through scripts/migrate_market_schemas.py so a timestamped backup schema
-- is created before this migration is applied.  Legacy public tables are retained.

CREATE SCHEMA IF NOT EXISTS common;
CREATE SCHEMA IF NOT EXISTS nse;
CREATE SCHEMA IF NOT EXISTS forex;
CREATE SCHEMA IF NOT EXISTS crypto;

DO $$
DECLARE
    target_schema text;
BEGIN
    FOREACH target_schema IN ARRAY ARRAY['nse', 'forex', 'crypto'] LOOP
        IF to_regclass('public.market_candles') IS NOT NULL
           AND EXISTS (
               SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = 'market_candles'
                 AND column_name = 'market'
           ) THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I.market_candles '
                '(LIKE public.market_candles INCLUDING ALL)',
                target_schema
            );
        END IF;

        IF to_regclass('public.signal_history') IS NOT NULL THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I.signals '
                '(LIKE public.signal_history INCLUDING ALL)',
                target_schema
            );
        END IF;

        IF to_regclass('public.decision_logs') IS NOT NULL THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I.decisions '
                '(LIKE public.decision_logs INCLUDING ALL)',
                target_schema
            );
        END IF;

        IF to_regclass('public.paper_positions') IS NOT NULL
           AND EXISTS (
               SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = 'paper_positions'
                 AND column_name = 'market'
           ) THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I.positions '
                '(LIKE public.paper_positions INCLUDING ALL)',
                target_schema
            );
        END IF;

        IF to_regclass('public.paper_orders') IS NOT NULL
           AND EXISTS (
               SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = 'paper_orders'
                 AND column_name = 'market'
           ) THEN
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I.orders '
                '(LIKE public.paper_orders INCLUDING ALL)',
                target_schema
            );
        END IF;

        -- Fresh installations may not have legacy public tables. Define a stable,
        -- market-owned minimum schema instead of leaving a market half-created.
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I.market_candles ('
            'market varchar(16) NOT NULL, provider varchar(24) NOT NULL, '
            'symbol varchar(32) NOT NULL, timeframe varchar(8) NOT NULL, '
            'candle_time timestamptz NOT NULL, open double precision NOT NULL, '
            'high double precision NOT NULL, low double precision NOT NULL, '
            'close double precision NOT NULL, volume bigint NOT NULL DEFAULT 0, '
            'source varchar(24) NOT NULL, volume_type varchar(32) NOT NULL, '
            'is_complete boolean NOT NULL DEFAULT true, '
            'ingested_at timestamptz NOT NULL DEFAULT now(), '
            'PRIMARY KEY (market, provider, symbol, timeframe, candle_time))',
            target_schema
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I.signals ('
            'signal_id text PRIMARY KEY, symbol text NOT NULL, timeframe text NOT NULL, '
            'side text NOT NULL, strategy text NOT NULL, payload jsonb NOT NULL, '
            'created_at timestamptz NOT NULL DEFAULT now())',
            target_schema
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I.decisions ('
            'decision_id text PRIMARY KEY, signal_id text, symbol text NOT NULL, '
            'final_action text NOT NULL, rejection_reason text, payload jsonb NOT NULL, '
            'created_at timestamptz NOT NULL DEFAULT now())',
            target_schema
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I.positions ('
            'position_id text PRIMARY KEY, symbol text NOT NULL, side text NOT NULL, '
            'quantity numeric NOT NULL, payload jsonb NOT NULL, '
            'updated_at timestamptz NOT NULL DEFAULT now())',
            target_schema
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I.orders ('
            'order_id text PRIMARY KEY, intent_id text UNIQUE NOT NULL, '
            'symbol text NOT NULL, side text NOT NULL, status text NOT NULL, '
            'payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now())',
            target_schema
        );

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I.strategy_registry ('
            'identity text PRIMARY KEY, payload jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())',
            target_schema
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I.ml_predictions ('
            'prediction_id text PRIMARY KEY, symbol text NOT NULL, timeframe text NOT NULL, '
            'settled_candle_timestamp timestamptz NOT NULL, model_version text NOT NULL, '
            'payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now())',
            target_schema
        );
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I.market_candles '
            '(symbol, timeframe, candle_time DESC)',
            'ix_' || target_schema || '_candles_symbol_timeframe', target_schema
        );
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I.signals '
            '(symbol, timeframe, created_at DESC)',
            'ix_' || target_schema || '_signals_symbol_timeframe', target_schema
        );
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I.decisions (symbol, created_at DESC)',
            'ix_' || target_schema || '_decisions_symbol', target_schema
        );
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I.orders (symbol, status, created_at DESC)',
            'ix_' || target_schema || '_orders_symbol_status', target_schema
        );
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I.ml_predictions '
            '(symbol, timeframe, settled_candle_timestamp DESC)',
            'ix_' || target_schema || '_ml_symbol_timeframe', target_schema
        );
    END LOOP;
END $$;

-- Convert the empty market-owned candle tables before copying legacy rows.  The
-- public hypertable stays untouched and available as a rollback source.
DO $$
DECLARE
    target_schema text;
    qualified_table text;
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        FOREACH target_schema IN ARRAY ARRAY['nse', 'forex', 'crypto'] LOOP
            qualified_table := format('%I.market_candles', target_schema);
            BEGIN
                PERFORM create_hypertable(
                    qualified_table,
                    by_range('candle_time'),
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            EXCEPTION WHEN undefined_function THEN
                PERFORM create_hypertable(
                    qualified_table,
                    'candle_time',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            END;
        END LOOP;
    END IF;
END $$;

-- Copy only rows whose market identity is explicit. Ambiguous legacy signal and
-- decision rows stay in public for manual attribution; this avoids silently
-- assigning old Forex/Crypto decisions to NSE.
DO $$
DECLARE
    target_schema text;
    market_name text;
BEGIN
    FOREACH target_schema IN ARRAY ARRAY['nse', 'forex', 'crypto'] LOOP
        market_name := upper(target_schema);
        IF to_regclass('public.market_candles') IS NOT NULL
           AND EXISTS (
               SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = 'market_candles'
                 AND column_name = 'market'
           ) THEN
            EXECUTE format(
                'INSERT INTO %I.market_candles SELECT * FROM public.market_candles '
                'WHERE upper(market) = %L ON CONFLICT DO NOTHING',
                target_schema, market_name
            );
        END IF;
        IF to_regclass('public.paper_positions') IS NOT NULL
           AND EXISTS (
               SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = 'paper_positions'
                 AND column_name = 'market'
           ) THEN
            EXECUTE format(
                'INSERT INTO %I.positions SELECT * FROM public.paper_positions '
                'WHERE upper(market) = %L ON CONFLICT DO NOTHING',
                target_schema, market_name
            );
        END IF;
        IF to_regclass('public.paper_orders') IS NOT NULL
           AND EXISTS (
               SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = 'paper_orders'
                 AND column_name = 'market'
           ) THEN
            EXECUTE format(
                'INSERT INTO %I.orders SELECT * FROM public.paper_orders '
                'WHERE upper(market) = %L ON CONFLICT DO NOTHING',
                target_schema, market_name
            );
        END IF;
    END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS common.market_schema_migrations (
    migration_id text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now(),
    details jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO common.market_schema_migrations(migration_id, details)
VALUES ('001_market_schema_isolation', '{"legacy_tables_retained": true}'::jsonb)
ON CONFLICT (migration_id) DO NOTHING;
