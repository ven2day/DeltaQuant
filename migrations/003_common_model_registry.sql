BEGIN;

CREATE SCHEMA IF NOT EXISTS common;

CREATE TABLE IF NOT EXISTS common.model_registry (
    market VARCHAR(16) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    strategy_version VARCHAR(100) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    artifact_location VARCHAR(1000) NOT NULL,
    artifact_checksum VARCHAR(128) NOT NULL,
    trained_at TIMESTAMPTZ,
    training_data_start TIMESTAMPTZ,
    training_data_end TIMESTAMPTZ,
    training_data_until TIMESTAMPTZ,
    validated_at TIMESTAMPTZ,
    oos_trade_count INTEGER NOT NULL DEFAULT 0,
    oos_win_rate DOUBLE PRECISION,
    oos_profit_factor DOUBLE PRECISION,
    oos_expectancy DOUBLE PRECISION,
    oos_max_drawdown DOUBLE PRECISION,
    oos_sharpe DOUBLE PRECISION,
    calibration_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_registry_identity UNIQUE (
        market, strategy_name, timeframe, strategy_version, model_version
    ),
    CONSTRAINT ck_model_registry_market CHECK (market IN ('NSE', 'FOREX')),
    CONSTRAINT ck_model_registry_status CHECK (
        status IN ('TRAINING','VALIDATING','REJECTED','APPROVED','ACTIVE','RETIRED','FAILED')
    ),
    CONSTRAINT ck_active_status_consistent CHECK (NOT is_active OR status = 'ACTIVE')
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_active_grain
ON common.model_registry (market, strategy_name, timeframe, strategy_version)
WHERE is_active;

CREATE INDEX IF NOT EXISTS ix_model_registry_market_status
ON common.model_registry (market, status, updated_at DESC);

COMMIT;
