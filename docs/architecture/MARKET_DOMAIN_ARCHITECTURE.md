# Market-domain architecture

DeltaQuant runs NSE, Forex, and Crypto as peer process domains. They share broker-neutral
intelligence under `src/core`, but each market owns its provider, configuration, runtime,
health, persistence boundary, risk adapter, eligibility factory, and ML artifact namespace.

```text
                         DeltaQuant
        +--------------------+--------------------+
        |                    |                    |
       NSE                 FOREX                CRYPTO
       Dhan                OANDA             unconfigured
        |                    |                    |
        +---------------- shared core ------------+
          models / indicators / features / strategies
          candidates / aggregation / ML / Qwen / risk contracts
```

## Ownership

- `src/core/`: canonical models and reusable trading intelligence. It must not import a
  market implementation.
- `src/markets/nse/`: Dhan adapter, NSE runtime, NSE persistence/risk/eligibility/ML
  factories, and NSE API surface.
- `src/markets/forex/`: OANDA adapter, Forex calendar, runtime, cost/risk model,
  persistence/eligibility/ML factories, and Forex API surface.
- `src/markets/crypto/`: disabled, fail-closed provider scaffold with independent runtime
  and ownership boundaries. No exchange or execution fallback is implied.
- The former `src/market/` compatibility tree has been removed. Internal imports resolve
  directly to `src/core` or the owning NSE/Forex domain.

## Configuration and secrets

Only one broker profile is loaded per worker. Precedence is:

1. `env/.env.common`
2. exactly one of `env/.env.nse`, `env/.env.forex.practice`, or
   `env/.env.forex.live`
3. process environment variables

Root-level dotenv files are not loaded. Crypto is outside the current implementation scope.

Actual environment files are ignored by Git. Safe templates live beside them. Broker
credentials do not belong in `env/.env.common`, the API process, browser responses,
traces, or logs.

Market configuration is isolated under `config/nse` and `config/forex`. Strategy
implementations stay shared; parameters and validation do not.

## Starting services

PowerShell examples:

```powershell
$env:MARKET='NSE'
uv run deltaquant-nse

$env:MARKET='FOREX'
$env:FOREX_ENVIRONMENT='practice'
uv run deltaquant-forex

$env:MARKET='CRYPTO'
uv run deltaquant-crypto

uv run --extra web deltaquant-api
```

The API is a read-only aggregation service over market-owned operational snapshots. It
does not run trading cycles and should receive only common UI/database configuration.
Systemd units are supplied in `deploy/systemd/` for independent restart and credential
scoping.

## Persistence migration

PostgreSQL/Timescale ownership uses `common`, `nse`, `forex`, and `crypto` schemas.
Repository boundaries reject cross-market access. The migration is deliberately
non-destructive:

```powershell
uv run python scripts/migrate_market_schemas.py
uv run python scripts/migrate_market_schemas.py --apply
```

The first command is a dry run. `--apply` creates a timestamped backup schema, creates
the market schemas/tables/indexes, copies only rows with explicit market identity, and
retains legacy public tables for rollback. Validate row counts before retiring any legacy
table.

ML artifacts and caches are market-qualified. Artifact paths begin with
`artifacts/nse`, `artifacts/forex`, or `artifacts/crypto`; cache and event keys begin with
the lowercase market name. Eligibility remains independent per market.

## Safety

`GLOBAL_KILL_SWITCH` overrides `NSE_KILL_SWITCH`, `FOREX_KILL_SWITCH`, and
`CRYPTO_KILL_SWITCH`. Market workers fail closed if their own provider/configuration is
missing. OANDA live environment selection alone cannot enable execution, and Crypto is
disabled until an explicit provider is implemented.
