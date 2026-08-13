# NSE and Forex peer domains

DeltaQuant has two independently bootstrapped market workers over a broker-neutral
core. Crypto is outside this migration.

```text
src/core/                 shared indicators, features, strategies, candidates,
                          aggregation, ML contracts, Qwen policy, risk contracts
src/markets/nse/          Dhan, NSE data/session/universe/risk/config/persistence/runtime
src/markets/forex/        OANDA, Forex data/session/risk/config/persistence/runtime
src/background/           process-isolated validator and challenger trainer launchers
```

Ownership rules:

- `src/core` cannot import Dhan, OANDA, or either market domain.
- Market workers load `env/.env.common` plus exactly one market profile.
- Repositories and cache/event keys bind market identity rather than accepting an
  ambiguous caller-supplied default.
- Live/paper workers perform inference only. Validation and training are separate OS
  processes.
- Runtime inference selects only an ACTIVE model from `common.model_registry`; a
  TRAINING, VALIDATING, REJECTED or merely APPROVED challenger cannot replace it.
- OANDA broker execution remains disabled even when the live data endpoint is chosen.

Entry points:

```bash
uv run deltaquant-nse
uv run deltaquant-forex
uv run --extra web deltaquant-api
uv run deltaquant-validator --market NSE
uv run deltaquant-validator --market FOREX
uv run deltaquant-ml-trainer --market NSE
uv run deltaquant-ml-trainer --market FOREX
```
