# Runtime logs

Market workers write independently to `logs/nse/` and `logs/forex/`. Log payloads are
gitignored and every structured record carries market, provider and runtime identity.
