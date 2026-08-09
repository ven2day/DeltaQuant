"""Quick config validation script."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from pydantic import ValidationError

from src.config import get_settings, resolve_effective_execution_mode

print("=" * 50)
print("₹DeltaQuant - Configuration Check")
print("=" * 50)

try:
    s = get_settings()
except ValidationError as exc:
    # get_settings()/Settings() now fails closed (raises) for the hazardous C-2/H-10
    # combinations instead of just warning — see src/config/settings.py
    # validate_configuration(). Surface that plainly instead of a raw traceback.
    print("\n[BLOCKED] Configuration failed a fail-closed safety check and refused to start:\n")
    print(str(exc))
    print("\n" + "=" * 50)
    print("[ERROR] Fix the setting(s) above in .env, then re-run this check.")
    print("=" * 50)
    sys.exit(1)

# Required
groq_ok = bool(s.groq_api_key)
print(f"Groq API Key:     {'[OK]' if groq_ok else '[MISSING]'}")

# Optional - DhanHQ
dhan_ok = bool(s.dhan_client_id and (s.dhan_access_token or (s.dhan_pin and s.dhan_totp_secret)))
print(f"DhanHQ API:       {'[OK]' if dhan_ok else '[Not configured - optional]'}")

# Free tier
print(f"\nData Source:      {s.market_data_source}")
print(f"Trading Mode:     {s.trading_mode}")
print(f"Execution Mode:   {s.execution_mode}  (requested)")

# Resolved effective mode: the same conjunction ExecutionService applies at runtime
# (trading_mode=live AND allow_live_orders AND Dhan credentials), so an operator can
# see here — before starting anything — whether this config can ever place a real
# order, rather than inferring it from several settings read together (M-5).
effective_execution_mode = resolve_effective_execution_mode(
    s.execution_mode,
    allow_live_orders=s.allow_live_orders,
    trading_mode=s.trading_mode,
    has_dhan_credentials=bool(s.dhan_client_id and s.dhan_access_token),
)
real_orders = effective_execution_mode == "live"
print(f"Effective Mode:   {effective_execution_mode}  (after C-2 safety resolution)")
print(f"REAL ORDERS:      {'[YES] -- genuine broker orders CAN be placed' if real_orders else '[NO]'}")
print(f"Paper Wallet:     Rs.{s.paper_wallet_balance:,.0f}")
print(f"News Analysis:    {'[Enabled]' if s.enable_news_analysis else '[Disabled]'}")
print(f"Scalping:         {'[Enabled]' if s.scalp_enabled else '[Disabled]'}")

# Telegram
telegram_ok = bool(getattr(s, "telegram_bot_token", None) and getattr(s, "telegram_chat_id", None))
print(f"Telegram Alerts:  {'[OK]' if telegram_ok else '[Not configured - optional]'}")

# Cross-field validation warnings (live-mode creds, risk-param sanity, market hours, ...)
config_warnings = getattr(s, "config_warnings", [])
if config_warnings:
    print(f"\nConfiguration Warnings ({len(config_warnings)}):")
    for warning in config_warnings:
        print(f"  [WARN] {warning}")
else:
    print("\nConfiguration Warnings:  [None]")

print("\n" + "=" * 50)
if groq_ok:
    if config_warnings:
        print(f"[READY] System ready to run ({len(config_warnings)} config warning(s) above).")
    else:
        print("[READY] System ready to run!")
else:
    print("[ERROR] Please set GROQ_API_KEY in .env")
print("=" * 50)
