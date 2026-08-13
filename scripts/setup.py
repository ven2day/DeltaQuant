"""
Guided one-command setup for DeltaQuant.

Run this first:  uv run python scripts/setup.py

It creates the isolated common/NSE profiles from their templates if needed, checks
the required API keys, surfaces configuration warnings, and prints a readiness checklist.
ASCII-only output so it works on any terminal.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ENV_DIR = ROOT / "env"
ENV_PROFILES = (
    (ENV_DIR / ".env.common", ENV_DIR / ".env.common.example"),
    (ENV_DIR / ".env.nse", ENV_DIR / ".env.nse.example"),
)
LINE = "=" * 64


def _status(ok: bool) -> str:
    return "[ OK ]" if ok else "[MISSING]"


def main() -> None:
    print(LINE)
    print(" DeltaQuant - Setup")
    print(LINE)
    print(f"[i] Python {sys.version_info.major}.{sys.version_info.minor} "
          f"({'OK' if sys.version_info >= (3, 11) else 'needs 3.11+'})")

    # 1. Ensure the isolated NSE profiles exist.
    missing_templates = [
        example
        for target, example in ENV_PROFILES
        if not target.exists() and not example.exists()
    ]
    if missing_templates:
        print("[!] Missing env templates: " + ", ".join(str(path) for path in missing_templates))
        print(LINE)
        return
    created: list[Path] = []
    for target, example in ENV_PROFILES:
        if not target.exists():
            shutil.copy(example, target)
            created.append(target)
            print(f"[+] Created {target.relative_to(ROOT)} from {example.name}")
    if created:
        print()
        print("[ACTION] Open env/.env.common and env/.env.nse and set the required keys:")
        print("   - GROQ_API_KEY        https://console.groq.com/keys")
        print("   - LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY   https://cloud.langfuse.com")
        print()
        print("Then re-run:  uv run python scripts/setup.py")
        print(LINE)
        return

    print("[+] env/.env.common and env/.env.nse present")

    # 2. Load settings (catches missing required keys gracefully).
    try:
        from src.config import get_settings

        settings = get_settings()
    except Exception as exc:
        print(f"[!] Configuration could not load: {exc}")
        print("    Set GROQ_API_KEY in env/.env.common and optionally configure Langfuse.")
        print(LINE)
        return

    # 3. Readiness checklist.
    groq = settings.groq_api_key.get_secret_value()
    groq_ok = bool(groq) and "your_" not in groq
    langfuse_enabled = settings.langfuse_tracing_enabled
    langfuse_public = settings.langfuse_public_key.get_secret_value()
    langfuse_secret = settings.langfuse_secret_key.get_secret_value()
    langfuse_ok = bool(langfuse_public and langfuse_secret)

    print()
    print(f"  Groq API key:     {_status(groq_ok)}")
    print(
        f"  Langfuse tracing: {_status(langfuse_ok)}  "
        f"({'enabled' if langfuse_enabled else 'disabled'}; optional)"
    )
    print(f"  Data source:      {settings.market_data_source}")
    print(f"  Execution mode:   {settings.execution_mode}  "
          f"(allow_live_orders={settings.allow_live_orders})")
    print(f"  Paper wallet:     Rs.{settings.paper_wallet_balance:,.0f}")
    print(f"  Learning loop:    {'on' if settings.enable_learning else 'off'}")
    print(f"  FinOps:           {'on' if settings.finops_enabled else 'off'}")

    warnings = getattr(settings, "config_warnings", [])
    if warnings:
        print()
        for warning in warnings:
            print(f"  [WARN] {warning}")

    print()
    if groq_ok:
        print("[READY] Start the dashboard:")
        print("        uv run deltaquant-nse")
    else:
        print("[ACTION] Set GROQ_API_KEY in env/.env.common, then re-run.")
    print(LINE)


if __name__ == "__main__":
    main()
