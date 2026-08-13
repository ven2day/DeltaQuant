"""
Generate the dashboard login credentials required by ENABLE_WEB_UI=true.

    uv run python scripts/set_dashboard_password.py

Prompts for a password (hidden input, confirmed twice), then prints two lines to paste
into env/.env.common:

    WEB_UI_PASSWORD_HASH=pbkdf2_sha256$600000$...$...
    WEB_UI_SESSION_SECRET=<random>

Never writes to the profile itself -- operator configuration is edited by convention throughout this
project (see src/config/settings.py), and a script silently mutating it would be a
surprising side effect for something this security-sensitive. Run this again any time
you want to change the password; it prints a fresh WEB_UI_SESSION_SECRET too, which
invalidates every existing dashboard login session (a deliberate side effect of a
password change, not a bug).

See src/webui/auth.py for the hashing/session-signing implementation, and M-8 in
docs/audits/DeltaQuant-Quant-Risk-Review.md for why this exists: the dashboard exposes wallet
balance, positions, and trading activity, and ENABLE_WEB_UI=true now fails closed at
startup (validate_configuration) without WEB_UI_PASSWORD_HASH and WEB_UI_SESSION_SECRET
both being set.
"""

import getpass
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.webui.auth import hash_password

MIN_PASSWORD_LENGTH = 8


def main() -> None:
    print("=" * 70)
    print(" ₹DeltaQuant dashboard login setup")
    print("=" * 70)

    password = getpass.getpass("New dashboard password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        print(
            f"\nRefusing a password shorter than {MIN_PASSWORD_LENGTH} characters — "
            "this account is reachable from the internet if ENABLE_WEB_UI is on.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("\nPasswords did not match — nothing generated.", file=sys.stderr)
        raise SystemExit(1)

    password_hash = hash_password(password)
    session_secret = secrets.token_urlsafe(48)

    print("\nPaste these into env/.env.common (replacing any existing values):\n")
    print(f"WEB_UI_PASSWORD_HASH={password_hash}")
    print(f"WEB_UI_SESSION_SECRET={session_secret}")
    print(
        "\nWEB_UI_USERNAME defaults to 'admin' — configure WEB_UI_USERNAME in "
        "env/.env.common if you want a different login name."
    )
    print(
        "\nNote: generating a new WEB_UI_SESSION_SECRET invalidates every existing "
        "dashboard login session. Restart the backend for these values to take effect."
    )


if __name__ == "__main__":
    main()
