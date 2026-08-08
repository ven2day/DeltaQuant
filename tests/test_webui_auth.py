"""Tests for src/webui/auth.py: password hashing, session tokens, login rate limiting."""

from src.webui.auth import (
    LoginAttemptTracker,
    create_session_token,
    hash_password,
    verify_password,
    verify_session_token,
)


class TestPasswordHashing:
    def test_correct_password_verifies(self):
        stored = hash_password("correct-horse-battery-staple")
        assert verify_password("correct-horse-battery-staple", stored) is True

    def test_wrong_password_rejected(self):
        stored = hash_password("correct-horse-battery-staple")
        assert verify_password("wrong-password", stored) is False

    def test_empty_password_rejected(self):
        stored = hash_password("correct-horse-battery-staple")
        assert verify_password("", stored) is False

    def test_malformed_stored_hash_fails_closed(self):
        # Missing WEB_UI_PASSWORD_HASH / corrupted .env value must reject every
        # login, never raise into the route handler.
        assert verify_password("anything", "not-a-real-hash") is False
        assert verify_password("anything", "") is False
        assert verify_password("anything", "pbkdf2_sha256$notanumber$abc$def") is False

    def test_wrong_algorithm_tag_rejected(self):
        assert verify_password("x", "md5$1$abc$def") is False

    def test_salts_are_unique_per_call(self):
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2
        assert verify_password("same-password", h1) is True
        assert verify_password("same-password", h2) is True

    def test_hash_is_self_describing_format(self):
        h = hash_password("x", iterations=1000)
        algorithm, iterations, _salt, _digest = h.split("$", 3)
        assert algorithm == "pbkdf2_sha256"
        assert iterations == "1000"


class TestSessionTokens:
    def test_valid_token_round_trips(self):
        token = create_session_token("admin", secret="s3cr3t", ttl_minutes=30)
        assert verify_session_token(token, secret="s3cr3t") == "admin"

    def test_wrong_secret_rejected(self):
        token = create_session_token("admin", secret="s3cr3t", ttl_minutes=30)
        assert verify_session_token(token, secret="different-secret") is None

    def test_expired_token_rejected(self):
        token = create_session_token("admin", secret="s3cr3t", ttl_minutes=-1)
        assert verify_session_token(token, secret="s3cr3t") is None

    def test_tampered_payload_rejected(self):
        token = create_session_token("admin", secret="s3cr3t", ttl_minutes=30)
        payload_b64, sig_b64 = token.split(".", 1)
        flipped = ("A" if payload_b64[-1] != "A" else "B") + payload_b64[1:]
        tampered = f"{flipped}.{sig_b64}"
        assert verify_session_token(tampered, secret="s3cr3t") is None

    def test_garbage_token_rejected_not_raised(self):
        assert verify_session_token("garbage", secret="s3cr3t") is None
        assert verify_session_token("", secret="s3cr3t") is None
        assert verify_session_token("a.b.c", secret="s3cr3t") is None
        assert verify_session_token("not-base64!!.also-not", secret="s3cr3t") is None

    def test_cannot_forge_a_token_without_the_secret(self):
        # A forged token using a *guessed* secret must not verify against the real one.
        forged = create_session_token("admin", secret="guessed-wrong", ttl_minutes=30)
        assert verify_session_token(forged, secret="real-secret") is None


class TestLoginAttemptTracker:
    def test_not_locked_before_threshold(self):
        tracker = LoginAttemptTracker(max_attempts=3, lockout_minutes=15)
        tracker.record_failure("1.2.3.4")
        tracker.record_failure("1.2.3.4")
        assert tracker.is_locked("1.2.3.4") is False

    def test_locked_at_threshold(self):
        tracker = LoginAttemptTracker(max_attempts=3, lockout_minutes=15)
        for _ in range(3):
            tracker.record_failure("1.2.3.4")
        assert tracker.is_locked("1.2.3.4") is True

    def test_other_ips_unaffected(self):
        tracker = LoginAttemptTracker(max_attempts=1, lockout_minutes=15)
        tracker.record_failure("1.2.3.4")
        assert tracker.is_locked("1.2.3.4") is True
        assert tracker.is_locked("5.6.7.8") is False

    def test_reset_clears_lockout(self):
        tracker = LoginAttemptTracker(max_attempts=1, lockout_minutes=15)
        tracker.record_failure("1.2.3.4")
        assert tracker.is_locked("1.2.3.4") is True
        tracker.reset("1.2.3.4")
        assert tracker.is_locked("1.2.3.4") is False

    def test_old_failures_outside_window_are_pruned(self):
        tracker = LoginAttemptTracker(max_attempts=2, lockout_minutes=15)
        # Simulate a failure far in the past by manipulating the internal deque directly.
        import time

        tracker._failures["1.2.3.4"].append(time.time() - 20 * 60)  # 20 min ago, window=15
        tracker.record_failure("1.2.3.4")  # one recent failure
        # Only 1 failure inside the window -> not locked at max_attempts=2.
        assert tracker.is_locked("1.2.3.4") is False
