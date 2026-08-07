"""Tests for the shared engine/session factory (src/db/base.py).

Covers a bug found while migrating to Postgres: get_session(database_url=...) used to
create a brand-new engine on every call, so a bare "sqlite:///:memory:" URL got a fresh,
empty database each time — nothing ever appeared to persist across two calls, even
within a single test/instance.
"""

from sqlalchemy import Column, Integer, String

from src.db.base import Base, get_engine, get_session


class _Probe(Base):
    """A throwaway table for exercising the session factory (registered on the shared Base)."""

    __tablename__ = "test_db_base_probe"

    id = Column(Integer, primary_key=True, autoincrement=True)
    value = Column(String(50))


def test_repeated_calls_with_same_override_url_share_state():
    database_url = "sqlite:///:memory:?cache=test_repeated_calls_with_same_override_url_share_state"

    session1 = get_session(database_url)
    session1.add(_Probe(value="hello"))
    session1.commit()
    session1.close()

    session2 = get_session(database_url)
    rows = session2.query(_Probe).all()
    session2.close()

    assert [r.value for r in rows] == ["hello"]


def test_different_override_urls_are_isolated_from_each_other():
    url_a = "sqlite:///:memory:?cache=test_different_override_urls_are_isolated_a"
    url_b = "sqlite:///:memory:?cache=test_different_override_urls_are_isolated_b"

    session_a = get_session(url_a)
    session_a.add(_Probe(value="only-in-a"))
    session_a.commit()
    session_a.close()

    session_b = get_session(url_b)
    rows_b = session_b.query(_Probe).all()
    session_b.close()

    assert rows_b == []


def test_get_engine_with_same_url_returns_the_same_cached_engine():
    database_url = "sqlite:///:memory:?cache=test_get_engine_with_same_url_returns_the_same_cached_engine"

    engine1 = get_engine(database_url)
    engine2 = get_engine(database_url)

    assert engine1 is engine2
