"""Tests for src/markets/nse/broker/dhan/instruments.py — DhanHQ instrument-master resolution."""

from unittest.mock import MagicMock, patch

import src.markets.nse.broker.dhan.instruments as dhan_instruments
from src.markets.nse.broker.dhan.instruments import FALLBACK_SECURITY_IDS, fetch_security_id_map

_SAMPLE_CSV = (
    "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,SEM_EXPIRY_CODE,"
    "SEM_TRADING_SYMBOL,SEM_LOT_UNITS,SEM_CUSTOM_SYMBOL,SEM_EXPIRY_DATE,SEM_STRIKE_PRICE,"
    "SEM_OPTION_TYPE,SEM_TICK_SIZE,SEM_EXPIRY_FLAG,SEM_EXCH_INSTRUMENT_TYPE,SEM_SERIES,"
    "SM_SYMBOL_NAME\n"
    # A plain NSE cash-equity row — should be picked up.
    "NSE,E,2885,EQUITY,0,RELIANCE,1.0,Reliance Industries,,,,10.0000,NA,ES,EQ,"
    "RELIANCE INDUSTRIES LTD\n"
    # Same security-ID space but a currency derivative — must be excluded.
    "NSE,C,2885,OPTCUR,0,EURINR-Aug2025-102.75-CE,1.0,x,2025-08-01,102.75,CE,0.25,W,CUR OP,,\n"
    # BSE listing of the same company — wrong exchange, must be excluded.
    "BSE,E,500325,EQUITY,0,RELIANCE,1.0,x,,,,5.0000,NA,ES,A,RELIANCE INDUSTRIES LTD.\n"
    # SME-series NSE equity — must be excluded (not a regular EQ listing).
    "NSE,E,1,EQUITY,0,GOLDSTAR,11250.0,x,,,,5.0000,NA,ES,SM,GOLDSTAR POWER LIMITED\n"
    # A symbol with special characters, to prove those aren't mangled.
    "NSE,E,2031,EQUITY,0,M&M,1.0,x,,,,5.0000,NA,ES,EQ,MAHINDRA AND MAHINDRA LTD\n"
)


def _reset_cache():
    dhan_instruments._cached_map = None


def _mock_response(text: str):
    resp = MagicMock()
    resp.text = text
    return resp


def test_fetch_security_id_map_filters_to_nse_cash_equities():
    _reset_cache()
    with patch(
        "src.markets.nse.broker.dhan.instruments.requests.get", return_value=_mock_response(_SAMPLE_CSV)
    ):
        result = fetch_security_id_map()

    assert result == {"RELIANCE": "2885", "M&M": "2031"}


def test_fetch_security_id_map_filters_to_requested_symbols():
    _reset_cache()
    with patch(
        "src.markets.nse.broker.dhan.instruments.requests.get", return_value=_mock_response(_SAMPLE_CSV)
    ):
        result = fetch_security_id_map(["RELIANCE", "NOTPRESENT"])

    assert result == {"RELIANCE": "2885"}


def test_fetch_security_id_map_caches_across_calls():
    _reset_cache()
    with patch(
        "src.markets.nse.broker.dhan.instruments.requests.get", return_value=_mock_response(_SAMPLE_CSV)
    ) as mock_get:
        fetch_security_id_map()
        fetch_security_id_map()

    assert mock_get.call_count == 1


def test_fetch_security_id_map_falls_back_on_network_failure():
    _reset_cache()
    with patch("src.markets.nse.broker.dhan.instruments.requests.get", side_effect=Exception("no network")):
        result = fetch_security_id_map(["RELIANCE"])

    assert result == {"RELIANCE": FALLBACK_SECURITY_IDS["RELIANCE"]}


def test_fetch_security_id_map_falls_back_on_empty_result():
    _reset_cache()
    empty_csv = "SEM_EXM_EXCH_ID,SEM_SEGMENT\nBSE,C\n"
    with patch("src.markets.nse.broker.dhan.instruments.requests.get", return_value=_mock_response(empty_csv)):
        result = fetch_security_id_map(["RELIANCE"])

    assert result == {"RELIANCE": FALLBACK_SECURITY_IDS["RELIANCE"]}
