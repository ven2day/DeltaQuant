"""
Local Paper Trading Engine

Simulates a broker exchange locally for 100% free paper trading.
Maintains virtual wallet, positions, and order history.

Features:
- Virtual wallet with configurable starting balance
- Position tracking with P&L calculation
- Market and limit order simulation
- Persistent state via Postgres (no local/file fallback)
"""

import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.base import Base, get_session
from src.markets.nse.execution.runtime_mode import (
    RuntimeExecutionMode,
    assert_namespace_matches_mode,
)
from src.markets.nse.risk.costs import CostModel

logger = logging.getLogger(__name__)

# Positions/orders are re-synced to Postgres on every fill; the in-memory order list is
# capped on load purely for RAM sanity (Postgres itself keeps the unbounded history).
MAX_PERSISTED_ORDERS = 5000


@dataclass
class Position:
    """Represents an open position."""

    position_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    entry_price: float
    entry_time: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    entry_charges: float = 0.0  # charges paid on entry (for net-PnL attribution)
    # MAE/MFE tracking for trade quality analysis
    mae: float = 0.0  # Maximum Adverse Excursion (worst drawdown)
    mfe: float = 0.0  # Maximum Favorable Excursion (best unrealized profit)
    highest_price: float = 0.0
    lowest_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    strategy: str = ""
    # "real" (genuine DhanHQ quotes) or "simulated" (market closed, weekend/off-hours
    # pipeline test) -- set once at entry from market_manager.data_source and never
    # touched again, so a position opened on fake data stays honestly labeled even
    # after real trading resumes and data_source flips back.
    entry_data_source: str = "real"
    execution_mode: str = RuntimeExecutionMode.MARKET_PAPER.value
    market: str = "NSE"
    provider: str = "DHAN"
    broker_account_ref: str = ""

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price

    def update_pnl(self, current_price: float):
        """Update unrealized P&L and MAE/MFE based on current price."""
        self.current_price = current_price
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)

        if self.side == "BUY":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.mfe = max(self.mfe, (self.highest_price - self.entry_price) * self.quantity)
            self.mae = max(self.mae, (self.entry_price - self.lowest_price) * self.quantity)
        else:  # SELL (short)
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.mfe = max(self.mfe, (self.entry_price - self.lowest_price) * self.quantity)
            self.mae = max(self.mae, (self.highest_price - self.entry_price) * self.quantity)

        if self.entry_price > 0:
            self.unrealized_pnl_pct = (
                self.unrealized_pnl / (self.entry_price * self.quantity)
            ) * 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Order:
    """Represents an executed order."""

    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str  # "MARKET" or "LIMIT"
    price: float
    status: str  # "FILLED", "PENDING", "CANCELLED"
    timestamp: str
    # Set on exit orders to the ExitManager rule that triggered them (e.g. "stop_loss",
    # "target_hit", "trailing_stop", "time_exit", "regime_change"); empty for entries.
    reason: str = ""
    trade_id: str = ""
    position_id: str = ""
    idempotency_key: str = ""
    realized_pnl: float = 0.0
    entry_charges: float = 0.0
    exit_charges: float = 0.0
    execution_mode: str = RuntimeExecutionMode.MARKET_PAPER.value
    market: str = "NSE"
    provider: str = "DHAN"
    broker_account_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperWalletRecord(Base):
    """Singleton row (id=1) holding the wallet's cash/aggregate stats."""

    __tablename__ = "paper_wallet"

    id = Column(Integer, primary_key=True)
    namespace = Column(String(40), nullable=False, unique=True, index=True)
    balance = Column(Float, nullable=False)
    initial_balance = Column(Float, nullable=False)
    realized_pnl = Column(Float, default=0.0, nullable=False)
    total_trades = Column(Integer, default=0, nullable=False)
    winning_trades = Column(Integer, default=0, nullable=False)
    losing_trades = Column(Integer, default=0, nullable=False)


class PaperPositionRecord(Base):
    """One row per currently open position.

    Structural/entry fields are persisted on every fill. ``current_price`` is the one
    mark-to-market field also persisted -- not on every price tick (that would be pure
    write overhead for a value update_positions_pnl() recomputes from live prices
    anyway), but opportunistically: on every trade-triggered resync, and via a
    throttled periodic snapshot (see LocalPaperEngine.snapshot_current_prices()) so a
    restart during closed-market hours (no live quotes available yet -- see
    MarketDataManager.start()'s documented fail-closed behavior) shows the last known
    mark instead of resetting the display to entry price / zero P&L. Nullable so
    existing rows from before this column existed load safely (falls back to
    entry_price, today's exact prior behavior).
    """

    __tablename__ = "paper_positions"

    position_id = Column(String(40), primary_key=True)
    namespace = Column(String(40), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    market = Column(String(16), nullable=False, default="NSE", index=True)
    provider = Column(String(24), nullable=False, default="DHAN", index=True)
    broker_account_ref = Column(String(128), nullable=False, default="")
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(String(40), nullable=False)
    entry_charges = Column(Float, default=0.0, nullable=False)
    stop_loss = Column(Float, default=0.0, nullable=False)
    target_price = Column(Float, default=0.0, nullable=False)
    strategy = Column(String(50), default="", nullable=False)
    entry_data_source = Column(String(20), default="real", nullable=False)
    current_price = Column(Float, nullable=True)


class PaperOrderRecord(Base):
    """One row per order ever placed (unbounded — Postgres, unlike a JSON file, can hold this)."""

    __tablename__ = "paper_orders"

    order_id = Column(String(60), primary_key=True)
    namespace = Column(String(40), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    market = Column(String(16), nullable=False, default="NSE", index=True)
    provider = Column(String(24), nullable=False, default="DHAN", index=True)
    broker_account_ref = Column(String(128), nullable=False, default="")
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    order_type = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)
    timestamp = Column(String(40), nullable=False, index=True)
    reason = Column(String(20), nullable=True, default="")
    trade_id = Column(String(60), nullable=True, default="", index=True)
    position_id = Column(String(60), nullable=True, default="", index=True)
    idempotency_key = Column(String(200), nullable=True, unique=True, index=True)
    realized_pnl = Column(Float, default=0.0, nullable=False)


class LocalPaperEngine:
    """
    Local paper trading engine that simulates a broker.

    Provides:
    - Virtual wallet management
    - Position tracking
    - Order execution (market orders filled at current price)
    - P&L calculation
    - Persistent state (Postgres)
    """

    def __init__(
        self,
        initial_balance: float | None = None,
        database_url: str | None = None,
        cost_model: CostModel | None = None,
        execution_mode: RuntimeExecutionMode | str = RuntimeExecutionMode.MARKET_PAPER,
        namespace: str | None = None,
    ):
        """
        Initialize the paper trading engine.

        Args:
            initial_balance: Starting balance in INR (default from config)
            database_url: Override DB URL (e.g. sqlite:///:memory: in tests); defaults to
                settings.database_url
            cost_model: Slippage/fee model (defaults to CostModel.from_settings();
                pass CostModel.zero() for ideal fills)
        """
        settings = get_settings()
        self.execution_mode = RuntimeExecutionMode.parse(execution_mode)
        self.namespace = namespace or self.execution_mode.namespace
        assert_namespace_matches_mode(self.execution_mode, self.namespace)
        self.initial_balance = initial_balance or settings.paper_wallet_balance
        self._database_url = database_url
        self.long_only = settings.long_only
        self.cost_model = cost_model or CostModel.from_settings(settings)

        self.balance = self.initial_balance
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Load existing state if available
        self._load_state()

    def _session(self) -> Session:
        return get_session(self._database_url)

    def _load_state(self) -> None:
        """Load wallet/positions/recent orders from Postgres, if any exist yet."""
        session = self._session()
        try:
            wallet = session.query(PaperWalletRecord).filter_by(namespace=self.namespace).first()
            if wallet is None:
                return  # fresh wallet — defaults set in __init__ already apply

            self.balance = wallet.balance
            self.initial_balance = wallet.initial_balance
            self.realized_pnl = wallet.realized_pnl
            self.total_trades = wallet.total_trades
            self.winning_trades = wallet.winning_trades
            self.losing_trades = wallet.losing_trades

            for row in session.query(PaperPositionRecord).filter_by(namespace=self.namespace).all():
                position = Position(
                    position_id=row.position_id,
                    symbol=row.symbol,
                    side=row.side,
                    quantity=row.quantity,
                    entry_price=row.entry_price,
                    entry_time=row.entry_time,
                    current_price=row.entry_price,  # update_positions_pnl() refreshes this
                    entry_charges=row.entry_charges,
                    stop_loss=row.stop_loss,
                    target_price=row.target_price,
                    strategy=row.strategy,
                    entry_data_source=row.entry_data_source or "real",
                    execution_mode=self.execution_mode.value,
                    market=str(getattr(row, "market", "NSE") or "NSE"),
                    provider=str(getattr(row, "provider", "DHAN") or "DHAN"),
                    broker_account_ref=str(getattr(row, "broker_account_ref", "") or ""),
                )
                last_known_price = getattr(row, "current_price", None)
                if last_known_price:
                    # Seed the display with the last price seen before this restart
                    # (NULL for rows persisted before this column existed, or a
                    # position opened and never snapshotted -- falls back to entry
                    # price/zero P&L exactly as before). update_positions_pnl()
                    # overwrites this the moment a real live quote arrives.
                    position.update_pnl(float(last_known_price))
                self.positions[row.position_id] = position

            recent_orders = (
                session.query(PaperOrderRecord)
                .filter_by(namespace=self.namespace)
                .order_by(PaperOrderRecord.timestamp.desc())
                .limit(MAX_PERSISTED_ORDERS)
                .all()
            )
            for row in reversed(recent_orders):
                self.orders.append(
                    Order(
                        order_id=row.order_id,
                        symbol=row.symbol,
                        side=row.side,
                        quantity=row.quantity,
                        order_type=row.order_type,
                        price=row.price,
                        status=row.status,
                        timestamp=row.timestamp,
                        reason=row.reason or "",
                        trade_id=row.trade_id or "",
                        position_id=row.position_id or "",
                        idempotency_key=row.idempotency_key or "",
                        realized_pnl=float(row.realized_pnl or 0.0),
                        execution_mode=self.execution_mode.value,
                        market=str(getattr(row, "market", "NSE") or "NSE"),
                        provider=str(getattr(row, "provider", "DHAN") or "DHAN"),
                        broker_account_ref=str(
                            getattr(row, "broker_account_ref", "") or ""
                        ),
                    )
                )

            logger.info(
                f"Loaded paper wallet from Postgres: Rs.{self.balance:,.2f} balance, "
                f"{len(self.positions)} positions"
            )
        except Exception:
            logger.exception("Failed to load paper wallet state from Postgres; starting fresh.")
        finally:
            session.close()

    def _persist_after_trade(self, new_order: Order) -> bool:
        """Write the wallet row, resync open positions, and append the new order."""
        session = self._session()
        try:
            wallet = session.query(PaperWalletRecord).filter_by(namespace=self.namespace).first()
            if wallet is None:
                wallet = PaperWalletRecord(
                    id=1 if self.execution_mode is RuntimeExecutionMode.MARKET_PAPER else 2,
                    namespace=self.namespace,
                    balance=self.balance,
                    initial_balance=self.initial_balance,
                )
                session.add(wallet)
            wallet.balance = self.balance
            wallet.initial_balance = self.initial_balance
            wallet.realized_pnl = self.realized_pnl
            wallet.total_trades = self.total_trades
            wallet.winning_trades = self.winning_trades
            wallet.losing_trades = self.losing_trades

            # Open-position count is always small (a handful at most) — a full resync on
            # every trade is simpler and cheaper than diffing, and can't drift out of sync.
            session.query(PaperPositionRecord).filter_by(namespace=self.namespace).delete()
            for pos in self.positions.values():
                session.add(
                    PaperPositionRecord(
                        namespace=self.namespace,
                        position_id=pos.position_id,
                        symbol=pos.symbol,
                        side=pos.side,
                        quantity=pos.quantity,
                        entry_price=pos.entry_price,
                        entry_time=pos.entry_time,
                        entry_charges=pos.entry_charges,
                        stop_loss=pos.stop_loss,
                        target_price=pos.target_price,
                        strategy=pos.strategy,
                        entry_data_source=pos.entry_data_source,
                        market=pos.market,
                        provider=pos.provider,
                        broker_account_ref=pos.broker_account_ref,
                        current_price=pos.current_price,
                    )
                )

            session.add(
                PaperOrderRecord(
                    namespace=self.namespace,
                    order_id=new_order.order_id,
                    symbol=new_order.symbol,
                    side=new_order.side,
                    quantity=new_order.quantity,
                    order_type=new_order.order_type,
                    price=new_order.price,
                    status=new_order.status,
                    timestamp=new_order.timestamp,
                    reason=new_order.reason,
                    trade_id=new_order.trade_id,
                    position_id=new_order.position_id,
                    idempotency_key=new_order.idempotency_key or None,
                    realized_pnl=new_order.realized_pnl,
                    market=new_order.market,
                    provider=new_order.provider,
                    broker_account_ref=new_order.broker_account_ref,
                )
            )
            session.commit()
            logger.debug("Paper wallet state saved to Postgres")
            return True
        except Exception:
            logger.exception("Failed to persist paper wallet state to Postgres")
            session.rollback()
            return False
        finally:
            session.close()

    def snapshot_current_prices(self) -> None:
        """Persist each open position's current mark, independent of any trade.

        Positions can sit open for a long time between fills with only their price
        moving -- _persist_after_trade() alone would leave current_price stale at
        whatever it was on the last trade. The caller (the live loop's fast
        position-management tick) throttles how often this runs; a handful of
        single-row UPDATEs is cheap regardless (see _persist_after_trade's own
        "open-position count is always small" note). Best-effort: a failure here
        only affects what a future restart displays before fresh quotes arrive,
        never the live in-memory P&L this process is already showing.
        """
        if not self.positions:
            return
        session = self._session()
        try:
            for pos in self.positions.values():
                session.query(PaperPositionRecord).filter_by(
                    namespace=self.namespace, position_id=pos.position_id
                ).update({"current_price": pos.current_price})
            session.commit()
        except Exception:
            logger.debug("Periodic current_price snapshot skipped", exc_info=True)
            session.rollback()
        finally:
            session.close()

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        order_type: str = "MARKET",
        reason: str = "",
        *,
        trade_id: str = "",
        position_id: str = "",
        idempotency_key: str = "",
        stop_loss: float = 0.0,
        target_price: float = 0.0,
        strategy: str = "",
        entry_data_source: str = "real",
    ) -> Order:
        """
        Place an order (immediately filled for market orders).

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares
            current_price: Current market price
            order_type: "MARKET" or "LIMIT"
            reason: For exits, the ExitManager rule that triggered this order (e.g.
                "stop_loss", "target_hit") — persisted so the Trade History UI can show
                why a position closed, not just that it did. Empty for entries.

        Returns:
            Order object with fill details
        """
        if entry_data_source != self.execution_mode.expected_quote_source:
            return Order(
                order_id=f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}",
                symbol=symbol,
                side=side.upper(),
                quantity=0,
                order_type=order_type,
                price=max(0.0, current_price),
                status="REJECTED",
                timestamp=datetime.now().isoformat(),
                reason="execution_mode_quote_source_mismatch",
                trade_id=trade_id,
                position_id=position_id or trade_id,
                idempotency_key=idempotency_key,
                execution_mode=self.execution_mode.value,
            )

        if idempotency_key:
            session = self._session()
            try:
                prior = (
                    session.query(PaperOrderRecord)
                    .filter_by(
                        idempotency_key=idempotency_key,
                        namespace=self.namespace,
                    )
                    .first()
                )
                if prior is not None:
                    return Order(
                        order_id=str(prior.order_id),
                        symbol=str(prior.symbol),
                        side=str(prior.side),
                        quantity=int(prior.quantity),
                        order_type=str(prior.order_type),
                        price=float(prior.price),
                        status="DUPLICATE",
                        timestamp=str(prior.timestamp),
                        reason=str(prior.reason or ""),
                        trade_id=str(prior.trade_id or ""),
                        position_id=str(prior.position_id or ""),
                        idempotency_key=idempotency_key,
                        realized_pnl=float(prior.realized_pnl or 0.0),
                        execution_mode=self.execution_mode.value,
                    )
            finally:
                session.close()

        order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}"
        position_id = position_id or trade_id

        if quantity <= 0 or current_price <= 0:
            return Order(
                order_id=order_id,
                symbol=symbol,
                side=side.upper(),
                quantity=0,
                order_type=order_type,
                price=max(0.0, current_price),
                status="REJECTED",
                timestamp=datetime.now().isoformat(),
                reason="invalid_quantity_or_price",
                trade_id=trade_id,
                position_id=position_id,
                idempotency_key=idempotency_key,
                execution_mode=self.execution_mode.value,
            )

        # Limit orders are not simulated yet — return PENDING unchanged.
        if order_type != "MARKET":
            return Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=current_price,
                status="PENDING",
                timestamp=datetime.now().isoformat(),
                reason=reason,
                trade_id=trade_id,
                position_id=position_id,
                idempotency_key=idempotency_key,
                execution_mode=self.execution_mode.value,
            )

        side = side.upper()
        # Apply adverse slippage to get the actual fill price.
        fill_price = self.cost_model.fill_price(current_price, side)

        def _rejected() -> Order:
            return Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=fill_price,
                status="REJECTED",
                timestamp=datetime.now().isoformat(),
                reason=reason,
                trade_id=trade_id,
                position_id=position_id,
                idempotency_key=idempotency_key,
                execution_mode=self.execution_mode.value,
            )

        # In long-only cash-investing mode a SELL may only close existing long holdings.
        # Reject before changing any state so an oversell can never leave a partial close
        # or accidentally create a short position.
        if self.long_only and side == "SELL":
            available = sum(
                position.quantity
                for position in self.positions.values()
                if position.symbol == symbol
                and position.side == "BUY"
                and (not position_id or position.position_id == position_id)
            )
            if quantity > available:
                logger.warning(
                    "Rejected long-only sell for %s: requested %d shares, holding %d",
                    symbol,
                    quantity,
                    available,
                )
                return _rejected()

        snapshot = {
            "balance": self.balance,
            "positions": deepcopy(self.positions),
            "orders": list(self.orders),
            "realized_pnl": self.realized_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
        }

        # An order in the opposite direction closes/covers existing positions first.
        opposite = "SELL" if side == "BUY" else "BUY"
        remaining = quantity
        closed_any = False
        order_realized_pnl = 0.0
        order_entry_charges = 0.0
        order_exit_charges = 0.0
        affected_position_id = position_id
        while remaining > 0:
            matching = next(
                (
                    position
                    for position in self.positions.values()
                    if position.symbol == symbol
                    and position.side == opposite
                    and (not position_id or position.position_id == position_id)
                ),
                None,
            )
            if matching is None:
                break

            close_qty = min(remaining, matching.quantity)
            close_notional = fill_price * close_qty
            close_charges = self.cost_model.charges(close_notional, side)
            if matching.side == "BUY":  # selling to close a long
                gross = (fill_price - matching.entry_price) * close_qty
            else:  # buying to cover a short
                gross = (matching.entry_price - fill_price) * close_qty
            entry_charges_prorata = matching.entry_charges * (close_qty / matching.quantity)
            net_pnl = gross - entry_charges_prorata - close_charges
            order_realized_pnl += net_pnl
            order_entry_charges += entry_charges_prorata
            order_exit_charges += close_charges
            affected_position_id = matching.position_id

            # Release the principal committed on entry, add the gross gain, pay exit charges.
            self.balance += matching.entry_price * close_qty + gross - close_charges
            self.realized_pnl += net_pnl
            self.total_trades += 1
            if net_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            if close_qty >= matching.quantity:
                del self.positions[matching.position_id]
            else:
                matching.quantity -= close_qty
                matching.entry_charges -= entry_charges_prorata

            logger.info(
                f"{side} filled (close): {close_qty} {symbol} @ ₹{fill_price:,.2f} | "
                f"net P&L: ₹{net_pnl:+,.2f}"
            )
            remaining -= close_qty
            closed_any = True

        if remaining > 0:
            # Long-only mode never opens a short. The pre-check above should make this
            # unreachable for SELLs, but keep this guard as the final execution boundary.
            if self.long_only and side == "SELL":
                return _rejected()

            # Opening a new position (long for BUY, short for SELL when explicitly enabled).
            open_notional = fill_price * remaining
            open_charges = self.cost_model.charges(open_notional, side)
            required = open_notional + open_charges
            if required > self.balance:
                logger.warning(
                    f"Insufficient balance for {symbol}: need ₹{required:,.2f}, "
                    f"have ₹{self.balance:,.2f}"
                )
                # If we already closed part of an opposite position above, that stands;
                # only the new-open remainder is rejected.
                if not closed_any:
                    return _rejected()
            else:
                self.balance -= required
                opened = self._open_position(
                    symbol,
                    side,
                    remaining,
                    fill_price,
                    open_charges,
                    position_id=position_id,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    strategy=strategy,
                    entry_data_source=entry_data_source,
                )
                affected_position_id = opened.position_id
                order_entry_charges += open_charges
                logger.info(
                    f"{side} filled (open): {remaining} {symbol} @ ₹{fill_price:,.2f} | "
                    f"Balance: ₹{self.balance:,.2f}"
                )
                remaining = 0

        # remaining > 0 here means a close leg succeeded (closed_any) but the leftover
        # open leg was skipped for insufficient balance — report what actually filled,
        # not the originally requested quantity, and mark it PARTIALLY_FILLED rather
        # than lying that the whole order completed.
        filled_quantity = quantity - remaining
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=filled_quantity,
            order_type=order_type,
            price=fill_price,
            status="FILLED" if remaining == 0 else "PARTIALLY_FILLED",
            timestamp=datetime.now().isoformat(),
            reason=reason,
            trade_id=trade_id,
            position_id=affected_position_id,
            idempotency_key=idempotency_key,
            realized_pnl=order_realized_pnl,
            entry_charges=order_entry_charges,
            exit_charges=order_exit_charges,
            execution_mode=self.execution_mode.value,
        )
        self.orders.append(order)
        if not self._persist_after_trade(order):
            self.balance = float(snapshot["balance"])
            self.positions = snapshot["positions"]
            self.orders = snapshot["orders"]
            self.realized_pnl = float(snapshot["realized_pnl"])
            self.total_trades = int(snapshot["total_trades"])
            self.winning_trades = int(snapshot["winning_trades"])
            self.losing_trades = int(snapshot["losing_trades"])
            order.status = "REJECTED"
            order.reason = "persistence_error"
        return order

    def _open_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        charges: float,
        *,
        position_id: str = "",
        stop_loss: float = 0.0,
        target_price: float = 0.0,
        strategy: str = "",
        entry_data_source: str = "real",
    ) -> Position:
        """Create and register a new open position (long or short)."""
        resolved_id = position_id or f"POS-{uuid4().hex[:8]}"
        existing = self.positions.get(resolved_id)
        if existing is not None:
            if existing.symbol != symbol or existing.side != side:
                raise ValueError("position identity cannot be reused for another symbol or side")
            new_quantity = existing.quantity + quantity
            existing.entry_price = (
                existing.entry_price * existing.quantity + fill_price * quantity
            ) / new_quantity
            existing.quantity = new_quantity
            existing.entry_charges += charges
            existing.current_price = fill_price
            existing.stop_loss = stop_loss or existing.stop_loss
            existing.target_price = target_price or existing.target_price
            existing.strategy = strategy or existing.strategy
            return existing

        position = Position(
            position_id=resolved_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=datetime.now().isoformat(),
            current_price=fill_price,
            entry_charges=charges,
            stop_loss=stop_loss,
            target_price=target_price,
            strategy=strategy,
            entry_data_source=entry_data_source,
            execution_mode=self.execution_mode.value,
        )
        self.positions[position.position_id] = position
        return position

    def update_positions_pnl(self, market_prices: dict[str, float]):
        """
        Update unrealized P&L for all positions based on current prices.

        Args:
            market_prices: Dictionary of symbol -> current price
        """
        for position in self.positions.values():
            if position.symbol in market_prices:
                position.update_pnl(market_prices[position.symbol])

    def get_balance(self) -> float:
        """Get current cash balance."""
        return self.balance

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_total_value(self) -> float:
        """
        Total portfolio value = cash + committed capital + unrealized P&L.

        Uses committed capital (entry notional + entry charges) plus unrealized P&L rather
        than naive market value, so short positions are valued correctly and opening a
        position doesn't change net worth.
        """
        positions_value = sum(
            p.entry_price * p.quantity + p.entry_charges + p.unrealized_pnl
            for p in self.positions.values()
        )
        return self.balance + positions_value

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_win_rate(self) -> float:
        """Get win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive trading statistics."""
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "total_value": self.get_total_value(),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.get_unrealized_pnl(),
            "total_pnl": self.realized_pnl + self.get_unrealized_pnl(),
            "return_pct": ((self.get_total_value() - self.initial_balance) / self.initial_balance)
            * 100,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.get_win_rate(),
            "open_positions": len(self.positions),
        }

    def get_orders_for_trade(self, trade_id: str) -> list[Order]:
        """All durable orders tagged with this trade_id (entry + adds + exits), oldest first.

        Ground truth for lifecycle-ledger reconciliation after a crash: this table is
        written synchronously inside ``place_order``'s own transaction, so it can never be
        behind the separate lifecycle ledger the way the ledger can be behind it after a
        crash between "order filled here" and "ledger updated there". Charge fields are not
        persisted per-order (see ``get_closed_trade_history``'s note on the same tradeoff),
        so callers needing charge granularity should treat 0.0 as "unknown, not zero" —
        ``realized_pnl`` is already net-of-charges and unaffected.
        """
        if not trade_id:
            return []
        session = self._session()
        try:
            rows = (
                session.query(PaperOrderRecord)
                .filter_by(trade_id=trade_id, namespace=self.namespace)
                .order_by(PaperOrderRecord.timestamp)
                .all()
            )
        finally:
            session.close()
        return [
            Order(
                order_id=str(row.order_id),
                symbol=str(row.symbol),
                side=str(row.side),
                quantity=int(row.quantity),
                order_type=str(row.order_type),
                price=float(row.price),
                status=str(row.status),
                timestamp=str(row.timestamp),
                reason=str(row.reason or ""),
                trade_id=str(row.trade_id or ""),
                position_id=str(row.position_id or ""),
                idempotency_key=str(row.idempotency_key or ""),
                realized_pnl=float(row.realized_pnl or 0.0),
                execution_mode=self.execution_mode.value,
            )
            for row in rows
        ]

    def get_closed_trade_history(self, limit: int = 250) -> list[dict[str, Any]]:
        """Reconstruct closed paper fills from the durable order ledger.

        Older sessions predate a dedicated close-fill table, but every filled order is
        durable. Replaying those fills with the same matching and cost rules used by
        :meth:`place_order` yields an auditable net-P&L ledger across restarts.
        """
        session = self._session()
        try:
            orders = (
                session.query(PaperOrderRecord)
                .filter_by(namespace=self.namespace)
                .order_by(PaperOrderRecord.timestamp)
                .all()
            )
        finally:
            session.close()

        open_positions: list[dict[str, Any]] = []
        closed: list[dict[str, Any]] = []
        for order in orders:
            # A PARTIALLY_FILLED order still has a real, executed close leg (the open
            # leg was what got skipped) — exclude only orders with no fill at all.
            if order.status not in ("FILLED", "PARTIALLY_FILLED"):
                continue

            side = order.side.upper()
            opposite = "SELL" if side == "BUY" else "BUY"
            matching = next(
                (
                    position
                    for position in open_positions
                    if position["symbol"] == order.symbol and position["side"] == opposite
                ),
                None,
            )
            remaining = order.quantity
            if matching is not None:
                quantity = min(remaining, matching["quantity"])
                exit_charges = self.cost_model.charges(order.price * quantity, side)
                entry_charges = matching["entry_charges"] * (quantity / matching["quantity"])
                gross_pnl = (
                    (order.price - matching["entry_price"]) * quantity
                    if matching["side"] == "BUY"
                    else (matching["entry_price"] - order.price) * quantity
                )
                closed.append(
                    {
                        "close_order_id": order.order_id,
                        "timestamp": order.timestamp,
                        "symbol": order.symbol,
                        "side": matching["side"],
                        "quantity": quantity,
                        "entry_price": matching["entry_price"],
                        "entry_time": matching["entry_time"],
                        "exit_price": order.price,
                        "exit_reason": order.reason or "",
                        "gross_pnl": gross_pnl,
                        "entry_charges": entry_charges,
                        "exit_charges": exit_charges,
                        "net_pnl": gross_pnl - entry_charges - exit_charges,
                    }
                )
                if quantity >= matching["quantity"]:
                    open_positions.remove(matching)
                else:
                    matching["quantity"] -= quantity
                    matching["entry_charges"] -= entry_charges
                remaining -= quantity

            if remaining > 0:
                open_positions.append(
                    {
                        "symbol": order.symbol,
                        "side": side,
                        "quantity": remaining,
                        "entry_price": order.price,
                        "entry_time": order.timestamp,
                        "entry_charges": self.cost_model.charges(order.price * remaining, side),
                    }
                )

        return list(reversed(closed[-max(1, limit) :]))

    def reset(self) -> None:
        """Reset the paper wallet to initial state."""
        self.balance = self.initial_balance
        self.positions = {}
        self.orders = []
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        session = self._session()
        try:
            session.query(PaperPositionRecord).filter_by(namespace=self.namespace).delete()
            session.query(PaperOrderRecord).filter_by(namespace=self.namespace).delete()
            session.query(PaperWalletRecord).filter_by(namespace=self.namespace).delete()
            session.commit()
        except Exception:
            logger.exception("Failed to reset paper wallet in Postgres")
            session.rollback()
        finally:
            session.close()

        logger.info(f"Paper wallet reset to ₹{self.initial_balance:,.2f}")


def test_paper_engine():
    """Test the paper trading engine."""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("[PAPER] ₹DeltaQuant - Local Paper Engine Test")
    print("=" * 60)

    # Isolated in-memory DB — never touch the real configured Postgres for a demo run.
    engine = LocalPaperEngine(initial_balance=1000000, database_url="sqlite:///:memory:")

    print(f"\n[INIT] Starting balance: ₹{engine.get_balance():,.2f}")

    # Simulate buying RELIANCE
    print("\n[ORDER] Placing BUY order for RELIANCE...")
    order1 = engine.place_order(
        symbol="RELIANCE",
        side="BUY",
        quantity=10,
        current_price=2500.0,
    )
    print(f"  Order: {order1.status} @ ₹{order1.price:,.2f}")
    print(f"  Balance: ₹{engine.get_balance():,.2f}")

    # Simulate price going up
    print("\n[UPDATE] Price increased to ₹2550...")
    engine.update_positions_pnl({"RELIANCE": 2550.0})

    positions = engine.get_positions()
    for pos in positions:
        print(f"  Position: {pos.quantity} {pos.symbol} @ ₹{pos.entry_price:,.2f}")
        print(f"  Unrealized P&L: ₹{pos.unrealized_pnl:+,.2f} ({pos.unrealized_pnl_pct:+.2f}%)")

    # Sell to realize profit
    print("\n[ORDER] Placing SELL order to close position...")
    order2 = engine.place_order(
        symbol="RELIANCE",
        side="SELL",
        quantity=10,
        current_price=2550.0,
    )
    print(f"  Order: {order2.status} @ ₹{order2.price:,.2f}")

    # Show stats
    print("\n[STATS] Trading Statistics:")
    stats = engine.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(
                f"  {key}: ₹{value:,.2f}"
                if "pnl" in key or "balance" in key or "value" in key
                else f"  {key}: {value:.2f}%"
            )
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Paper trading engine working!")
    print("=" * 60)


if __name__ == "__main__":
    test_paper_engine()
