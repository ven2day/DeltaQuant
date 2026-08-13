"""Canonical durable lifecycle for permanent simulated-paper trades.

The wallet remains the cash/position accounting engine. This ledger is the authoritative
cross-component identity and audit trail linking the signal decision, exact approved
levels/size, paper fills, managed exits, journal, performance, and learning context.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Session

from src.db.base import Base, get_engine, get_session

logger = logging.getLogger(__name__)

MOCK_NAMESPACE = "mock_simulated"
PAPER_DATA_NAMESPACE = "paper_market_data"


class PaperTradeLifecycleRecord(Base):
    __tablename__ = "paper_trade_lifecycles"

    trade_id = Column(String(60), primary_key=True)
    namespace = Column(String(40), nullable=False, index=True)
    run_id = Column(String(80), nullable=False, index=True)
    workflow_id = Column(String(80), nullable=False, index=True)
    idempotency_key = Column(String(200), nullable=False, unique=True, index=True)
    signal_id = Column(String(80), default="", nullable=False)
    position_id = Column(String(60), default="", nullable=False, index=True)
    entry_order_id = Column(String(60), default="", nullable=False)
    symbol = Column(String(30), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    strategy = Column(String(60), default="", nullable=False, index=True)
    timeframe = Column(String(20), default="", nullable=False)
    status = Column(String(30), nullable=False, index=True)
    original_quantity = Column(Integer, nullable=False)
    remaining_quantity = Column(Integer, nullable=False)
    add_count = Column(Integer, default=0, nullable=False)
    entry_price = Column(Float, nullable=False)
    weighted_entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    cumulative_pnl = Column(Float, default=0.0, nullable=False)
    entry_charges = Column(Float, default=0.0, nullable=False)
    exit_charges = Column(Float, default=0.0, nullable=False)
    # Leg-level worst/best excursion, accumulated durably across partial exits so a
    # crash-recovery replay (which cannot see the exit manager's in-memory ManagedPosition)
    # can still recover it for the final performance/learning outcome.
    mae = Column(Float, default=0.0, nullable=False)
    mfe = Column(Float, default=0.0, nullable=False)
    rationale_json = Column(Text, default="[]", nullable=False)
    context_json = Column(Text, default="{}", nullable=False)
    active_lessons_json = Column(Text, default="[]", nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    # Set only once the *downstream* finalization (journal close + performance record +
    # lesson crediting) has completed for a CLOSED trade. This is the durable outbox
    # checkpoint: status=="CLOSED" with finalized_at NULL means the wallet/exit-manager/
    # ledger side of the close is durable, but the journal/performance/learning fan-out is
    # still pending — startup replay (see get_unfinalized_closed) must (re)drive it rather
    # than silently losing it, which was exactly the C-2/C-5 defect.
    finalized_at = Column(DateTime(timezone=True), nullable=True)


class PaperTradeEventRecord(Base):
    __tablename__ = "paper_trade_events"

    event_id = Column(String(60), primary_key=True)
    trade_id = Column(String(60), nullable=False, index=True)
    namespace = Column(String(40), nullable=False, index=True)
    event_type = Column(String(30), nullable=False, index=True)
    order_id = Column(String(60), default="", nullable=False)
    quantity = Column(Integer, default=0, nullable=False)
    price = Column(Float, default=0.0, nullable=False)
    realized_pnl = Column(Float, default=0.0, nullable=False)
    reason = Column(String(60), default="", nullable=False)
    metadata_json = Column(Text, default="{}", nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)


@dataclass(frozen=True)
class ReconciliationReport:
    restored_managers: int = 0
    imported_wallet_positions: int = 0
    lifecycle_without_wallet: int = 0
    removed_ghost_managers: int = 0
    # A wallet position existed (entry fill committed) but its lifecycle record was still
    # INTENT — the process crashed between the paper-engine fill and lifecycle_store.mark_open.
    # Repaired in place using the wallet's own fill data (same trade_id, no new one minted).
    repaired_open_from_wallet: int = 0
    # An INTENT (or other non-terminal) lifecycle record had no matching wallet position and
    # no matching lifecycle-store history — resolved using the paper engine's durable order
    # ledger (ground truth) rather than left to rot as an unexplained INTENT forever.
    repaired_from_order_ledger: int = 0
    # A stale INTENT resolved to "no confirmed fill ever happened" — safely marked REJECTED.
    resolved_never_filled: int = 0
    # Could not be resolved from any durable source (wallet, order ledger). Surfaced loudly;
    # never silently treated as fine.
    ambiguous: int = 0

    @property
    def in_sync(self) -> bool:
        return self.lifecycle_without_wallet == 0 and self.ambiguous == 0

    def summary(self) -> str:
        return (
            f"restored={self.restored_managers}, imported={self.imported_wallet_positions}, "
            f"missing_wallet={self.lifecycle_without_wallet}, "
            f"ghost_managers={self.removed_ghost_managers}, "
            f"repaired_open={self.repaired_open_from_wallet}, "
            f"repaired_from_orders={self.repaired_from_order_ledger}, "
            f"resolved_never_filled={self.resolved_never_filled}, "
            f"ambiguous={self.ambiguous}"
        )


class PaperTradeLifecycleStore:
    """Durable canonical trade identity and append-only lifecycle event store."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url
        # Ensures newly registered lifecycle tables exist even when another store created
        # the shared engine before this object was constructed.
        Base.metadata.create_all(get_engine(database_url))

    def _session(self) -> Session:
        return get_session(self._database_url)

    @staticmethod
    def new_trade_id() -> str:
        return f"PTR-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"

    @staticmethod
    def _event(
        trade_id: str,
        namespace: str,
        event_type: str,
        *,
        order_id: str = "",
        quantity: int = 0,
        price: float = 0.0,
        realized_pnl: float = 0.0,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PaperTradeEventRecord:
        return PaperTradeEventRecord(
            event_id=f"PTE-{uuid4().hex}",
            trade_id=trade_id,
            namespace=namespace,
            event_type=event_type,
            order_id=order_id,
            quantity=quantity,
            price=price,
            realized_pnl=realized_pnl,
            reason=reason,
            metadata_json=json.dumps(metadata or {}, default=str),
            timestamp=datetime.now(UTC),
        )

    def create_intent(
        self,
        *,
        namespace: str,
        run_id: str,
        workflow_id: str,
        idempotency_key: str,
        signal_id: str,
        symbol: str,
        side: str,
        strategy: str,
        timeframe: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        rationale: list[str],
        context: dict[str, Any],
        active_lessons: list[str] | None = None,
        trade_id: str | None = None,
    ) -> str:
        if quantity <= 0:
            raise ValueError("paper lifecycle intent requires positive quantity")
        now = datetime.now(UTC)
        trade_id = trade_id or self.new_trade_id()
        session = self._session()
        try:
            existing = (
                session.query(PaperTradeLifecycleRecord)
                .filter_by(idempotency_key=idempotency_key)
                .first()
            )
            if existing is not None:
                return str(existing.trade_id)
            record = PaperTradeLifecycleRecord(
                trade_id=trade_id,
                namespace=namespace,
                run_id=run_id,
                workflow_id=workflow_id,
                idempotency_key=idempotency_key,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                strategy=strategy,
                timeframe=timeframe,
                status="INTENT",
                original_quantity=quantity,
                remaining_quantity=0,
                entry_price=entry_price,
                weighted_entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                rationale_json=json.dumps(rationale, default=str),
                context_json=json.dumps(context, default=str),
                active_lessons_json=json.dumps(active_lessons or []),
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.add(
                self._event(
                    trade_id,
                    namespace,
                    "INTENT",
                    quantity=quantity,
                    price=entry_price,
                    metadata={"stop_loss": stop_loss, "target_price": target_price},
                )
            )
            session.commit()
            return trade_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def mark_open(
        self,
        trade_id: str,
        *,
        order_id: str,
        position_id: str,
        quantity: int,
        fill_price: float,
        entry_charges: float = 0.0,
    ) -> None:
        session = self._session()
        try:
            record = session.get(PaperTradeLifecycleRecord, trade_id)
            if record is None:
                raise KeyError(f"Unknown paper trade lifecycle: {trade_id}")
            if record.status == "OPEN":
                return
            record.status = "OPEN"
            record.entry_order_id = order_id
            record.position_id = position_id
            record.original_quantity = quantity
            record.remaining_quantity = quantity
            record.entry_price = fill_price
            record.weighted_entry_price = fill_price
            record.current_price = fill_price
            record.entry_charges = entry_charges
            record.updated_at = datetime.now(UTC)
            session.add(
                self._event(
                    trade_id,
                    str(record.namespace),
                    "ENTRY_FILL",
                    order_id=order_id,
                    quantity=quantity,
                    price=fill_price,
                    metadata={"position_id": position_id, "entry_charges": entry_charges},
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def record_add(
        self,
        trade_id: str,
        *,
        order_id: str,
        quantity: int,
        fill_price: float,
        entry_charges: float = 0.0,
    ) -> None:
        session = self._session()
        try:
            record = session.get(PaperTradeLifecycleRecord, trade_id)
            if record is None or record.status not in ("OPEN", "PARTIAL"):
                raise ValueError("add requires an open paper lifecycle")
            old_quantity = int(record.remaining_quantity)
            new_quantity = old_quantity + quantity
            record.weighted_entry_price = (
                float(record.weighted_entry_price) * old_quantity + fill_price * quantity
            ) / new_quantity
            record.remaining_quantity = new_quantity
            record.original_quantity = int(record.original_quantity) + quantity
            record.add_count = int(record.add_count) + 1
            record.entry_charges = float(record.entry_charges) + entry_charges
            record.current_price = fill_price
            record.updated_at = datetime.now(UTC)
            session.add(
                self._event(
                    trade_id,
                    str(record.namespace),
                    "ADD_FILL",
                    order_id=order_id,
                    quantity=quantity,
                    price=fill_price,
                    metadata={"weighted_entry_price": record.weighted_entry_price},
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def record_exit(
        self,
        trade_id: str,
        *,
        order_id: str,
        quantity: int,
        exit_price: float,
        realized_pnl: float,
        exit_charges: float = 0.0,
        reason: str,
        mae: float = 0.0,
        mfe: float = 0.0,
    ) -> bool:
        """Record one exit leg and return True only when the lifecycle is fully closed.

        ``mae``/``mfe`` are optional per-call excursion snapshots (from the exit manager's
        in-memory ``ManagedPosition`` at the moment of this fill); they are max-accumulated
        onto the durable record so the eventual finalize step (journal/performance/learning)
        can read them even if a crash later wipes the exit manager's in-memory state.
        """
        session = self._session()
        try:
            record = session.get(PaperTradeLifecycleRecord, trade_id)
            if record is None:
                raise KeyError(f"Unknown paper trade lifecycle: {trade_id}")
            if record.status == "CLOSED":
                return True
            remaining = max(0, int(record.remaining_quantity) - quantity)
            record.remaining_quantity = remaining
            record.current_price = exit_price
            record.cumulative_pnl = float(record.cumulative_pnl) + realized_pnl
            record.exit_charges = float(record.exit_charges) + exit_charges
            record.mae = max(float(record.mae), mae)  # type: ignore[arg-type]
            record.mfe = max(float(record.mfe), mfe)  # type: ignore[arg-type]
            record.status = "CLOSED" if remaining == 0 else "PARTIAL"
            record.updated_at = datetime.now(UTC)
            if remaining == 0:
                record.closed_at = datetime.now(UTC)
            session.add(
                self._event(
                    trade_id,
                    str(record.namespace),
                    "EXIT_FILL" if remaining == 0 else "PARTIAL_EXIT_FILL",
                    order_id=order_id,
                    quantity=quantity,
                    price=exit_price,
                    realized_pnl=realized_pnl,
                    reason=reason,
                    metadata={"remaining_quantity": remaining},
                )
            )
            session.commit()
            return remaining == 0
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def mark_rejected(self, trade_id: str, reason: str) -> None:
        session = self._session()
        try:
            record = session.get(PaperTradeLifecycleRecord, trade_id)
            if record is None:
                return
            record.status = "REJECTED"
            record.updated_at = datetime.now(UTC)
            session.add(
                self._event(
                    trade_id,
                    str(record.namespace),
                    "REJECTED",
                    reason=reason,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def mark_finalized(self, trade_id: str) -> bool:
        """Mark the journal/performance/learning fan-out complete for a CLOSED trade.

        Idempotent — returns True only the first time (i.e. the caller that actually
        performed the fan-out should record it); a repeat call is a no-op returning False
        so a caller can distinguish "I just finalized it" from "someone already did."
        """
        session = self._session()
        try:
            record = session.get(PaperTradeLifecycleRecord, trade_id)
            if record is None:
                raise KeyError(f"Unknown paper trade lifecycle: {trade_id}")
            if record.finalized_at is not None:
                return False
            record.finalized_at = datetime.now(UTC)
            record.updated_at = datetime.now(UTC)
            session.add(
                self._event(trade_id, str(record.namespace), "FINALIZED"),
            )
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_unfinalized_closed(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """CLOSED trades whose journal/performance/learning fan-out never completed.

        This is the durable-outbox drain list: on a normal run it is always empty (the
        finalize step runs synchronously right after the closing fill); after a crash
        between the ledger commit and the fan-out, it surfaces exactly the trades that
        still need journal/performance/lesson-crediting replay at startup.
        """
        session = self._session()
        try:
            query = session.query(PaperTradeLifecycleRecord).filter(
                PaperTradeLifecycleRecord.status == "CLOSED",
                PaperTradeLifecycleRecord.finalized_at.is_(None),
            )
            if namespace:
                query = query.filter(PaperTradeLifecycleRecord.namespace == namespace)
            return [self._to_dict(record) for record in query.all()]
        finally:
            session.close()

    def get_by_status(self, status: str, namespace: str | None = None) -> list[dict[str, Any]]:
        session = self._session()
        try:
            query = session.query(PaperTradeLifecycleRecord).filter(
                PaperTradeLifecycleRecord.status == status
            )
            if namespace:
                query = query.filter(PaperTradeLifecycleRecord.namespace == namespace)
            return [self._to_dict(record) for record in query.all()]
        finally:
            session.close()

    def get(self, trade_id: str) -> dict[str, Any] | None:
        session = self._session()
        try:
            record = session.get(PaperTradeLifecycleRecord, trade_id)
            return self._to_dict(record) if record is not None else None
        finally:
            session.close()

    def get_open(self, namespace: str | None = None) -> list[dict[str, Any]]:
        session = self._session()
        try:
            query = session.query(PaperTradeLifecycleRecord).filter(
                PaperTradeLifecycleRecord.status.in_(["OPEN", "PARTIAL"])
            )
            if namespace:
                query = query.filter(PaperTradeLifecycleRecord.namespace == namespace)
            return [self._to_dict(record) for record in query.all()]
        finally:
            session.close()

    def events(self, trade_id: str) -> list[dict[str, Any]]:
        session = self._session()
        try:
            rows = (
                session.query(PaperTradeEventRecord)
                .filter_by(trade_id=trade_id)
                .order_by(PaperTradeEventRecord.timestamp)
                .all()
            )
            return [
                {
                    "event_id": row.event_id,
                    "trade_id": row.trade_id,
                    "namespace": row.namespace,
                    "event_type": row.event_type,
                    "order_id": row.order_id,
                    "quantity": row.quantity,
                    "price": row.price,
                    "realized_pnl": row.realized_pnl,
                    "reason": row.reason,
                    "metadata": json.loads(str(row.metadata_json or "{}")),
                    "timestamp": row.timestamp,
                }
                for row in rows
            ]
        finally:
            session.close()

    def _repair_from_orders(self, trade_id: str, orders: list[Any]) -> str:
        """Replay a trade_id's fills from the paper engine's durable order ledger.

        Ground truth for a lifecycle record stuck at INTENT with no matching wallet
        position: the paper engine's order table is written inside ``place_order``'s own
        transaction, so it can never be *behind* the ledger the way this separate lifecycle
        store can after a crash between "order filled" and "ledger updated". Replaying every
        confirmed fill (entry, capped-averaging adds, exits — in timestamp order) rebuilds
        the ledger's status/quantity/cumulative-P&L exactly as if no crash had occurred.

        Returns one of "repaired_open", "repaired_from_orders" (fully replayed, incl. any
        exits), or "resolved_never_filled" (no confirmed fill exists — safe to reject).
        """
        fills = [o for o in orders if o.status in ("FILLED", "PARTIALLY_FILLED")]
        if not fills:
            self.mark_rejected(trade_id, "startup-reconcile: no confirmed fill found for intent")
            return "resolved_never_filled"

        entries = [o for o in fills if o.reason == "entry"]
        if not entries:
            # Fills exist but none is tagged as the entry leg — cannot safely reconstruct
            # an opening basis. Surface rather than guess.
            logger.error(
                "Cannot repair lifecycle %s from order ledger: %d fill(s) found but none "
                "tagged reason='entry'.",
                trade_id,
                len(fills),
            )
            return "ambiguous"

        entry = entries[0]
        self.mark_open(
            trade_id,
            order_id=entry.order_id,
            position_id=entry.position_id or trade_id,
            quantity=entry.quantity,
            fill_price=entry.price,
            entry_charges=entry.entry_charges,
        )
        for order in fills:
            if order is entry:
                continue
            if order.reason == "capped_average":
                self.record_add(
                    trade_id,
                    order_id=order.order_id,
                    quantity=order.quantity,
                    fill_price=order.price,
                    entry_charges=order.entry_charges,
                )
            else:
                self.record_exit(
                    trade_id,
                    order_id=order.order_id,
                    quantity=order.quantity,
                    exit_price=order.price,
                    realized_pnl=order.realized_pnl,
                    exit_charges=order.exit_charges,
                    reason=order.reason or "unknown",
                )
        return "repaired_from_orders"

    def reconcile(
        self,
        wallet_positions: Iterable[Any],
        exit_manager: Any,
        order_lookup: Callable[[str], list[Any]] | None = None,
        namespace: str = PAPER_DATA_NAMESPACE,
    ) -> ReconciliationReport:
        """Rebuild managed exits from the ledger and import legacy wallet positions.

        ``order_lookup``, if given, is called with a trade_id and must return every durable
        paper-engine order tagged with it (oldest first) — see
        ``LocalPaperEngine.get_orders_for_trade``. It is the ground truth used to resolve
        lifecycle records left in an ambiguous intermediate state (INTENT with no matching
        wallet position) rather than leaving them to silently rot; without it, such records
        are left untouched and counted as ``ambiguous`` so the caller can decide how loud to
        be (this store owns making the ledger reconcilable, not the startup gate policy).
        """
        wallet = {str(position.position_id): position for position in wallet_positions}
        open_records = self.get_open(namespace)
        lifecycle_by_position = {str(record["position_id"]): record for record in open_records}
        restored = imported = missing = ghosts = 0
        repaired_open = repaired_from_orders = resolved_never_filled = ambiguous = 0

        for position_id, position in wallet.items():
            record = lifecycle_by_position.get(position_id)
            if record is None:
                existing = self.get(position_id)
                if existing is not None and existing["status"] == "INTENT":
                    # The paper-engine fill committed (this wallet position exists) but the
                    # crash happened before lifecycle_store.mark_open ran. Repair the SAME
                    # trade_id from the wallet's own fill data — do not mint a new one, and
                    # do not call create_intent (its primary key already exists).
                    self.mark_open(
                        position_id,
                        order_id="startup-repair",
                        position_id=position_id,
                        quantity=position.quantity,
                        fill_price=position.entry_price,
                        entry_charges=position.entry_charges,
                    )
                    record = self.get(position_id)
                    repaired_open += 1
                elif existing is not None:
                    # The ledger believes this trade_id is already CLOSED/REJECTED/PARTIAL
                    # yet the wallet still holds an open position under the same id. Neither
                    # side is trustworthy here — surface it loudly instead of guessing.
                    logger.error(
                        "Lifecycle/wallet mismatch for trade_id=%s: ledger status=%s but "
                        "wallet still holds an open position — leaving both untouched.",
                        position_id,
                        existing["status"],
                    )
                    ambiguous += 1
                    continue
                else:
                    trade_id = position_id or self.new_trade_id()
                    idempotency_key = f"legacy-reconcile:{trade_id}"
                    self.create_intent(
                        namespace=namespace,
                        run_id="startup-reconcile",
                        workflow_id="startup-reconcile",
                        idempotency_key=idempotency_key,
                        signal_id="",
                        symbol=position.symbol,
                        side=position.side,
                        strategy=position.strategy,
                        timeframe="",
                        quantity=position.quantity,
                        entry_price=position.entry_price,
                        stop_loss=position.stop_loss,
                        target_price=position.target_price,
                        rationale=[
                            "Imported legacy wallet position during startup reconciliation."
                        ],
                        context={"reconciled": True},
                        trade_id=trade_id,
                    )
                    self.mark_open(
                        trade_id,
                        order_id="startup-reconcile",
                        position_id=position_id,
                        quantity=position.quantity,
                        fill_price=position.entry_price,
                        entry_charges=position.entry_charges,
                    )
                    record = self.get(trade_id)
                    imported += 1
            if record is not None and exit_manager.get_position(str(record["trade_id"])) is None:
                exit_manager.register_position(
                    position_id=str(record["trade_id"]),
                    symbol=position.symbol,
                    side=position.side,
                    quantity=position.quantity,
                    entry_price=float(record["weighted_entry_price"]),
                    stop_loss=float(record["stop_loss"]),
                    target_price=float(record["target_price"]),
                    strategy=str(record["strategy"]),
                    regime=str(record["context"].get("regime", "")),
                )
                restored += 1

        for record in open_records:
            if str(record["position_id"]) not in wallet:
                missing += 1

        # Orphaned INTENT records: no matching wallet position, so the entry either never
        # filled or filled-and-fully-exited without the ledger ever being updated. Resolve
        # deterministically from the durable order ledger rather than leaving them stuck.
        for record in self.get_by_status("INTENT"):
            trade_id = str(record["trade_id"])
            if trade_id in wallet:
                continue  # handled by the wallet-driven repair above
            if order_lookup is None:
                logger.warning(
                    "Lifecycle %s stuck at INTENT with no order ledger to reconcile against "
                    "— leaving as-is and surfacing as ambiguous.",
                    trade_id,
                )
                ambiguous += 1
                continue
            outcome = self._repair_from_orders(trade_id, order_lookup(trade_id))
            if outcome == "repaired_from_orders":
                repaired_from_orders += 1
            elif outcome == "resolved_never_filled":
                resolved_never_filled += 1
            else:
                ambiguous += 1

        valid_trade_ids = {str(record["trade_id"]) for record in self.get_open()}
        for managed in list(exit_manager.get_managed_positions()):
            if managed.position_id not in valid_trade_ids:
                exit_manager.unregister_position(managed.position_id)
                ghosts += 1

        return ReconciliationReport(
            restored,
            imported,
            missing,
            ghosts,
            repaired_open,
            repaired_from_orders,
            resolved_never_filled,
            ambiguous,
        )

    @staticmethod
    def _to_dict(record: PaperTradeLifecycleRecord) -> dict[str, Any]:
        return {
            column.name: getattr(record, column.name)
            for column in PaperTradeLifecycleRecord.__table__.columns
        } | {
            "rationale": json.loads(str(record.rationale_json or "[]")),
            "context": json.loads(str(record.context_json or "{}")),
            "active_lessons": json.loads(str(record.active_lessons_json or "[]")),
        }
