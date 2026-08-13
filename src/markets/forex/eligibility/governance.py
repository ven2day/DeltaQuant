"""Forex validation publication guardrails."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from src.backtesting.strategy_eligibility import (
    EligibilityStatus,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)

if TYPE_CHECKING:
    from src.core.persistence import SchemaBoundTradingRepository


def demote_superseded_forex_approvals(
    registry: StrategyEligibilityRegistry,
    *,
    approved_identities: set[tuple[str, str, str, str]],
    current_model_version: str,
    validated_at: str,
    repository: SchemaBoundTradingRepository | None = None,
) -> tuple[StrategyEligibility, ...]:
    """Fail closed for old Forex approvals not admitted by the current OOS run.

    Historical metrics remain attached to the record for auditability.  Only its
    executable status is changed; no artifact or legacy registry row is deleted.
    """

    demoted: list[StrategyEligibility] = []
    for record in registry.load_all():
        if record.market.strip().upper() != "FOREX":
            continue
        if record.validation_status not in {
            EligibilityStatus.PAPER_APPROVED,
            EligibilityStatus.LIVE_APPROVED,
        }:
            continue
        if record.identity in approved_identities:
            continue
        metadata = dict(record.validation_metadata)
        metadata.update(
            {
                "previous_validation_status": record.validation_status.value,
                "superseded_by_model_version": current_model_version,
                "superseded_at": validated_at,
            }
        )
        replacement = replace(
            record,
            validation_status=EligibilityStatus.SHADOW,
            validated_at=validated_at,
            status_reason=(
                "SUPERSEDED_BY_FOREX_OOS_VALIDATION;"
                f"previous={record.validation_status.value};"
                f"current_model_version={current_model_version}"
            ),
            validation_metadata=metadata,
        )
        registry.register(replacement)
        if repository is not None:
            repository.persist_strategy_eligibility(
                identity="|".join(replacement.identity), payload=replacement.to_dict()
            )
        demoted.append(replacement)
    return tuple(demoted)


__all__ = ["demote_superseded_forex_approvals"]
