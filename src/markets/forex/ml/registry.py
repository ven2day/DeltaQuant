"""Forex-bound model registry; cross-market records fail closed."""

from src.core.ml import ModelGrain, ModelRecord, ModelRegistry, ModelStatus


class ForexModelRegistry(ModelRegistry):
    @staticmethod
    def _check(grain: ModelGrain) -> None:
        if grain.market.strip().upper() != "FOREX":
            raise ValueError("Forex model registry cannot access another market")

    def register(self, record: ModelRecord) -> None:
        self._check(record.grain)
        super().register(record)

    def latest_active(self, grain: ModelGrain) -> ModelRecord | None:
        self._check(grain)
        return super().latest_active(grain)

    def promote(self, grain: ModelGrain, model_version: str) -> ModelRecord:
        self._check(grain)
        return super().promote(grain, model_version)

    def set_status(self, grain: ModelGrain, model_version: str, status: ModelStatus) -> None:
        self._check(grain)
        super().set_status(grain, model_version, status)

    def list_records(self, *, market: str | None = None) -> list[ModelRecord]:
        if market is not None and market.strip().upper() != "FOREX":
            raise ValueError("Forex model registry cannot list another market")
        return super().list_records(market="FOREX")


__all__ = ["ForexModelRegistry"]
