"""Future crypto provider boundary. No exchange SDK is selected."""

from typing import Any, Protocol


class CryptoBrokerProvider(Protocol):
    async def health(self) -> dict[str, Any]: ...

