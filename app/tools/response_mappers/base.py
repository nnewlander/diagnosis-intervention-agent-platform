from abc import ABC, abstractmethod
from typing import Any


class BaseResponseMapper(ABC):
    """Map remote provider JSON into internal normalized evidence schema."""

    @abstractmethod
    def map_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError
