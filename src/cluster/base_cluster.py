from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Paragraph, TextBlock


class BaseClusterAlgorithm(ABC):
    @abstractmethod
    def build_clusters(
        self,
        text_boxes: list[TextBlock],
        page_width: float = 0.0,
        page_height: float = 0.0,
        **kwargs,
    ) -> list[Paragraph]:
        ...
