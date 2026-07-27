from __future__ import annotations

from ..models import Paragraph, TextBlock
from ..merger import merge_paragraphs
from .base_cluster import BaseClusterAlgorithm


class SimpleClusterAlgorithm(BaseClusterAlgorithm):
    def __init__(
        self,
        line_spacing_ratio: float = 1.5,
        column_threshold: float = 50.0,
    ):
        self.line_spacing_ratio = line_spacing_ratio
        self.column_threshold = column_threshold

    def build_clusters(
        self,
        text_boxes: list[TextBlock],
        page_width: float = 0.0,
        page_height: float = 0.0,
        **kwargs,
    ) -> list[Paragraph]:
        return merge_paragraphs(
            text_boxes,
            line_spacing_ratio=self.line_spacing_ratio,
            column_threshold=self.column_threshold,
        )
