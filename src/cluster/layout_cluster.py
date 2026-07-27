from __future__ import annotations

from ..models import Paragraph, TextBlock
from .base_cluster import BaseClusterAlgorithm
from .cluster_builder import run_cluster_pipeline


class LayoutClusterAlgorithm(BaseClusterAlgorithm):
    def __init__(
        self,
        column_gap: float = 30.0,
        line_spacing_ratio: float = 1.5,
        min_region_size: int = 25,
    ):
        self.column_gap = column_gap
        self.line_spacing_ratio = line_spacing_ratio
        self.min_region_size = min_region_size

    def build_clusters(
        self,
        text_boxes: list[TextBlock],
        page_width: float = 0.0,
        page_height: float = 0.0,
        **kwargs,
    ) -> list[Paragraph]:
        return run_cluster_pipeline(
            text_boxes,
            page_width=page_width,
            page_height=page_height,
            column_gap=self.column_gap,
            line_spacing_ratio=self.line_spacing_ratio,
            min_region_size=self.min_region_size,
        )
