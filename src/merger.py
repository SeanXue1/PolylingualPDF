from __future__ import annotations

from .models import PageResult, Paragraph, TextBlock, TextCluster


def _line_height(bbox: tuple[float, float, float, float]) -> float:
    return bbox[3] - bbox[1]


def _vertical_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return b[1] - a[3]


def _horizontal_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    return max(0.0, overlap)


def _x_center(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[0] + bbox[2]) / 2


def _column_cluster(blocks: list[TextBlock], x_gap_threshold: float = 50.0) -> list[list[TextBlock]]:
    if not blocks:
        return []
    # Keep blocks in approximate left-to-right column order, but split when the
    # horizontal gap is large enough that the items are unlikely to belong to
    # the same column. This avoids merging separate columns that happen to sit
    # on the same horizontal band.
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[0], b.bbox[1]))
    clusters: list[list[TextBlock]] = [[sorted_blocks[0]]]

    for block in sorted_blocks[1:]:
        prev_b = clusters[-1][-1].bbox
        curr_b = block.bbox
        x_gap = curr_b[0] - prev_b[2]
        if x_gap > x_gap_threshold:
            clusters.append([block])
        else:
            clusters[-1].append(block)

    return clusters


def _merge_column(blocks: list[TextBlock], line_spacing_ratio: float = 1.5) -> list[Paragraph]:
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    paragraphs: list[Paragraph] = []
    current: list[TextBlock] = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        prev = current[-1]
        prev_h = _line_height(prev.bbox)
        gap = _vertical_gap(prev.bbox, block.bbox)

        if gap <= max(prev_h * line_spacing_ratio, 4.0):
            current.append(block)
        else:
            paragraphs.append(Paragraph(blocks=list(current), line_bboxes=[(b.bbox, b.text, b.font_size) for b in current]))
            current = [block]

    if current:
        paragraphs.append(Paragraph(blocks=list(current), line_bboxes=[(b.bbox, b.text, b.font_size) for b in current]))

    return paragraphs


def merge_paragraphs(
    blocks: list[TextBlock],
    line_spacing_ratio: float = 1.5,
    column_threshold: float = 50.0,
) -> list[Paragraph]:
    if not blocks:
        return []

    clusters = _column_cluster(blocks, x_gap_threshold=column_threshold)
    result: list[Paragraph] = []
    for cluster in clusters:
        result.extend(_merge_column(cluster, line_spacing_ratio=line_spacing_ratio))

    return result


def build_page_paragraphs(result: PageResult, **kwargs) -> PageResult:
    if result.blocks:
        result.paragraphs = merge_paragraphs(result.blocks, **kwargs)
    return result


def cluster_paragraphs(
    paragraphs: list[Paragraph],
    line_spacing_ratio: float = 2.5,
    column_threshold: float = 50.0,
) -> list[TextCluster]:
    if not paragraphs:
        return []
    # Each paragraph object represents a distinct logical paragraph cluster.
    return [TextCluster(paragraphs=[p]) for p in paragraphs]
