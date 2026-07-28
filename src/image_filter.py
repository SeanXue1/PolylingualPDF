"""Filter OCR text blocks that fall inside photographic/image regions.

Image bounding boxes come from PyMuPDF in PDF point space (72 DPI),
while OCR bounding boxes are in pixel space at `source_dpi`.  This
module handles the coordinate conversion and applies overlap-based
filtering to remove spurious text detections inside photographs.
"""

from __future__ import annotations

from .models import TextBlock


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    pad: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _scale_image_bbox_to_ocr_space(
    image_bbox: tuple[float, float, float, float],
    source_dpi: int,
    page_height: float,
) -> tuple[float, float, float, float]:
    """Convert a bbox from PDF point space (72 DPI) to OCR pixel space.

    PyMuPDF page coordinates use a bottom-left origin, while OCR outputs are in
    raster image coordinates with a top-left origin. We therefore scale and flip
    Y using the page height.
    """
    scale = source_dpi / 72.0
    x0, y0, x1, y1 = image_bbox
    img_h = page_height * scale
    return (x0 * scale, img_h - (y1 * scale), x1 * scale, img_h - (y0 * scale))


def _overlap_ratio(
    text_bbox: tuple[float, float, float, float],
    image_bbox: tuple[float, float, float, float],
) -> float:
    """Return the fraction of the text bbox's area that overlaps with the image bbox."""
    tx0, ty0, tx1, ty1 = text_bbox
    ix0, iy0, ix1, iy1 = image_bbox

    inter_x0 = max(tx0, ix0)
    inter_y0 = max(ty0, iy0)
    inter_x1 = min(tx1, ix1)
    inter_y1 = min(ty1, iy1)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    text_area = (tx1 - tx0) * (ty1 - ty0)
    return inter_area / text_area if text_area > 0 else 0.0


def _center_inside(
    text_bbox: tuple[float, float, float, float],
    image_bbox: tuple[float, float, float, float],
) -> bool:
    """Return True if the center of the text bbox falls inside the image bbox."""
    cx = (text_bbox[0] + text_bbox[2]) / 2
    cy = (text_bbox[1] + text_bbox[3]) / 2
    return image_bbox[0] <= cx <= image_bbox[2] and image_bbox[1] <= cy <= image_bbox[3]


def filter_image_text(
    text_blocks: list[TextBlock],
    image_bboxes: list[tuple[float, float, float, float]],
    source_dpi: int,
    page_width: float = 0.0,
    page_height: float = 0.0,
    overlap_threshold: float = 0.70,
) -> list[TextBlock]:
    """Remove text blocks that are inside photographic image regions.

    A text block is removed if:
    - Its center point falls inside an image bounding box, AND
    - At least *overlap_threshold* (default 70%) of its area overlaps with that image, AND
    - The image is not a background image (covering more than 80% of the page area).

    Both conditions must be met so that text that merely touches or is at the
    edge of an image (e.g., captions, titles overlaid on banners) is preserved.

    Parameters
    ----------
    text_blocks : list[TextBlock]
        OCR-detected text blocks in pixel space (at *source_dpi*).
    image_bboxes : list[tuple]
        Image bounding boxes in PDF point space (72 DPI).
    source_dpi : int
        The DPI used for page rasterization / OCR.
    page_width : float
        Page width in PDF points (72 DPI).
    page_height : float
        Page height in PDF points (72 DPI).
    overlap_threshold : float
        Minimum area overlap ratio to consider a text block as "inside" an image.

    Returns
    -------
    list[TextBlock]
        Filtered text blocks with image-region text removed.
    """
    if not image_bboxes or not text_blocks:
        return text_blocks

    page_area = page_width * page_height if page_width > 0 and page_height > 0 else 0.0
    pad = max(4.0, source_dpi * 0.01)

    # Scale non-background image bboxes to OCR pixel space
    scaled_image_bboxes = []
    for ib in image_bboxes:
        if page_area > 0:
            ib_w = ib[2] - ib[0]
            ib_h = ib[3] - ib[1]
            ib_area = ib_w * ib_h
            if ib_area / page_area > 0.85:
                continue # Skip background/full-page images
        scaled = _scale_image_bbox_to_ocr_space(ib, source_dpi, page_height)
        scaled_image_bboxes.append(_expand_bbox(scaled, pad))

    filtered: list[TextBlock] = []
    for block in text_blocks:
        in_image = False
        for img_bbox in scaled_image_bboxes:
            overlap = _overlap_ratio(block.bbox, img_bbox)
            if overlap >= overlap_threshold:
                in_image = True
                break
            if _center_inside(block.bbox, img_bbox) and overlap >= 0.25:
                in_image = True
                break
        if not in_image:
            filtered.append(block)

    return filtered
