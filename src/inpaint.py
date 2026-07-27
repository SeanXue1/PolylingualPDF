from __future__ import annotations

import numpy as np

from .models import PageResult


def inpaint_page(
    img_bytes: bytes,
    img_width: int,
    img_height: int,
    pr: PageResult,
) -> bytes:
    import cv2

    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_height, img_width, 3)

    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    dpi = pr.source_dpi
    pad = int(2 * dpi / 72)

    for para in pr.paragraphs:
        for raw_bbox, _, _ in para.line_bboxes:
            lx0, ly0, lx1, ly1 = raw_bbox
            x0 = max(0, int(lx0) - pad)
            y0 = max(0, int(ly0) - pad)
            x1 = min(img_width, int(lx1) + pad)
            y1 = min(img_height, int(ly1) + pad)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 255

    radius = max(1, min(int(3 * dpi / 72), 10))
    inpainted = cv2.inpaint(img, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

    _, buf = cv2.imencode(".jpg", cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()
