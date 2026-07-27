from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextBlock:
    bbox: tuple[float, float, float, float]
    text: str
    source: str  # "native" | "ocr"
    page_num: int
    block_type: str = "text"  # "text" | "image" | "table"
    font_size: float = 0.0


@dataclass
class Paragraph:
    blocks: list[TextBlock] = field(default_factory=list)
    translation: str = ""
    line_bboxes: list[tuple[tuple[float, float, float, float], str, float]] = field(default_factory=list)
    translated_lines: list[str] = field(default_factory=list)
    layout_type: str = ""

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks)

    @text.setter
    def text(self, value: str) -> None:
        if self.blocks:
            first = self.blocks[0]
            self.blocks = [TextBlock(bbox=first.bbox, text=value, source=first.source, page_num=first.page_num)]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if not self.blocks:
            return (0, 0, 0, 0)
        x0 = min(b.bbox[0] for b in self.blocks)
        y0 = min(b.bbox[1] for b in self.blocks)
        x1 = max(b.bbox[2] for b in self.blocks)
        y1 = max(b.bbox[3] for b in self.blocks)
        return (x0, y0, x1, y1)


@dataclass
class TextCluster:
    paragraphs: list[Paragraph] = field(default_factory=list)
    translation: str = ""

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        boxes = self.all_line_bboxes
        if not boxes:
            return (0, 0, 0, 0)
        bboxes = [b for b, _, _ in boxes]
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return (x0, y0, x1, y1)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    @property
    def all_line_bboxes(self) -> list[tuple[tuple[float, float, float, float], str, float]]:
        result = []
        for p in self.paragraphs:
            result.extend(p.line_bboxes)
        return result


@dataclass
class PageResult:
    page_num: int
    blocks: list[TextBlock] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    clusters: list[TextCluster] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    needs_ocr: bool = False
    image: Optional[bytes] = None
    source_dpi: int = 72
