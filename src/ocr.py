from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .models import TextBlock


class EasyOcrEngine:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, lang: str = "ja", use_gpu: bool = True, confidence: float = 0.1,
                 text_threshold: float = 0.7, low_text: float = 0.4, min_size: int = 10, canvas_size: int = 4096):
        lang_map = {"japan": "ja", "ja": "ja", "japanese": "ja", "chinese": "ch_sim", "eng": "en"}
        self.lang = lang_map.get(lang, lang)
        self.use_gpu = use_gpu
        self.confidence = confidence
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.min_size = min_size
        self.canvas_size = canvas_size
        self._reader = None

    def _lazy_init(self):
        if self._reader is not None:
            return
        with self._lock:
            if self._reader is not None:
                return
            import easyocr
            self._reader = easyocr.Reader(
                [self.lang],
                gpu=self.use_gpu,
            )

    def ocr_image(self, image_bytes: bytes, page_num: int) -> list[TextBlock]:
        self._lazy_init()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert("RGB"))

        results = self._reader.readtext(
            img_array,
            text_threshold=self.text_threshold,
            low_text=self.low_text,
            min_size=self.min_size,
            canvas_size=self.canvas_size,
        )
        blocks = []
        for bbox, text, conf in results:
            if conf < self.confidence:
                continue
            pts = bbox
            x0 = int(min(p[0] for p in pts))
            y0 = int(min(p[1] for p in pts))
            x1 = int(max(p[0] for p in pts))
            y1 = int(max(p[1] for p in pts))

            blocks.append(
                TextBlock(
                    bbox=(x0, y0, x1, y1),
                    text=text,
                    source="ocr",
                    page_num=page_num,
                )
            )
        return blocks


class PaddleOcrEngine:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, lang: str = "japan", confidence: float = 0.3):
        self.lang = lang
        self.confidence = confidence
        self._ocr = None

    def _lazy_init(self):
        if self._ocr is not None:
            return
        with self._lock:
            if self._ocr is not None:
                return
            import paddle
            paddle.device.set_device("cpu")
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=True,
                use_gpu=False,
                show_log=False,
            )

    def ocr_image(self, image_bytes: bytes, page_num: int) -> list[TextBlock]:
        self._lazy_init()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img.convert("RGB"))[:, :, ::-1].copy()

        results = self._ocr.ocr(img_array, cls=True)
        blocks = []
        if not results or not results[0]:
            return blocks

        for line in results[0]:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]
            if conf < self.confidence:
                continue

            x0 = int(min(p[0] for p in bbox))
            y0 = int(min(p[1] for p in bbox))
            x1 = int(max(p[0] for p in bbox))
            y1 = int(max(p[1] for p in bbox))

            blocks.append(
                TextBlock(
                    bbox=(x0, y0, x1, y1),
                    text=text,
                    source="ocr",
                    page_num=page_num,
                )
            )
        return blocks


_easyocr_engine: Optional[EasyOcrEngine] = None
_paddle_engine: Optional[PaddleOcrEngine] = None


def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0


def _merge_blocks(all_blocks: list[list[TextBlock]]) -> list[TextBlock]:
    flat = []
    for blocks in all_blocks:
        flat.extend(blocks)
    flat.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

    merged = []
    for b in flat:
        dup = False
        for i, m in enumerate(merged):
            iou = _iou(b.bbox, m.bbox)
            if iou > 0.3:
                dup = True
                if len(b.text) > len(m.text):
                    merged[i] = b
                break
        if not dup:
            merged.append(b)
    return merged


def get_ocr_engine(engine: str = "easyocr", lang: str = "ja", use_gpu: bool = True, confidence: float = 0.1,
                   text_threshold: float = 0.7, low_text: float = 0.4, min_size: int = 10, canvas_size: int = 4096):
    return engine  # engines created on-demand in ocr_page


def ocr_page(image_bytes: bytes, page_num: int, engine: str = "easyocr", lang: str = "ja", use_gpu: bool = True,
             confidence: float = 0.1, text_threshold: float = 0.7, low_text: float = 0.4,
             min_size: int = 10, canvas_size: int = 4096) -> list[TextBlock]:
    global _easyocr_engine, _paddle_engine

    if engine == "both":
        if _easyocr_engine is None:
            _easyocr_engine = EasyOcrEngine(lang=lang, use_gpu=use_gpu, confidence=confidence,
                                            text_threshold=text_threshold, low_text=low_text,
                                            min_size=min_size, canvas_size=canvas_size)

        def _run_paddle() -> list[TextBlock]:
            tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_img.write(image_bytes)
            tmp_img.close()
            tmp_out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            tmp_out.close()
            worker = Path(__file__).parent / "_paddle_worker.py"
            try:
                subprocess.run(
                    [sys.executable, str(worker), tmp_img.name, str(page_num), tmp_out.name],
                    capture_output=True, timeout=300,
                )
                with open(tmp_out.name, encoding="utf-8") as f:
                    data = json.load(f)
                blocks = []
                for d in data:
                    b = d["bbox"]
                    blocks.append(TextBlock(bbox=tuple(b), text=d["text"], source="ocr", page_num=page_num))
                return blocks
            except Exception:
                return []
            finally:
                try:
                    os.unlink(tmp_img.name)
                    os.unlink(tmp_out.name)
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=2) as pool:
            paddle_future = pool.submit(_run_paddle)
            easy_future = pool.submit(_easyocr_engine.ocr_image, image_bytes, page_num=page_num)
            paddle_blocks = paddle_future.result()
            easy_blocks = easy_future.result()

        return _merge_blocks([easy_blocks, paddle_blocks])
    elif engine == "paddle":
        if _paddle_engine is None:
            _paddle_engine = PaddleOcrEngine(lang="japan", use_gpu=use_gpu, confidence=confidence)
        return _paddle_engine.ocr_image(image_bytes, page_num=page_num)
    else:
        if _easyocr_engine is None:
            _easyocr_engine = EasyOcrEngine(lang=lang, use_gpu=use_gpu, confidence=confidence,
                                            text_threshold=text_threshold, low_text=low_text,
                                            min_size=min_size, canvas_size=canvas_size)
        return _easyocr_engine.ocr_image(image_bytes, page_num=page_num)
