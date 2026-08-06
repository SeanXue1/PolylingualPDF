from __future__ import annotations

import glob
import json
import os
import time
import warnings

import fitz
from tqdm import tqdm

from . import db
from .cache import TranslationCache
from .cluster.base_cluster import BaseClusterAlgorithm
from .cluster.layout_cluster import LayoutClusterAlgorithm
from .cluster.simple_cluster import SimpleClusterAlgorithm
from .config import Config
from .extractor import extract_pdf
from .merger import build_page_paragraphs, cluster_paragraphs
from .models import PageResult
from .ocr import ocr_page
from .image_filter import filter_image_text
from .renderer import _find_cjk_font, _get_cjk_font, _optimal_font_size, _pdf_point, render_pdf, render_single_page_to_doc
from .translator import _is_identity_failure
from .translator import create_translator




def cleanup_artifacts() -> None:
    """Remove OCR DB, translation cache, and debug JSON files before a run.

    Ensures every pipeline run starts from a clean state so data from a
    previous document (same page numbers) cannot leak into the new output.
    """
    patterns = [
        db.OCR_DB_PATH,
        "translation_cache.db",
        "localworking/debug_batch_*.json",
    ]
    removed = 0
    for pattern in patterns:
        for p in glob.glob(pattern):
            try:
                os.remove(p)
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"  [CLEANUP] Removed {removed} cache/DB/debug artifact(s)")


def _get_cluster_algorithm(config: Config) -> BaseClusterAlgorithm:
    algo = getattr(config, "cluster_algorithm", "simple")
    if algo == "layout":
        return LayoutClusterAlgorithm(
            column_gap=config.column_gap,
            line_spacing_ratio=config.line_spacing_ratio,
        )
    return SimpleClusterAlgorithm(
        line_spacing_ratio=config.line_spacing_ratio,
        column_threshold=config.column_gap,
    )


def _is_chinese_translation(text: str, original: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in t):
        return False
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in original)
    if has_cjk:
        if not any('\u4e00' <= c <= '\u9fff' for c in t):
            return False
    return True

def _para_page(para) -> int:
    return para.blocks[0].page_num if para.blocks else 0


def _save_debug_json(pages: list[PageResult], out_path: str = "debug_ocr_translate.json") -> None:
    if "localworking/" not in out_path:
        out_path = f"localworking/{out_path}"
    data = []
    for pr in pages:
        page_data = {
            "page_num": pr.page_num,
            "paragraphs": [],
        }
        for pi, para in enumerate(pr.paragraphs):
            para_data = {
                "para_idx": pi,
                "source_text": para.text,
                "translation": para.translation,
                "lines": [],
            }
            for (bbox, orig_text, _), tline in zip(para.line_bboxes, para.translated_lines or [""] * len(para.line_bboxes)):
                para_data["lines"].append({
                    "bbox": list(bbox),
                    "original": orig_text,
                    "translated": tline,
                })
            page_data["paragraphs"].append(para_data)
        data.append(page_data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    warnings.warn(f"  Debug info saved to {out_path}")


def _split_translation(para, dpi: int | None = None) -> None:
    # JSON batching already fills translated_lines — nothing to do
    if para.translated_lines:
        return
    if not para.translation:
        para.translated_lines = [""] * len(para.line_bboxes) if para.line_bboxes else []
        return
    txt = para.translation
    n = len(para.line_bboxes)
    if n <= 1:
        para.translated_lines = [txt]
        return

    # Width-aware splitting using box dimensions
    if dpi and para.line_bboxes and _find_cjk_font():
        lines = []
        remaining = txt
        for raw_bbox, _, _ in para.line_bboxes:
            if not remaining:
                lines.append("")
                continue
            lx0, ly0, lx1, ly1 = raw_bbox
            box_w = _pdf_point(lx1 - lx0, dpi)
            box_h = _pdf_point(ly1 - ly0, dpi)
            if box_w < 2 or box_h < 2:
                lines.append("")
                continue

            min_fs = 6
            lo, hi = 1, len(remaining)
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                piece = remaining[:mid]
                fs = _optimal_font_size(piece, box_w, box_h)
                if fs >= min_fs:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if best <= 0:
                best = 1
            lines.append(remaining[:best])
            remaining = remaining[best:]

        if remaining:
            lines[-1] += remaining
        para.translated_lines = lines
        return

    # Fallback: proportional splitting by original character lengths
    orig_lengths = [len(t) for _, t, _ in para.line_bboxes]
    total_orig = sum(orig_lengths) or 1
    total_t = len(txt)
    result_lines = []
    start = 0
    for ol in orig_lengths:
        seg_len = int(total_t * ol / total_orig)
        end = min(start + seg_len, len(txt))
        result_lines.append(txt[start:end])
        start = end
    if result_lines and start < len(txt):
        result_lines[-1] += txt[start:]
    para.translated_lines = result_lines


def run_extraction_pipeline(
    path: str,
    config: Config,
    progress: bool = True,
) -> list[PageResult]:
    pages = extract_pdf(
        path,
        page_start=config.page_start,
        page_end=config.page_end,
        dpi=config.extract_dpi,
    )

    iterator = tqdm(pages, desc="Extracting") if progress else pages
    for result in iterator:
        pn = result.page_num
        if result.needs_ocr and not config.no_ocr and result.image:
            try:
                ocr_blocks = ocr_page(
                    result.image,
                    page_num=pn,
                    engine=config.ocr_engine,
                    lang=config.ocr_lang,
                    use_gpu=config.use_gpu,
                    confidence=config.ocr_confidence,
                    text_threshold=config.text_threshold,
                    low_text=config.low_text,
                    min_size=config.min_size,
                    canvas_size=config.canvas_size,
                )
                if ocr_blocks:
                    # Filter out text blocks detected inside image regions
                    if result.image_bboxes:
                        before_count = len(ocr_blocks)
                        ocr_blocks = filter_image_text(
                            ocr_blocks, result.image_bboxes, result.source_dpi,
                            page_width=result.width, page_height=result.height
                        )
                        filtered_count = before_count - len(ocr_blocks)
                        if filtered_count > 0 and progress:
                            tqdm.write(f"  [FILTER] Page {pn}: removed {filtered_count} text blocks inside images")
                    result.blocks = ocr_blocks
            except Exception as e:
                tqdm.write(f"  [WARN] OCR failed on page {pn}: {e}")
            result.image = None

        algorithm = _get_cluster_algorithm(config)
        is_ocr_coords = any(b.source == "ocr" for b in result.blocks)
        if is_ocr_coords:
            dpi = result.source_dpi
            pw = result.width * dpi / 72
            ph = result.height * dpi / 72
        else:
            pw = result.width
            ph = result.height
        result.paragraphs = algorithm.build_clusters(
            result.blocks,
            page_width=pw,
            page_height=ph,
        )

        if progress:
            tqdm.write(f"  [OCR] Page {pn}: {len(result.blocks)} blocks, {len(result.paragraphs)} paragraphs")

    return pages


def run_translation_pipeline(
    pages: list[PageResult],
    config: Config,
    progress: bool = True,
) -> list[PageResult]:
    all_paragraphs = []
    for p in pages:
        all_paragraphs.extend(p.paragraphs)

    if not all_paragraphs:
        tqdm.write("No paragraphs to translate.")
        return pages

    # Build clusters per page (1-to-1 mapping with logical paragraphs)
    for p in pages:
        all_clusters = cluster_paragraphs(
            p.paragraphs,
            line_spacing_ratio=config.line_spacing_ratio,
            column_threshold=config.column_gap,
        )
        p.clusters = all_clusters

    clustered_ids = set()
    for p in pages:
        for c in p.clusters:
            for para in c.paragraphs:
                clustered_ids.add(id(para))

    cache = TranslationCache(config.cache_db)
    translator = create_translator(config, cache=cache)

    # Translate each cluster as a whole paragraph
    cluster_count = 0
    for p in pages:
        for cluster in p.clusters:
            text = cluster.text
            if text.strip():
                cluster_count += 1
                cached = cache.get(text)
                if cached and not _is_identity_failure(text, cached):
                    cluster.translation = cached
                else:
                    translation = translator.translate_paragraph(text)
                    if translation.strip() and not _is_identity_failure(text, translation):
                        cluster.translation = translation
                        cache.put(text, translation)

    if progress and cluster_count:
        tqdm.write(f"Translated {cluster_count} clusters")

    # Fallback: any cluster whose translation is empty — extract its paragraphs
    # so they get translated line-by-line instead of going blank.
    for p in pages:
        for c in p.clusters:
            if not c.translation:
                for para in c.paragraphs:
                    clustered_ids.discard(id(para))

    # Standalone paragraphs (not in any cluster) — existing line-by-line logic
    standalone_paragraphs = [para for para in all_paragraphs if id(para) not in clustered_ids]

    if standalone_paragraphs:
        all_lines: list[tuple[int, object, int, str]] = []
        gid = 0
        for para in standalone_paragraphs:
            for li, (_, text, _) in enumerate(para.line_bboxes or []):
                if text.strip():
                    all_lines.append((gid, para, li, text))
                    gid += 1

        if all_lines:
            chunk_size = 50
            chunks: list[list[tuple[int, object, int, str]]] = []
            for i in range(0, len(all_lines), chunk_size):
                chunks.append(all_lines[i:i + chunk_size])

            if progress:
                tqdm.write(f"Translating {len(chunks)} chunks ({len(all_lines)} standalone lines)")

            for ci, chunk in enumerate(chunks):
                items = [(gid, text) for gid, _, _, text in chunk]
                result = translator.translate_json_batch(items)

                if progress:
                    n_ok = 0
                    for gid, _, _, orig in chunk:
                        trans = result.get(gid, "")
                        if trans.strip() and _is_chinese_translation(trans, orig):
                            n_ok += 1
                    tqdm.write(f"  Chunk {ci+1}/{len(chunks)}: {n_ok}/{len(chunk)} lines OK")

                for gid, para, li, _ in chunk:
                    if not para.translated_lines:
                        para.translated_lines = [""] * len(para.line_bboxes)
                    para.translated_lines[li] = result.get(gid, "")

    cache.close()

    # Set para.translation for backward compat
    for para in all_paragraphs:
        if para.translated_lines:
            para.translation = "\n".join(para.translated_lines)

    if progress:
        for p in pages:
            n_total = len(p.paragraphs)
            n_done = 0
            n_bad = 0
            for para in p.paragraphs:
                if not para.translated_lines:
                    continue
                for li, (_, orig, _) in enumerate(para.line_bboxes or []):
                    if not orig.strip():
                        continue
                    tline = para.translated_lines[li] if li < len(para.translated_lines) else ""
                    if tline.strip() and _is_chinese_translation(tline, orig) and not _is_identity_failure(orig, tline):
                        n_done += 1
                    else:
                        n_bad += 1
            tqdm.write(f"  [TRANSLATE] Page {p.page_num}: {n_done}/{n_done+n_bad} lines OK (non-cn: {n_bad})")

    return pages


def _load_pages_from_db(config: Config) -> list[PageResult]:
    total_start = config.page_start or 1
    total_end = config.page_end or db.get_max_page()
    ocr_data = db.load_translate_batch(total_start, total_end)
    return [
        PageResult(page_num=n, width=w, height=h, source_dpi=dpi, paragraphs=paras)
        for n, w, h, dpi, paras in ocr_data
    ]


def run_ocr_stage(
    path: str,
    config: Config,
    output_path: str | None = None,
    progress: bool = True,
) -> str | None:
    total_start = config.page_start or 1
    total_end = config.page_end or 9999
    batch_size = config.batch_pages

    all_pages: list[PageResult] = []

    for batch_start in range(total_start, total_end + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, total_end)

        pages_to_do = [p for p in range(batch_start, batch_end + 1) if not db.page_exists(p)]
        if not pages_to_do:
            if progress:
                tqdm.write(f"  [SKIP] pages {batch_start}-{batch_end}: already OCR'd")
            continue

        if progress:
            tqdm.write(f"\n=== OCR Stage: pages {batch_start}-{batch_end} ===")

        saved_start = config.page_start
        saved_end = config.page_end
        config.page_start = batch_start
        config.page_end = batch_end

        try:
            batch_pages = run_extraction_pipeline(path, config, progress=progress)
            for pr in batch_pages:
                if pr.paragraphs:
                    db.save_page_ocr(pr.page_num, pr.width, pr.height, pr.source_dpi, pr.paragraphs)
            all_pages.extend(batch_pages)
        finally:
            config.page_start = saved_start
            config.page_end = saved_end

    if progress:
        total = db.get_max_page()
        tqdm.write(f"\n  [OCR] Stage complete: {total} pages saved to {db.OCR_DB_PATH}")

    if output_path:
        render_pages = all_pages if all_pages else _load_pages_from_db(config)
        if render_pages:
            for pr in render_pages:
                pr.clusters = cluster_paragraphs(
                    pr.paragraphs,
                    line_spacing_ratio=config.line_spacing_ratio,
                    column_threshold=config.column_gap,
                )
            from .renderer import render_pdf
            render_pdf(path, output_path, render_pages, debug_only=True)
            if progress:
                tqdm.write(f"  Debug PDF saved to {output_path}")
            return output_path
    return None


def run_translate_render_stage(
    path: str,
    output_path: str,
    config: Config,
    progress: bool = True,
) -> str:
    total_start = config.page_start or 1
    total_end = config.page_end or db.get_max_page()
    translate_batch = config.translate_batch_pages

    src_doc = fitz.open(path)
    out_doc = fitz.open()
    fontname, fontbuffer = _get_cjk_font()

    try:
        for batch_start in range(total_start, total_end + 1, translate_batch):
            batch_end = min(batch_start + translate_batch - 1, total_end)

            ocr_data = db.load_translate_batch(batch_start, batch_end)
            if not ocr_data:
                if progress:
                    tqdm.write(f"  [SKIP] pages {batch_start}-{batch_end}: no OCR data")
                continue

            if progress:
                tqdm.write(f"\n=== Translate+Render: pages {batch_start}-{batch_end} ===")

            pages = [
                PageResult(page_num=n, width=w, height=h, source_dpi=dpi, paragraphs=paras)
                for n, w, h, dpi, paras in ocr_data
            ]

            t_start = time.time()
            pages = run_translation_pipeline(pages, config, progress=progress)

            cache = TranslationCache(config.cache_db)
            translator = create_translator(config, cache=cache)

            # Retry empty/identity lines via JSON batch
            for attempt in range(3):
                failed_items: list[tuple[int, object, int, str]] = []
                gid = 0
                for p in pages:
                    for para in p.paragraphs:
                        if not para.line_bboxes:
                            continue
                        if not para.translated_lines:
                            para.translated_lines = [""] * len(para.line_bboxes)
                        for li, (_, text, _) in enumerate(para.line_bboxes):
                            if not text.strip():
                                continue
                            tline = para.translated_lines[li]
                            if not tline.strip() or _is_identity_failure(text, tline) or not _is_chinese_translation(tline, text):
                                failed_items.append((gid, para, li, text))
                                gid += 1
                if not failed_items:
                    break
                if progress:
                    tqdm.write(f"  [RETRY] {len(failed_items)} lines (attempt {attempt+1}/3)")

                # Chunk retry items to avoid Ollama timeout on large batches
                retry_chunk_size = 50
                retry_chunks = [failed_items[i:i+retry_chunk_size] for i in range(0, len(failed_items), retry_chunk_size)]
                any_ok = False
                for rci, rchunk in enumerate(retry_chunks):
                    items = [(gid, text) for gid, _, _, text in rchunk]
                    try:
                        results = translator.translate_json_batch(items)
                        for gid, para, li, orig in rchunk:
                            trans = results.get(gid, "")
                            if trans.strip() and not _is_identity_failure(orig, trans) and _is_chinese_translation(trans, orig):
                                para.translated_lines[li] = trans
                                any_ok = True
                    except Exception as e:
                        if progress:
                            tqdm.write(f"  [WARN] retry sub-chunk {rci+1}/{len(retry_chunks)} failed: {e}")
                        break
                if not any_ok:
                    break

            # Rebuild para.translation from potentially updated translated_lines
            for p in pages:
                for para in p.paragraphs:
                    if para.translated_lines:
                        para.translation = "\n".join(para.translated_lines)
            cache.close()

            for p in pages:
                dpi = p.source_dpi
                for para in p.paragraphs:
                    _split_translation(para, dpi)

            _save_debug_json(pages, f"debug_batch_{batch_start:04d}-{batch_end:04d}.json")

            for p in pages:
                dpi = p.source_dpi
                for para in p.paragraphs:
                    _split_translation(para, dpi)

            _save_debug_json(pages, f"debug_batch_{batch_start:04d}-{batch_end:04d}.json")

            for p in pages:
                db.save_translations(p.page_num, p.paragraphs)

            if progress:
                tqdm.write(f"  [BATCH] Translated ({time.time() - t_start:.0f}s)")

            for pr in pages:
                render_single_page_to_doc(src_doc, out_doc, pr, fontname, fontbuffer, disable_inpaint=True)

            for p in pages:
                db.mark_page_translated(p.page_num)

    finally:
        out_doc.save(output_path, garbage=4, deflate=True)
        out_doc.close()
        src_doc.close()

    if progress:
        tqdm.write(f"\nDone: {output_path}")
    return output_path


def run_full_pipeline_with_output(
    path: str,
    output_path: str,
    config: Config,
    progress: bool = True,
) -> str:
    cleanup_artifacts()
    if config.ocr_only:
        total_start = config.page_start or 1
        total_end = config.page_end or 9999
        batch_size = config.batch_pages
        all_pages: list[PageResult] = []

        for batch_start in range(total_start, total_end + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_end)

            saved_start = config.page_start
            saved_end = config.page_end
            config.page_start = batch_start
            config.page_end = batch_end

            try:
                batch_pages = run_extraction_pipeline(path, config, progress=progress)
                for p in batch_pages:
                    for para in p.paragraphs:
                        para.translated_lines = [t for _, t, _ in para.line_bboxes] if para.line_bboxes else []
                all_pages.extend(batch_pages)
            finally:
                config.page_start = saved_start
                config.page_end = saved_end

        render_pdf(path, output_path, all_pages, disable_inpaint=True)
        if progress:
            tqdm.write(f"\nDone: {output_path}")
        return output_path

    run_ocr_stage(path, config, progress=progress)
    return run_translate_render_stage(path, output_path, config, progress=progress)
