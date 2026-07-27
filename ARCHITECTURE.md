# ARCHITECTURE

## Pipeline

```
PDF
  │  PyMuPDF (fitz)
  ▼
extractor.py
  │  Native text extraction (dict mode) OR page rasterization @ configurable DPI
  ▼
ocr.py
  │  EasyOCR + PaddleOCR (parallel, results merged via IoU dedup at 0.3 threshold)
  │  PaddleOCR runs in a subprocess (_paddle_worker.py) to avoid CUDA DLL conflicts
  ▼
cluster/ (layout algorithm)
  │  XY-Cut segmentation → line detection → paragraph detection
  │  → layout classification → reading order reconstruction
  │  Falls back to simple column-based clustering (merger.py) if cluster_algorithm = "simple"
  ▼
pipeline.py
  │  Chunk builder (configurable min/max tokens) → translator.py
  ▼
translator.py
  │  Ollama, OpenRouter, or Gemini (JSON batch API, cache via TranslationCache)
  ▼
renderer.py
  │  Cluster-based translation rendering + line-by-line fallback
  │  White rect overlay or OpenCV TELEA inpainting → CJK font insertion
  │  Debug mode (debug_only) draws red line bboxes + blue cluster borders without text
  ▼
output.pdf
```

## Two-stage execution

Controlled by `config.stage` (or CLI `--stage`):

| Stage | Action |
|-------|--------|
| `ocr` | Extract → OCR → cluster → save to `ocr_cache.db` + optional debug PDF |
| `translate` | Load from `ocr_cache.db` → translate → render → output PDF |
| `full` | Both stages sequentially |

OCR and translate stages are independently resumable: already-processed pages are skipped.

## Debug output

- `*_debug.pdf` — rendered by OCR stage when `--output` given (original page + red line bboxes + blue cluster borders, no text overlay)
- `debug_batch_XXXX-XXXX.json` — per-batch JSON dump during translate stage (source text, translation, line bboxes)

## Thread model

- **OCR engines**: EasyOCR (in-process) + PaddleOCR (subprocess) run concurrently via `ThreadPoolExecutor`
- **Translation**: JSON batch API calls (sequential per batch, configurable retries)
- **Rendering**: Single-threaded PyMuPDF page processing
- **Caches**: Thread-local SQLite connections for `translation_cache.db` (SHA-256 keyed)

## Key design decisions

1. **Parallel OCR engines** — EasyOCR + PaddleOCR run concurrently, results merged via IoU dedup
2. **PaddleOCR in subprocess** — avoids CUDA symbol conflicts between PyTorch and PaddlePaddle
3. **Layout-aware clustering** — XY-Cut + line detection + paragraph detection + layout classification + reading order
4. **JSON batch translation** — structured I/O with id-based matching, cache via SHA-256 hash
5. **Cluster-based rendering** — multi-paragraph clusters rendered as unified blocks; standalone paragraphs rendered line-by-line
6. **Two cache layers** — `translation_cache.db` (API response) + `ocr_cache.db` (per-page OCR + translations)
7. **Identity failure detection** — unchanged Japanese/Kanji text triggers retry
8. **Inpainting + white rect** — OpenCV TELEA inpainting when line bboxes available, white rect overlay otherwise
9. **Stage-by-stage recovery** — re-running a stage skips already-done pages via DB status checks
