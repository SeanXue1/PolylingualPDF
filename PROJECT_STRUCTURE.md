# AI PDF Japanese Translator — Project Structure

## Overview
Translate Japanese PDF magazines → Simplified Chinese. Preserves layout. Supports native-text and scanned PDFs.
Two-stage pipeline (OCR → Translate+Render), independently resumable. Dual OCR engines (EasyOCR + PaddleOCR in parallel).

## Pipeline (data flow)
```
PDF
  │  PyMuPDF
  ▼
extractor.py   → TextBlock[] (native text) or page image (for OCR)
  │
  ▼
ocr.py         → EasyOcrEngine + PaddleOcrEngine (parallel, IoU-merged)
  │  PaddleOCR runs as subprocess (_paddle_worker.py)
  ▼
cluster/       → Layout-aware clustering pipeline:
  │  XY-Cut → line detection → paragraph detection
  │  → layout classification → reading order
  │  Or: simple column-based clustering (merger.py)
  ▼
pipeline.py    → Build chunks → translator.py
  │  Ollama / OpenRouter / Gemini (JSON batch API)
  ▼
renderer.py    → Cluster rendering + line-by-line fallback
  │  White rect / OpenCV inpaint → CJK font insertion
  │  Debug mode: red line bboxes + blue cluster borders
  ▼
output.pdf
```

## Directory Layout
```
pdfTranslator/
├── config.yaml              # User config (gitignored)
├── config.template.yaml     # Template with all fields documented
├── requirements.txt         # Python dependencies
├── reset_status.py          # Utility: reset DB pages to ocr_done
├── README.md                # Quick-start guide
├── ARCHITECTURE.md          # Pipeline flow + design decisions
├── PRODUCT.md               # Goals & scope
├── PROJECT_STRUCTURE.md     # This file
│
├── src/
│   ├── main.py              # Entry: argparse → Config.load → cli.run
│   ├── cli.py               # Argparse: --stage, -o, --page-start, --page-end, --ocr-only, --dry-run
│   ├── config.py            # Config dataclass (40+ fields, loaded from YAML)
│   ├── pipeline.py          # Orchestrator: run_ocr_stage / run_translate_render_stage / run_full_pipeline_with_output
│   ├── extractor.py         # PDF open → native text blocks / page image extraction
│   ├── ocr.py               # EasyOcrEngine / PaddleOcrEngine + parallel both-engine merge
│   ├── _paddle_worker.py    # Standalone subprocess for PaddleOCR (avoids CUDA DLL conflict)
│   ├── merger.py            # Column clustering + vertical line merging (simple algorithm)
│   ├── translator.py        # OllamaTranslator / OpenRouterTranslator / GeminiTranslator + JSON batch API
│   ├── cache.py             # TranslationCache (SQLite, SHA-256 keyed, thread-local conn)
│   ├── db.py                # ocr_cache.db: save/load OCR + translations per page
│   ├── renderer.py          # Cluster render + line-by-line render, debug_only mode (red/blue boxes)
│   ├── inpaint.py           # OpenCV TELEA inpainting over text regions
│   ├── models.py            # TextBlock, Paragraph, TextCluster, PageResult dataclasses
│   │
│   └── cluster/             # Layout clustering algorithms
│       ├── __init__.py
│       ├── base_cluster.py       # Abstract base: BaseClusterAlgorithm
│       ├── simple_cluster.py     # SimpleClusterAlgorithm (delegates to merger.py)
│       ├── layout_cluster.py     # LayoutClusterAlgorithm (delegates to cluster_builder.py)
│       ├── cluster_builder.py    # Full pipeline: XY-Cut → line detect → paragraph detect → classify → reading order
│       ├── xycut.py              # Recursive XY-Cut layout segmentation
│       ├── line_detector.py      # Text line detection from bounding boxes
│       ├── paragraph_detector.py # Paragraph boundary detection
│       ├── layout_classifier.py  # Layout type classifier (body, heading, etc.)
│       ├── reading_order.py      # Reading order reconstruction
│       └── geometry.py           # Geometric utility functions
│
├── tests/
│   └── test_merger.py       # Unit tests for merger
│
└── *.db                     # SQLite databases (gitignored)
```

## Data Models (`models.py`)
```python
TextBlock(bbox, text, source, page_num, block_type, font_size)
Paragraph(blocks, translation, line_bboxes, translated_lines, layout_type)
  .text  -> concatenated block text
  .bbox  -> union of block bboxes
TextCluster(paragraphs, translation)
  .bbox          -> union of all line bboxes
  .text          -> concatenated paragraph texts
  .all_line_bboxes -> flattened line bboxes from all paragraphs
PageResult(page_num, blocks, paragraphs, clusters, width, height, needs_ocr, image, source_dpi)
```

## Config (`config.py` — dataclass, loaded from YAML)

### Core
| Field | Default | Purpose |
|-------|---------|---------|
| `engine` | `ollama` | `ollama`, `openrouter`, or `gemini` |
| `model` | `qwen3:8b` | Ollama model name |
| `ollama_url` | `http://localhost:11434` | Ollama server URL |
| `openrouter_api_key` | `""` | OpenRouter API key |
| `openrouter_model` | `openai/gpt-4o-mini` | OpenRouter model |
| `gemini_api_key` | `""` | Gemini API key |
| `gemini_model` | `gemini-2.0-flash` | Gemini model |

### Pipeline control
| Field | Default | Purpose |
|-------|---------|---------|
| `stage` | `full` | Pipeline stage: `ocr`, `translate`, or `full` |
| `batch_pages` | `10` | OCR batch size |
| `translate_batch_pages` | `3` | Translate+render batch size |
| `page_start` | unset | First page to process |
| `page_end` | unset | Last page to process |
| `no_ocr` | `false` | Skip OCR for scanned pages |
| `ocr_only` | `false` | Legacy flag: OCR + render without translation |

### OCR
| Field | Default | Purpose |
|-------|---------|---------|
| `ocr_engine` | `both` | `easyocr`, `paddle`, or `both` (parallel+merge) |
| `ocr_lang` | `japan` | OCR language code |
| `extract_dpi` | `300` | DPI for page rasterization (higher = better OCR, slower) |
| `ocr_confidence` | `0.1` | Minimum confidence threshold |
| `text_threshold` | `0.7` | EasyOCR text confidence threshold |
| `low_text` | `0.4` | EasyOCR low-text threshold |
| `min_size` | `10` | EasyOCR minimum text size (px) |
| `canvas_size` | `4096` | EasyOCR canvas size |
| `use_gpu` | `true` | Enable GPU for OCR |

### Clustering
| Field | Default | Purpose |
|-------|---------|---------|
| `cluster_algorithm` | `simple` | `simple` (column-based) or `layout` (XY-Cut pipeline) |
| `line_spacing_ratio` | `1.5` | Vertical gap ratio for merging lines into paragraphs |
| `column_gap` | `50.0` | Horizontal gap threshold for column detection (px) |

### Translation
| Field | Default | Purpose |
|-------|---------|---------|
| `chunk_min_tokens` | `400` | Min tokens per translation chunk |
| `chunk_max_tokens` | `600` | Max tokens per translation chunk |
| `chunk_max_items` | `15` | Max items per translation chunk |
| `cache_db` | `translation_cache.db` | Path to translation cache DB |

### Output
| Field | Default | Purpose |
|-------|---------|---------|
| `output_suffix` | `_translated` | Suffix for auto-generated output filenames |
| `threads.ocr` | `2` | OCR worker threads |
| `threads.translator` | `1` | Translator worker threads |

## CLI Usage
```bash
# Run full pipeline (entry point)
python -m src.main input.pdf -o output.pdf

# With custom config
python -m src.main input.pdf -o output.pdf -c myconfig.yaml

# --- Stage-by-stage (config-driven) ---
# Set stage: ocr in config.yaml, then:
python -m src.main input.pdf --page-start 1 --page-end 15
# Produces input_debug.pdf with red line bboxes + blue cluster borders

# --- Two-stage (CLI-driven) ---

# Stage 1: OCR only (saves to ocr_cache.db)
python -m src.main input.pdf --stage ocr

# Stage 2: Translate + Render (reads from ocr_cache.db)
python -m src.main input.pdf -o output.pdf --stage translate

# --- Partial redo ---

# Translate only pages 1-3
python -m src.main input.pdf -o test.pdf --stage translate --page-start 1 --page-end 3

# --- Debug ---

# Dry run (shows what would be processed)
python -m src.main input.pdf --dry-run

# OCR only + render (legacy, renders OCR text overlaid)
python -m src.main input.pdf --ocr-only

# --- Clean / reset ---

# Reset specific pages for re-translation
sqlite3 ocr_cache.db "UPDATE ocr_pages SET status='ocr_done',translation=NULL WHERE page_num IN (1,2,3)"

# Reset all pages to ocr_done
python reset_status.py

# Delete full OCR cache (forces full re-OCR)
rm ocr_cache.db

# Delete translation API cache (forces re-translation, not re-OCR)
rm translation_cache.db

# Delete both
rm ocr_cache.db translation_cache.db
```

## Key Design Decisions
1. **Two-stage pipeline:** OCR first (save to DB) → translate+render (incremental, crash-resilient)
2. **Config-driven stage selection:** `stage` field in config.yaml controls which part of the pipeline runs
3. **Parallel OCR engines:** EasyOCR + PaddleOCR run concurrently, results merged via IoU dedup (threshold 0.3)
4. **PaddleOCR in subprocess:** Avoids CUDA symbol conflicts between PyTorch and PaddlePaddle
5. **Layout-aware clustering:** XY-Cut → line detection → paragraph detection → layout classification → reading order (or simple column-based fallback)
6. **JSON batch translation:** Items grouped in chunks, sent as structured JSON for reliable parsing
7. **Cluster-based rendering:** Multi-paragraph clusters rendered as unified blocks with white rect + text; standalone paragraphs rendered line-by-line
8. **Debug render mode:** `debug_only` produces original page image + red line bboxes + blue cluster borders without any text overlay or erasure
9. **Cache layers:** `translation_cache.db` (API response, SHA-256 keyed) + `ocr_cache.db` (per-page OCR+translation)
10. **Identity failure detection:** If translation == original text AND text contains Japanese/Kanji → retry
11. **Inpainting + white rect:** Two modes: OpenCV inpaint (when line bboxes available) or white rect overlay
12. **Recovery by re-run:** Already-processed pages skipped via DB status checks (`ocr_done`/`translated`)
