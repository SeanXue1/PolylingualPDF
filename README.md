# PolylingualPDF 📖 🌐

> **Layout-aware PDF translation engine:** OCR scanned magazines, cluster text blocks into natural paragraphs, translate with high contextual accuracy, and re-render back into native layouts.

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)

---

## 💡 Overview

Translating scanned magazines, comics, or double-column documents is notoriously difficult. Standard translation engines strip away page geometry, while basic OCR tools break vertical Japanese/Chinese text into scattered, disjointed lines.

**PolylingualPDF** solves this end-to-end:
1. **OCR Extraction:** Dual-engine OCR (EasyOCR + PaddleOCR) running in parallel for high-precision character recognition, results merged via IoU deduplication.
2. **Spatial Paragraph Clustering:** Advanced layout analysis — XY-Cut segmentation, line detection, paragraph detection, layout classification, and reading order reconstruction — to group fragmented bounding boxes into logical paragraphs.
3. **Contextual Translation:** Translates clustered paragraphs via Ollama (local), OpenRouter, or Gemini with JSON batch API for reliable structured output.
4. **Layout-Aware Typesetting:** Re-renders translated text onto the original document dimensions with automatic font fitting, white rect overlay or OpenCV inpainting, and CJK font insertion.

---

## ✨ Features

- 📑 **Layout & Geometry Preservation:** Maintains multi-column, sidebar, and image-wrapped text positioning.
- 📐 **Smart Text Block Clustering:** XY-Cut layout segmentation + line/paragraph detection prevents cut-off sentences from line-by-line OCR.
- 🔄 **Dual OCR Engines:** EasyOCR and PaddleOCR run concurrently; results automatically merged with IoU-based dedup for maximum coverage.
- 🔌 **Pluggable Translation:** Works with Ollama (fully local), OpenRouter, or Gemini as translation backends.
- ♻️ **Resumable Two-Stage Pipeline:** OCR and Translate+Render stages run independently; re-running skips already-processed pages via SQLite status tracking.
- 🖨️ **Debug Output:** OCR stage produces a PDF with red line bounding boxes and blue cluster borders for visual verification — no text overlay or erasure.
- 🎛️ **Config-Driven Pipeline:** Pipeline stage (`ocr`, `translate`, `full`), DPI, batch sizes, clustering algorithm, and all parameters configurable via `config.yaml`.

---

## 🛠️ Architecture

```
PDF
  │  PyMuPDF
  ▼
extractor.py   → TextBlock[] (native text) or page image
  │
  ▼
ocr.py         → EasyOCR + PaddleOCR (parallel, IoU-merged)
  │  PaddleOCR runs as subprocess (_paddle_worker.py)
  ▼
cluster/       → XY-Cut → line detection → paragraph detection
  │              → layout classification → reading order
  │              (or simple column-based clustering via merger.py)
  ▼
pipeline.py    → Chunk builder → translator.py
  │
  ▼
translator.py  → Ollama / OpenRouter / Gemini (JSON batch API)
  │  TranslationCache (SQLite, SHA-256 keyed)
  ▼
renderer.py    → White rect / OpenCV inpaint → CJK font insertion
  │  Debug mode: red line bboxes + blue cluster borders
  ▼
output.pdf
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/SeanXue1/PolylingualPDF.git
cd PolylingualPDF

pip install -r requirements.txt
```

### 2. Basic Usage

Full pipeline (OCR → translate → render):
```bash
python -m src.main input.pdf -o output.pdf
```

Stage-by-stage (config-driven, set `stage: ocr` in `config.yaml`):
```bash
python -m src.main input.pdf --page-start 1 --page-end 15
# Produces input_debug.pdf with red line bboxes + blue cluster borders
```

### 3. Two-Stage Pipeline

OCR only:
```bash
python -m src.main input.pdf --stage ocr -o debug.pdf
```

Translate + Render (uses OCR data from `ocr_cache.db`):
```bash
python -m src.main input.pdf -o output.pdf --stage translate
```

---

## ⚙️ Configuration

Copy `config.template.yaml` to `config.yaml`:

```yaml
engine: ollama                    # Translation engine: ollama, openrouter, gemini
model: qwen3:8b                   # Ollama model name
stage: full                       # Pipeline stage: ocr, translate, full
extract_dpi: 600                  # DPI for page rasterization
batch_pages: 10                   # OCR batch size
ocr_engine: both                  # OCR: easyocr, paddle, or both (parallel)
cluster_algorithm: layout         # Clustering: simple or layout (XY-Cut)
```

Full config reference: see `config.template.yaml`.

---

## 📄 License

This project is licensed under the **Sustainable Use License v1.0**.

- **Free for:** Personal use, academic research, internal testing, and non-commercial open-source development.
- **SaaS / Commercial Restriction:** You **may not** host or offer `PolylingualPDF` as a paid cloud service, SaaS, or commercial API endpoint to third parties without prior authorization.
- **Enterprise Licensing:** For commercial hosting permissions, enterprise integration, or custom licensing, please open an issue or contact xuesing@hotmail.com.

See the full [LICENSE](./LICENSE) file for complete legal terms.

---

## 🤝 Contributing

Contributions are very welcome! Please feel free to open an issue or submit a Pull Request.

1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⭐ Star History

If you find this project helpful in your document translation workflow, please consider giving it a star!
