# PRODUCT

## Goal
Translate Japanese PDF magazines (50–200MB+) into Simplified Chinese while preserving layout as much as practical.

## Scope
- Native-text PDFs + scanned PDFs
- Two-stage pipeline (OCR → Translate + Render), independently resumable
- Page-by-page processing with configurable page ranges
- Dual OCR engines: EasyOCR + PaddleOCR running in parallel (results merged)
- Layout-aware paragraph clustering (XY-Cut, line detection, layout classification, reading order)
- Translation via Ollama (local), OpenRouter, or Gemini
- Configurable pipeline stage via config.yaml (`stage: ocr | translate | full`)
- Debug output: PDF with red line bboxes + blue cluster borders, JSON debug dumps

## Out of Scope
- Perfect Adobe-quality layout
- Table reconstruction
- SaaS
- Multi-language (Japanese → Simplified Chinese only)
