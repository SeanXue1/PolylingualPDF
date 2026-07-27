import argparse
from pathlib import Path

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-tl",
        description="Translate Japanese PDF magazines to Simplified Chinese.",
    )
    parser.add_argument("input", help="Path to input PDF file")
    parser.add_argument("-o", "--output", default=None, help="Path to output PDF file")
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--engine",
        choices=["ollama", "openrouter", "gemini"],
        default=None,
        help="Translation engine",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (e.g. qwen3:8b)",
    )
    parser.add_argument("--page-start", type=int, default=None, help="First page")
    parser.add_argument("--page-end", type=int, default=None, help="Last page")
    parser.add_argument(
        "--no-ocr", action="store_true", help="Skip OCR for scanned pages"
    )
    parser.add_argument(
        "--batch-pages", type=int, default=None, help="Pages per batch"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Extract text without translating"
    )
    parser.add_argument(
        "--ocr-only", action="store_true", help="OCR without translation"
    )
    parser.add_argument(
        "--stage",
        choices=["ocr", "translate", "full"],
        default=None,
        help="Pipeline stage: ocr (OCR only, save to DB), translate (load+translate+render), full (default)",
    )
    parser.add_argument(
        "--translate-batch-pages", type=int, default=None, help="Pages per translate batch (default 3)"
    )
    return parser


from .pipeline import run_extraction_pipeline, run_full_pipeline_with_output, run_ocr_stage, run_translate_render_stage


def _infer_output_path(input_path: str, suffix: str = "_translated") -> str:
    p = Path(input_path)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")


def run(args: argparse.Namespace, config: Config) -> None:
    if args.dry_run:
        print(f"Input: {args.input}")
        print(f"Engine: {args.engine or config.engine}")
        print(f"Model: {args.model or config.model}")
        print(f"Pages: {config.page_start or 1} - {config.page_end or 'end'}")
        print(f"OCR: {'disabled' if config.no_ocr else 'auto'}")
        print("--- Preview ---")
        old_no_ocr = config.no_ocr
        config.no_ocr = True
        pages = run_extraction_pipeline(args.input, config)
        config.no_ocr = old_no_ocr
        for p in pages:
            src = "ocr" if p.needs_ocr else "native"
            preview_text = ""
            if p.paragraphs:
                first = p.paragraphs[0]
                preview_text = first.text[:80].replace("\n", " ")
            print(f"  Page {p.page_num}: {len(p.blocks)} blocks, {len(p.paragraphs)} paragraphs ({src})")
            if preview_text:
                print(f"    First paragraph: \"{preview_text}...\"")
        return

    import time
    stage = config.stage

    if stage == "ocr":
        output = args.output or _infer_output_path(args.input, "_debug")
        print(f"Input:   {args.input}")
        print(f"Output:  {output}")
        print(f"Stage:   OCR only (save to DB)")
        print(f"Pages:   {config.page_start or 1} - {config.page_end or 'end'}")
        eng = config.ocr_engine
        if eng == "both":
            eng = "easyocr+paddle"
        print(f"OCR:     {eng} @ {config.extract_dpi}dpi")
        print()
        t0 = time.time()
        run_ocr_stage(args.input, config, output_path=output)
        elapsed = time.time() - t0
        print(f"\nOCR stage done ({elapsed:.0f}s)")
        return

    if stage == "translate":
        output = args.output or _infer_output_path(args.input, config.output_suffix)
        print(f"Input:   {args.input}")
        print(f"Output:  {output}")
        print(f"Stage:   Translate + Render (from OCR DB)")
        print(f"Pages:   {config.page_start or 1} - {config.page_end or 'end'}")
        print(f"Engine:  {config.engine}")
        print(f"Model:   {config.model}")
        print(f"Batch:   {config.translate_batch_pages} pages")
        print(f"Chunks:  {config.chunk_min_tokens}-{config.chunk_max_tokens} tokens")
        print()
        t0 = time.time()
        run_translate_render_stage(args.input, output, config)
        elapsed = time.time() - t0
        print(f"\nDone: {output} ({elapsed:.0f}s)")
        return

    output = args.output or _infer_output_path(args.input, config.output_suffix)
    if config.ocr_only:
        print(f"Input:   {args.input}")
        print(f"Output:  {output}")
        print(f"Mode:    OCR-only (no translation)")
        eng = config.ocr_engine
        if eng == "both":
            eng = "easyocr+paddle"
        print(f"OCR:     {eng} @ {config.extract_dpi}dpi")
        print()
        t0 = time.time()
        run_full_pipeline_with_output(args.input, output, config)
        elapsed = time.time() - t0
        print(f"\nDone: {output} ({elapsed:.0f}s)")
        return

    print(f"Input:   {args.input}")
    print(f"Output:  {output}")
    print(f"Stage:   Full (OCR -> Translate -> Render)")
    print(f"Engine:  {config.engine}")
    print(f"Model:   {config.model}")
    eng = config.ocr_engine
    if eng == "both":
        eng = "easyocr+paddle"
    print(f"OCR:     {eng} @ {config.extract_dpi}dpi")
    print(f"Threads: OCR={config.threads.get('ocr',2)} x Translate={config.threads.get('translator',3)}")
    print(f"Chunks:  {config.chunk_min_tokens}-{config.chunk_max_tokens} tokens")
    print()

    t0 = time.time()
    run_full_pipeline_with_output(args.input, output, config)
    elapsed = time.time() - t0
    print(f"\nDone: {output} ({elapsed:.0f}s)")
