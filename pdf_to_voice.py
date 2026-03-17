from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


class MissingDependency(RuntimeError):
    pass


def _require_pdf_deps():
    try:
        import pyttsx3  # type: ignore
        from PyPDF2 import PdfReader  # type: ignore

        return pyttsx3, PdfReader
    except Exception as e:  # pragma: no cover
        raise MissingDependency(
            "Missing dependencies for PDF to Voice.\n"
            "Install with:\n"
            "  python -m pip install PyPDF2 pyttsx3\n"
            "Notes:\n"
            "- On some systems, pyttsx3 may need additional OS TTS components."
        ) from e


def _pick_pdf_via_dialog() -> str | None:
    try:
        from tkinter.filedialog import askopenfilename
    except Exception:
        return None
    return askopenfilename(filetypes=[("PDF files", "*.pdf")]) or None


def _default_state_path(pdf_path: Path) -> Path:
    # Stored next to the PDF by default; use --state to customize.
    return pdf_path.with_name(f"{pdf_path.name}.pdf_to_voice_progress.json")


def _pdf_fingerprint(pdf_path: Path) -> dict[str, object]:
    st = pdf_path.stat()
    return {
        "pdf_path": str(pdf_path),
        "pdf_size": int(st.st_size),
        "pdf_mtime": float(st.st_mtime),
    }


def _load_state(state_path: Path) -> dict[str, object] | None:
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _parse_pages_spec(spec: str, total_pages: int) -> list[int]:
    """
    Parse a pages spec like:
      - "10-30"
      - "1,3,5-7"
      - "10-" (from 10 to end)
    Returns 1-based page numbers, sorted and unique.
    """
    spec = spec.strip()
    if not spec:
        return list(range(1, total_pages + 1))

    pages: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            left, right = part.split("-", 1)
            left = left.strip()
            right = right.strip()
            start = int(left) if left else 1
            end = int(right) if right else total_pages
            if start < 1 or end < 1 or start > end:
                raise ValueError(f"Invalid page range: {part!r}")
            for p in range(start, min(end, total_pages) + 1):
                pages.add(p)
        else:
            p = int(part)
            if p < 1:
                raise ValueError(f"Invalid page number: {part!r}")
            if p <= total_pages:
                pages.add(p)

    return sorted(pages)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_to_voice.py",
        description="Read a PDF out loud using PyPDF2 + pyttsx3.",
    )
    p.add_argument("pdf", nargs="?", default=None, help="Path to a PDF. If omitted, opens a file picker.")
    p.add_argument(
        "--pages",
        type=str,
        default=None,
        help='Pages to read (1-based). Examples: "10-30", "1,3,5-7", "10-" (to end). Default: all pages.',
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last completed page using a progress JSON file.",
    )
    p.add_argument(
        "--state",
        type=str,
        default=None,
        help="Path to progress JSON (default: <pdf>.pdf_to_voice_progress.json next to the PDF).",
    )
    p.add_argument(
        "--reset-progress",
        action="store_true",
        help="Ignore and overwrite any existing progress state.",
    )
    p.add_argument("--rate", type=int, default=None, help="Speech rate (engine-dependent).")
    p.add_argument("--voice", type=str, default=None, help="Voice name substring to select (engine-dependent).")
    p.add_argument("--verbose", action="store_true", help="Print progress details.")
    return p


def _select_voice(engine, voice_hint: str) -> bool:
    hint = voice_hint.casefold().strip()
    if not hint:
        return False
    try:
        voices = engine.getProperty("voices") or []
    except Exception:
        return False
    for v in voices:
        name = getattr(v, "name", "") or ""
        vid = getattr(v, "id", "") or ""
        if hint in name.casefold() or hint in vid.casefold():
            try:
                engine.setProperty("voice", v.id)
                return True
            except Exception:
                return False
    return False


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    pdf_path_str = args.pdf or _pick_pdf_via_dialog()
    if not pdf_path_str:
        print("No PDF selected.", file=sys.stderr)
        return 2

    pdf_path = Path(pdf_path_str).expanduser().resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    try:
        pyttsx3, PdfReader = _require_pdf_deps()
    except MissingDependency as e:
        print(str(e), file=sys.stderr)
        return 2

    state_path = Path(args.state).expanduser().resolve() if args.state else _default_state_path(pdf_path)

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages <= 0:
        print("PDF has no pages.", file=sys.stderr)
        return 2

    try:
        selected_pages = (
            _parse_pages_spec(args.pages, total_pages) if args.pages is not None else list(range(1, total_pages + 1))
        )
    except Exception as e:
        print(f"Invalid --pages value: {e}", file=sys.stderr)
        return 2

    last_completed = 0
    if args.resume and not args.reset_progress and state_path.exists():
        prev = _load_state(state_path)
        if prev:
            if prev.get("pdf_path") == str(pdf_path):
                try:
                    last_completed = int(prev.get("last_completed_page", 0) or 0)
                except Exception:
                    last_completed = 0
            elif args.verbose:
                print("Existing progress file is for a different PDF; starting fresh.")

    pages_to_read = [p for p in selected_pages if p > last_completed]
    if not pages_to_read:
        if args.verbose:
            print("Nothing to do: already completed selected pages.")
        return 0

    engine = pyttsx3.init()
    if args.rate is not None:
        try:
            engine.setProperty("rate", int(args.rate))
        except Exception:
            pass
    if args.voice:
        ok = _select_voice(engine, args.voice)
        if args.verbose and not ok:
            print(f'No matching voice found for --voice "{args.voice}". Using default voice.')

    fingerprint = _pdf_fingerprint(pdf_path)
    progress_state: dict[str, object] = {
        **fingerprint,
        "pages_spec": args.pages,
        "total_pages": total_pages,
        "last_completed_page": last_completed,
        "updated_at": time.time(),
    }

    def checkpoint(completed_page: int) -> None:
        progress_state["last_completed_page"] = completed_page
        progress_state["updated_at"] = time.time()
        try:
            _write_state(state_path, progress_state)
        except Exception:
            pass

    try:
        for page_num in pages_to_read:
            if args.verbose:
                print(f"Reading page {page_num}/{total_pages}...")
            try:
                text = reader.pages[page_num - 1].extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                engine.say(text)
                engine.runAndWait()
            checkpoint(page_num)
    except KeyboardInterrupt:
        if args.verbose:
            print("Interrupted. Progress saved.")
        return 130

    if args.verbose:
        print(f"Done. Progress file: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
