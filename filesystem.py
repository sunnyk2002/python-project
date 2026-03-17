from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Default extension-to-folder mapping. Extensions are lowercase, without the dot.
DEFAULT_RULES: dict[str, str] = {
    # Documents
    "pdf": "PDFs",
    "doc": "Docs",
    "docx": "Docs",
    "rtf": "Docs",
    "txt": "Text",
    "md": "Text",
    "markdown": "Text",
    "odt": "Docs",
    # Spreadsheets / presentations
    "csv": "Spreadsheets",
    "tsv": "Spreadsheets",
    "xls": "Spreadsheets",
    "xlsx": "Spreadsheets",
    "ods": "Spreadsheets",
    "ppt": "Presentations",
    "pptx": "Presentations",
    "odp": "Presentations",
    # Images
    "jpg": "Images",
    "jpeg": "Images",
    "png": "Images",
    "gif": "Images",
    "webp": "Images",
    "bmp": "Images",
    "heic": "Images",
    "tif": "Images",
    "tiff": "Images",
    "svg": "Images",
    # Video
    "mp4": "Videos",
    "mov": "Videos",
    "mkv": "Videos",
    "webm": "Videos",
    "avi": "Videos",
    "m4v": "Videos",
    # Audio
    "mp3": "Audio",
    "wav": "Audio",
    "m4a": "Audio",
    "aac": "Audio",
    "flac": "Audio",
    "ogg": "Audio",
    # Archives / disk images
    "zip": "Archives",
    "rar": "Archives",
    "7z": "Archives",
    "tar": "Archives",
    "gz": "Archives",
    "bz2": "Archives",
    "xz": "Archives",
    "dmg": "DiskImages",
    "iso": "DiskImages",
    # Code
    "py": "Code",
    "js": "Code",
    "ts": "Code",
    "tsx": "Code",
    "jsx": "Code",
    "java": "Code",
    "c": "Code",
    "h": "Code",
    "cpp": "Code",
    "hpp": "Code",
    "go": "Code",
    "rs": "Code",
    "rb": "Code",
    "php": "Code",
    "html": "Code",
    "css": "Code",
    "json": "Code",
    "yaml": "Code",
    "yml": "Code",
    "xml": "Code",
    "toml": "Code",
    "sh": "Code",
    "bat": "Code",
    "ps1": "Code",
    # Installers / executables (common on Windows/macOS)
    "exe": "Executables",
    "msi": "Executables",
    "pkg": "Installers",
    "app": "Installers",
    # Fonts
    "ttf": "Fonts",
    "otf": "Fonts",
    "woff": "Fonts",
    "woff2": "Fonts",
}


DEFAULT_OTHER_FOLDER = "Other"
DEFAULT_LOG_NAME = ".file_sorter_log.jsonl"


@dataclass(frozen=True)
class Operation:
    ts: float
    action: str  # "move" or "copy"
    src: str
    dst: str


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def _now() -> float:
    return time.time()


def load_rules(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_RULES)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Rules file must be a JSON object mapping extension -> folder.")
    rules: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("Rules JSON must map strings to strings.")
        ext = k.lower().lstrip(".")
        folder = v.strip().strip("/\\")
        if not ext or not folder:
            continue
        rules[ext] = folder
    return rules


def classify(file_path: Path, rules: dict[str, str], other_folder: str) -> str:
    # Treat dotfiles without "real" suffix as "Other" (e.g., ".bashrc").
    suffix = file_path.suffix.lower().lstrip(".")
    if not suffix:
        return other_folder
    return rules.get(suffix, other_folder)


def iter_candidate_files(
    root: Path,
    *,
    recursive: bool,
    include_hidden: bool,
    excluded_dirs: set[Path],
) -> Iterable[Path]:
    if recursive:
        walker = root.rglob("*")
    else:
        walker = root.iterdir()

    for p in walker:
        try:
            if p.is_dir():
                continue
            if not include_hidden and _is_hidden(p):
                continue
            # Avoid operating on files inside excluded dirs (only meaningful in recursive mode).
            if recursive:
                for ex in excluded_dirs:
                    try:
                        p.relative_to(ex)
                        break
                    except ValueError:
                        pass
                else:
                    yield p
            else:
                yield p
        except OSError:
            # Permission errors / broken symlinks etc.
            continue


def ensure_dir(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def unique_destination(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def is_recent(file_path: Path, *, min_age_seconds: int) -> bool:
    if min_age_seconds <= 0:
        return False
    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        return True
    return (_now() - mtime) < min_age_seconds


def append_log(log_path: Path, op: Operation, *, dry_run: bool) -> None:
    if dry_run:
        return
    line = json.dumps(op.__dict__, ensure_ascii=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def move_or_copy(
    src: Path,
    dst: Path,
    *,
    action: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    if action == "move":
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return
    if action == "copy":
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return
    raise ValueError(f"Unknown action: {action}")


def sort_once(
    root: Path,
    *,
    rules: dict[str, str],
    other_folder: str,
    dry_run: bool,
    recursive: bool,
    include_hidden: bool,
    action: str,
    min_age_seconds: int,
    log_path: Path,
    verbose: bool,
) -> int:
    root = root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root folder does not exist or is not a directory: {root}")

    destination_dirs = {root / folder for folder in set(rules.values()) | {other_folder}}
    excluded_dirs = set(destination_dirs)

    moved = 0
    for src in iter_candidate_files(
        root,
        recursive=recursive,
        include_hidden=include_hidden,
        excluded_dirs=excluded_dirs,
    ):
        if is_recent(src, min_age_seconds=min_age_seconds):
            if verbose:
                print(f"skip (recent): {src}")
            continue

        bucket = classify(src, rules, other_folder)
        dst_dir = root / bucket
        dst = dst_dir / src.name

        # If already in the right destination folder, skip.
        if src.parent.resolve() == dst_dir.resolve():
            continue

        ensure_dir(dst_dir, dry_run=dry_run)
        final_dst = unique_destination(dst)
        if verbose or dry_run:
            prefix = "DRY-RUN" if dry_run else action.upper()
            print(f"{prefix}: {src} -> {final_dst}")

        move_or_copy(src, final_dst, action=action, dry_run=dry_run)
        append_log(
            log_path,
            Operation(ts=_now(), action=action, src=str(src), dst=str(final_dst)),
            dry_run=dry_run,
        )
        moved += 1
    return moved


def undo_moves(log_path: Path, *, dry_run: bool, verbose: bool) -> int:
    log_path = log_path.expanduser().resolve()
    if not log_path.exists():
        raise ValueError(f"Log file not found: {log_path}")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    ops: list[Operation] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            ops.append(
                Operation(
                    ts=float(data.get("ts", 0.0)),
                    action=str(data.get("action", "")),
                    src=str(data.get("src", "")),
                    dst=str(data.get("dst", "")),
                )
            )
        except Exception:
            continue

    # Undo in reverse order. Only undo moves (copies are left in place).
    undone = 0
    for op in reversed(ops):
        if op.action != "move":
            continue
        dst = Path(op.dst)
        src = Path(op.src)
        if not dst.exists():
            if verbose:
                print(f"skip (missing dst): {dst}")
            continue

        src.parent.mkdir(parents=True, exist_ok=True)
        final_src = unique_destination(src) if src.exists() else src
        if verbose or dry_run:
            prefix = "DRY-RUN" if dry_run else "UNDO"
            print(f"{prefix}: {dst} -> {final_src}")
        if not dry_run:
            shutil.move(str(dst), str(final_src))
        undone += 1

    return undone


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="filesystem.py",
        description="Smart File Sorter: organize a folder by moving/copying files into subfolders by extension.",
    )
    p.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Folder to organize (default: current directory). Example: ~/Downloads",
    )
    p.add_argument(
        "--rules",
        type=str,
        default=None,
        help="Path to JSON file mapping extension -> folder name.",
    )
    p.add_argument(
        "--other-folder",
        type=str,
        default=DEFAULT_OTHER_FOLDER,
        help=f'Folder for unknown/no-extension files (default: "{DEFAULT_OTHER_FOLDER}").',
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions but do not move/copy anything.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Also sort files inside subfolders (destination folders are excluded).",
    )
    p.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files (dotfiles).",
    )
    p.add_argument(
        "--action",
        choices=["move", "copy"],
        default="move",
        help="Whether to move or copy files (default: move).",
    )
    p.add_argument(
        "--min-age-seconds",
        type=int,
        default=10,
        help="Skip files modified in the last N seconds (default: 10). Helps avoid partial downloads.",
    )
    p.add_argument(
        "--log",
        type=str,
        default=None,
        help=f'Operation log for undo (default: "{DEFAULT_LOG_NAME}" inside the target folder).',
    )
    p.add_argument(
        "--undo",
        type=str,
        default=None,
        help="Undo moves using a JSONL log file (use the same log path as a previous run).",
    )
    p.add_argument(
        "--watch",
        type=int,
        default=0,
        help="Re-run the sorter every N seconds (0 disables).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra details (skips, missing files, etc.).",
    )
    return p


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.undo is not None:
        undone = undo_moves(Path(args.undo), dry_run=args.dry_run, verbose=args.verbose)
        if args.verbose or args.dry_run:
            print(f"undone: {undone}")
        return 0

    root = Path(args.path)
    rules = load_rules(Path(args.rules)) if args.rules else load_rules(None)
    other_folder = str(args.other_folder).strip().strip("/\\") or DEFAULT_OTHER_FOLDER

    root_resolved = root.expanduser().resolve()
    log_path = Path(args.log) if args.log else (root_resolved / DEFAULT_LOG_NAME)

    def run_once() -> None:
        moved = sort_once(
            root,
            rules=rules,
            other_folder=other_folder,
            dry_run=args.dry_run,
            recursive=args.recursive,
            include_hidden=args.include_hidden,
            action=args.action,
            min_age_seconds=args.min_age_seconds,
            log_path=log_path,
            verbose=args.verbose,
        )
        if args.verbose or args.dry_run:
            print(f"sorted: {moved}")

    run_once()
    if args.watch and args.watch > 0:
        if args.verbose:
            print(f"watching: every {args.watch}s (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(args.watch)
                run_once()
        except KeyboardInterrupt:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
