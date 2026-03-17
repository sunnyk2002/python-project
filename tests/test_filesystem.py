from __future__ import annotations

import json
from pathlib import Path

import pytest

import filesystem


def test_load_rules_none_returns_defaults() -> None:
    rules = filesystem.load_rules(None)
    assert rules["pdf"] == "PDFs"


def test_load_rules_normalizes_extensions_and_folders(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({".TXT": " Text/ ", "Md": "Notes"}), encoding="utf-8")
    rules = filesystem.load_rules(rules_path)
    assert rules["txt"] == "Text"
    assert rules["md"] == "Notes"


def test_classify_dotfile_without_suffix_goes_to_other() -> None:
    rules = filesystem.load_rules(None)
    other = "Other"
    assert filesystem.classify(Path(".bashrc"), rules, other) == other


def test_unique_destination_adds_counter(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    dest = filesystem.unique_destination(target)
    assert dest.name == "file (1).txt"


def test_sort_once_moves_and_logs(tmp_path: Path) -> None:
    root = tmp_path
    (root / "a.pdf").write_text("pdf", encoding="utf-8")
    (root / "b.unknown").write_text("x", encoding="utf-8")

    log_path = root / "ops.jsonl"
    moved = filesystem.sort_once(
        root,
        rules={"pdf": "PDFs"},
        other_folder="Other",
        dry_run=False,
        recursive=False,
        include_hidden=False,
        action="move",
        min_age_seconds=0,
        log_path=log_path,
        verbose=False,
    )
    assert moved == 2
    assert (root / "PDFs" / "a.pdf").exists()
    assert (root / "Other" / "b.unknown").exists()
    assert log_path.exists()
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2


def test_undo_moves_restores_files(tmp_path: Path) -> None:
    root = tmp_path
    (root / "a.pdf").write_text("pdf", encoding="utf-8")
    log_path = root / "ops.jsonl"

    moved = filesystem.sort_once(
        root,
        rules={"pdf": "PDFs"},
        other_folder="Other",
        dry_run=False,
        recursive=False,
        include_hidden=False,
        action="move",
        min_age_seconds=0,
        log_path=log_path,
        verbose=False,
    )
    assert moved == 1
    assert not (root / "a.pdf").exists()
    assert (root / "PDFs" / "a.pdf").exists()

    undone = filesystem.undo_moves(log_path, dry_run=False, verbose=False)
    assert undone == 1
    assert (root / "a.pdf").exists()

