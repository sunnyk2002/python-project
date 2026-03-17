from __future__ import annotations

from pathlib import Path

import pytest

import pdf_to_voice


def test_parse_pages_spec_empty_means_all() -> None:
    assert pdf_to_voice._parse_pages_spec("", 3) == [1, 2, 3]


def test_parse_pages_spec_list_and_ranges() -> None:
    assert pdf_to_voice._parse_pages_spec("1, 3, 5-6", 10) == [1, 3, 5, 6]


def test_parse_pages_spec_open_ended_range_caps_at_total() -> None:
    assert pdf_to_voice._parse_pages_spec("10-", 12) == [10, 11, 12]


def test_parse_pages_spec_invalid_range_raises() -> None:
    with pytest.raises(ValueError):
        pdf_to_voice._parse_pages_spec("3-1", 10)


def test_default_state_path_is_next_to_pdf() -> None:
    p = Path("/tmp/sample.pdf")
    s = pdf_to_voice._default_state_path(p)
    assert s.name.endswith(".pdf_to_voice_progress.json")
    assert s.parent == p.parent


def test_state_roundtrip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = {"pdf_path": "x", "last_completed_page": 2}
    pdf_to_voice._write_state(state_path, state)
    assert pdf_to_voice._load_state(state_path) == state


def test_main_returns_error_when_no_pdf_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_to_voice, "_pick_pdf_via_dialog", lambda: None)
    assert pdf_to_voice.main([]) == 2


def test_main_returns_error_when_pdf_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    assert pdf_to_voice.main([str(missing)]) == 2


def test_main_returns_error_when_deps_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    monkeypatch.setattr(pdf_to_voice, "_require_pdf_deps", lambda: (_ for _ in ()).throw(pdf_to_voice.MissingDependency("x")))
    assert pdf_to_voice.main([str(pdf)]) == 2

