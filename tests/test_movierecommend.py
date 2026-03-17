from __future__ import annotations

import csv
from pathlib import Path

import pytest

import movierecommend


def test_movie_text_features_contains_title_and_genres() -> None:
    m = movierecommend.Movie(movie_id=1, title="Heat (1995)", genres=["Action", "Crime", "Thriller"])
    text = m.text_features()
    assert "Heat" in text
    assert "Action" in text


def test_load_movielens_movies_csv_parses_and_skips_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "movies.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        w.writerow(["1", "Toy Story (1995)", "Animation|Children|Comedy"])
        w.writerow(["x", "Bad", "Action"])
        w.writerow(["2", "", "Comedy"])
        w.writerow(["3", "No Genres (1995)", "(no genres listed)"])

    movies = movierecommend.load_movielens_movies_csv(path)
    assert [m.movie_id for m in movies] == [1, 3]
    assert movies[0].genres == ["Animation", "Children", "Comedy"]
    assert movies[1].genres == []


def test_parse_ints_ignores_invalid() -> None:
    assert movierecommend._parse_ints(["1", "x", "2"]) == [1, 2]


def test_search_is_case_insensitive() -> None:
    model = movierecommend.ContentBasedRecommender()
    model._movies = movierecommend.demo_movies()
    hits = model.search("matrix")
    assert hits
    assert any("Matrix" in m.title for m in hits)


def test_recommend_requires_fit() -> None:
    model = movierecommend.ContentBasedRecommender()
    with pytest.raises(RuntimeError):
        model.recommend(liked_titles=["Heat (1995)"])


def test_main_requires_dataset() -> None:
    assert movierecommend.main([]) == 2

