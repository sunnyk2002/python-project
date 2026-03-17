from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: list[str]

    def text_features(self) -> str:
        # Keep it simple: TF-IDF over title + genres works surprisingly well for a basic recommender.
        genres_text = " ".join(g.replace("-", " ") for g in self.genres if g)
        return f"{self.title} {genres_text}".strip()


class MissingDependency(RuntimeError):
    pass


def _require_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        return TfidfVectorizer, cosine_similarity
    except Exception as e:  # pragma: no cover
        raise MissingDependency(
            "scikit-learn is required for this recommender.\n"
            "Install (in a supported Python env) with: python -m pip install scikit-learn\n"
            "Note: scikit-learn may not have wheels for Python 3.14 yet; if install fails, "
            "use Python 3.12/3.13 in a virtualenv."
        ) from e


def load_movielens_movies_csv(path: Path) -> list[Movie]:
    """
    Loads a MovieLens-style `movies.csv` with columns:
      - movieId,title,genres
    Genres are pipe-delimited (e.g. Action|Adventure|Sci-Fi).
    """
    movies: list[Movie] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"movieId", "title", "genres"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Expected columns {sorted(required)} in {path}")

        for row in reader:
            try:
                movie_id = int(row["movieId"])
            except Exception:
                continue
            title = (row.get("title") or "").strip()
            genres_raw = (row.get("genres") or "").strip()
            if not title:
                continue

            if not genres_raw or genres_raw.lower() == "(no genres listed)":
                genres = []
            else:
                genres = [g.strip() for g in genres_raw.split("|") if g.strip()]

            movies.append(Movie(movie_id=movie_id, title=title, genres=genres))
    return movies


def demo_movies() -> list[Movie]:
    # Tiny dataset so the script is runnable without external files.
    # Use `--movies-csv` for real results.
    return [
        Movie(1, "Toy Story (1995)", ["Animation", "Children", "Comedy"]),
        Movie(2, "Jumanji (1995)", ["Adventure", "Children", "Fantasy"]),
        Movie(3, "Grumpier Old Men (1995)", ["Comedy", "Romance"]),
        Movie(4, "Waiting to Exhale (1995)", ["Comedy", "Drama", "Romance"]),
        Movie(5, "Heat (1995)", ["Action", "Crime", "Thriller"]),
        Movie(6, "GoldenEye (1995)", ["Action", "Adventure", "Thriller"]),
        Movie(7, "Sabrina (1995)", ["Comedy", "Romance"]),
        Movie(8, "Tom and Huck (1995)", ["Adventure", "Children"]),
        Movie(9, "Sudden Death (1995)", ["Action"]),
        Movie(10, "The American President (1995)", ["Comedy", "Drama", "Romance"]),
        Movie(11, "Se7en (1995)", ["Crime", "Mystery", "Thriller"]),
        Movie(12, "The Usual Suspects (1995)", ["Crime", "Mystery", "Thriller"]),
        Movie(13, "Braveheart (1995)", ["Action", "Drama", "War"]),
        Movie(14, "Star Wars: Episode IV - A New Hope (1977)", ["Action", "Adventure", "Sci-Fi"]),
        Movie(15, "The Matrix (1999)", ["Action", "Sci-Fi", "Thriller"]),
        Movie(16, "Spirited Away (2001)", ["Animation", "Fantasy"]),
        Movie(17, "The Notebook (2004)", ["Drama", "Romance"]),
    ]


class ContentBasedRecommender:
    def __init__(self) -> None:
        self._movies: list[Movie] = []
        self._vectorizer = None
        self._matrix = None

    def fit(self, movies: Sequence[Movie]) -> "ContentBasedRecommender":
        TfidfVectorizer, _ = _require_sklearn()
        self._movies = list(movies)
        corpus = [m.text_features() for m in self._movies]
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)
        return self

    def _index_by_id(self) -> dict[int, int]:
        return {m.movie_id: i for i, m in enumerate(self._movies)}

    def _index_by_title_lower(self) -> dict[str, int]:
        # If duplicate titles exist, this picks the first. For real datasets, use IDs.
        out: dict[str, int] = {}
        for i, m in enumerate(self._movies):
            key = m.title.casefold()
            out.setdefault(key, i)
        return out

    def search(self, query: str, *, limit: int = 10) -> list[Movie]:
        q = query.casefold().strip()
        if not q:
            return []
        matches = [m for m in self._movies if q in m.title.casefold()]
        return matches[:limit]

    def recommend(
        self,
        *,
        liked_titles: Sequence[str] = (),
        liked_ids: Sequence[int] = (),
        top_n: int = 10,
        exclude_liked: bool = True,
    ) -> list[tuple[Movie, float]]:
        if self._matrix is None or self._vectorizer is None:
            raise RuntimeError("Model not fit. Call fit(movies) first.")

        _, cosine_similarity = _require_sklearn()
        import numpy as np  # Local import to keep startup light.

        id_index = self._index_by_id()
        title_index = self._index_by_title_lower()

        seed_indices: list[int] = []
        for mid in liked_ids:
            i = id_index.get(int(mid))
            if i is not None:
                seed_indices.append(i)

        for t in liked_titles:
            i = title_index.get(t.casefold().strip())
            if i is not None:
                seed_indices.append(i)

        seed_indices = sorted(set(seed_indices))
        if not seed_indices:
            raise ValueError("No liked movies found. Use --query to find exact titles or pass --like-id.")

        # scipy sparse mean() returns a `numpy.matrix` which sklearn increasingly rejects.
        seed_vec = self._matrix[seed_indices].mean(axis=0)
        seed_arr = np.asarray(seed_vec)
        if seed_arr.ndim == 1:
            seed_arr = seed_arr.reshape(1, -1)
        sims = cosine_similarity(seed_arr, self._matrix).ravel()

        blocked: set[int] = set(seed_indices) if exclude_liked else set()
        ranked = sorted(
            ((i, float(s)) for i, s in enumerate(sims) if i not in blocked),
            key=lambda x: x[1],
            reverse=True,
        )
        results: list[tuple[Movie, float]] = []
        for i, score in ranked[: max(top_n, 0)]:
            results.append((self._movies[i], score))
        return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="movierecommendor.py",
        description="Simple movie recommender using scikit-learn (TF-IDF + cosine similarity).",
    )
    p.add_argument("--movies-csv", type=str, default=None, help="Path to MovieLens-style movies.csv")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use a tiny built-in dataset (for quick demos).",
    )
    p.add_argument(
        "--like",
        action="append",
        default=[],
        help='Exact movie title to seed recommendations (repeatable). Example: --like "Heat (1995)"',
    )
    p.add_argument(
        "--like-id",
        action="append",
        default=[],
        help="Movie ID to seed recommendations (repeatable).",
    )
    p.add_argument("--top-n", type=int, default=10, help="Number of recommendations to show (default: 10).")
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search titles (substring match) and print candidates; does not recommend unless --like/--like-id provided.",
    )
    p.add_argument("--show-scores", action="store_true", help="Print similarity scores.")
    return p


def _parse_ints(values: Iterable[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    if args.demo:
        movies = demo_movies()
    elif args.movies_csv:
        movies_path = Path(args.movies_csv).expanduser()
        # If the user passes a directory (or a path that doesn't end with .csv),
        # assume MovieLens layout and look for "movies.csv" inside it.
        if movies_path.is_dir() or movies_path.suffix.lower() != ".csv":
            movies_path = movies_path / "movies.csv"
        if not movies_path.exists():
            print(
                f"movies.csv not found: {movies_path}\n"
                "Provide a real path, for example:\n"
                '  python3 movierecommendor.py --movies-csv ~/Downloads/ml-latest-small/movies.csv --like-id 1\n'
                "You can also pass the containing directory and it will look for movies.csv inside it.",
                file=sys.stderr,
            )
            return 2
        try:
            movies = load_movielens_movies_csv(movies_path)
        except (OSError, ValueError) as e:
            print(f"Failed to load movies CSV: {e}", file=sys.stderr)
            return 2
    else:
        print("No dataset provided. Use --movies-csv path/to/movies.csv or --demo.", file=sys.stderr)
        return 2

    model = ContentBasedRecommender()
    try:
        model.fit(movies)
    except MissingDependency as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.query:
        matches = model.search(args.query, limit=20)
        if not matches:
            print("No matches.")
        else:
            for m in matches:
                genres = ", ".join(m.genres) if m.genres else "Unknown"
                print(f"{m.movie_id}\t{m.title}\t[{genres}]")

    liked_titles = [t for t in (args.like or []) if str(t).strip()]
    liked_ids = _parse_ints(args.like_id or [])
    if not liked_titles and not liked_ids:
        # Query-only mode is allowed; otherwise tell the user how to proceed.
        if args.query:
            return 0
        print('Provide at least one seed movie via --like "Title (Year)" or --like-id 123.', file=sys.stderr)
        return 2

    try:
        recs = model.recommend(liked_titles=liked_titles, liked_ids=liked_ids, top_n=args.top_n)
    except Exception as e:
        print(f"Failed to recommend: {e}", file=sys.stderr)
        return 2

    for movie, score in recs:
        if args.show_scores:
            print(f"{score:.3f}\t{movie.movie_id}\t{movie.title}")
        else:
            print(f"{movie.movie_id}\t{movie.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
