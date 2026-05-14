from __future__ import annotations

import ast
import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATASET_DIR = BASE_DIR / "FRED" / "raw data" / "the_movies_dataset"
RAW_DATASET_FALLBACK_DIR = BASE_DIR / "FRED" / "raw data"

INFLATION_PATH = DATA_DIR / "United-States-Inflation-Rate-Consumer-Price-Index-2026-05-06-15-32.csv"

OUTPUT_FULL_PATH = DATA_DIR / "movies_with_genres_and_cast.csv"
OUTPUT_TRAIN_PATH = DATA_DIR / "movies_with_genres_and_cast_train.csv"
OUTPUT_TEST_PATH = DATA_DIR / "movies_with_genres_and_cast_test.csv"

MIN_YEAR = 1990
TEST_YEARS = 5
TOP_N_CAST = 5

MOVIE_BASE_COLUMNS = [
    "budget",
    "id",
    "popularity",
    "release_date",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
]

GENRE_COLUMNS = [
    "genre_drama",
    "genre_comedy",
    "genre_thriller",
    "genre_romance",
    "genre_action",
    "genre_horror",
    "genre_crime",
    "genre_documentary",
    "genre_adventure",
    "genre_science_fiction",
    "other_genre",
]

ACTOR_COLUMNS = [f"Actor_{i}" for i in range(1, TOP_N_CAST + 1)]

OUTPUT_COLUMNS = [
    "budget",
    "id",
    "popularity",
    "release_date",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "doy",
    "year",
    "dow",
    "week",
    *GENRE_COLUMNS,
    *ACTOR_COLUMNS,
    "revenue_inflated",
    "budget_inflated",
    "title",
    "keywords",
    "month",
    "month_sin",
    "month_cos",
]

TARGET_GENRE_NAMES = [
    "Drama",
    "Comedy",
    "Thriller",
    "Romance",
    "Action",
    "Horror",
    "Crime",
    "Documentary",
    "Adventure",
    "Science Fiction",
]
GENRE_NAME_TO_COLUMN = {
    name: f"genre_{name.lower().replace(' ', '_')}" for name in TARGET_GENRE_NAMES
}


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_id(value: object) -> int | None:
    numeric = parse_float(value)
    if numeric is None:
        return None
    return int(numeric)


def parse_literal_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def resolve_raw_path(filename: str) -> Path:
    preferred = RAW_DATASET_DIR / filename
    if preferred.exists():
        return preferred

    fallback = RAW_DATASET_FALLBACK_DIR / filename
    if fallback.exists():
        return fallback

    return preferred


MOVIES_METADATA_PATH = resolve_raw_path("movies_metadata.csv")
CREDITS_PATH = resolve_raw_path("credits.csv")
KEYWORDS_PATH = resolve_raw_path("keywords.csv")


def load_inflation_factors() -> dict[int, float]:
    yearly_rates: dict[int, float] = {}

    with INFLATION_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            year_raw = (row.get("year") or row.get('"year"') or "").strip().strip('"')
            rate_raw = (row.get("Inflation Rate") or row.get('"Inflation Rate"') or "").strip().strip('"')
            if not year_raw or not rate_raw:
                continue
            year = int(year_raw)
            rate = float(rate_raw.replace(",", ".")) / 100.0 + 1.0
            yearly_rates[year] = rate

    if not yearly_rates:
        raise RuntimeError("Inflation data could not be loaded.")

    factors: dict[int, float] = {}
    running = 1.0
    for year in sorted(yearly_rates, reverse=True):
        factors[year] = running
        running *= yearly_rates[year]

    return factors


def load_keywords_lookup() -> dict[int, str]:
    lookup: dict[int, str] = {}

    with KEYWORDS_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = parse_id(row.get("id"))
            if movie_id is None:
                continue
            keywords = (row.get("keywords") or "").strip()
            lookup[movie_id] = keywords or "[]"

    return lookup


def load_credits_lookup() -> dict[int, list[list[int | None]]]:
    lookup: dict[int, list[list[int | None]]] = {}

    with CREDITS_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = parse_id(row.get("id"))
            if movie_id is None:
                continue

            cast_list = parse_literal_list(row.get("cast"))
            normalized_cast: list[dict[str, object]] = []
            for member in cast_list:
                if not isinstance(member, dict):
                    continue
                order = parse_float(member.get("order"))
                actor_id = parse_id(member.get("id"))
                if order is None or actor_id is None:
                    continue
                normalized_cast.append({"order": order, "id": actor_id})

            normalized_cast.sort(key=lambda item: item["order"])
            actor_ids = [int(member["id"]) for member in normalized_cast[:TOP_N_CAST]]
            actor_ids.extend([None] * (TOP_N_CAST - len(actor_ids)))
            lookup.setdefault(movie_id, []).append(actor_ids)

    return lookup


def build_genre_values(genre_literal: object) -> dict[str, float]:
    row_genres = parse_literal_list(genre_literal)
    genre_names = []
    for item in row_genres:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                genre_names.append(name.strip())

    total_genres = max(len(genre_names), 1)
    weight = round(1.0 / total_genres, 2)
    values = {column: 0.0 for column in GENRE_COLUMNS}

    found_known_genre = False
    for name in genre_names:
        column = GENRE_NAME_TO_COLUMN.get(name)
        if column is not None:
            values[column] = weight
            found_known_genre = True

    values["other_genre"] = 0.0 if found_known_genre else weight
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_rows() -> tuple[list[dict[str, object]], Counter]:
    removal_counts: Counter[str] = Counter()
    rows_kept: list[dict[str, object]] = []

    inflation_factors = load_inflation_factors()
    keywords_lookup = load_keywords_lookup()
    credits_lookup = load_credits_lookup()

    with MOVIES_METADATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for raw_row in reader:
            release_date = (raw_row.get("release_date") or "").strip()
            if not release_date:
                removal_counts["missing_release_date"] += 1
                continue

            try:
                release_dt = datetime.strptime(release_date, "%Y-%m-%d")
            except ValueError:
                removal_counts["invalid_release_date"] += 1
                continue

            year_value = parse_float(raw_row.get("year"))
            if year_value is None:
                year_value = parse_float(release_dt.year)
            year = int(year_value)
            if year != release_dt.year:
                removal_counts["year_release_mismatch"] += 1
                continue
            if year < MIN_YEAR:
                removal_counts["before_1990"] += 1
                continue

            movie_id = parse_id(raw_row.get("id"))
            if movie_id is None:
                removal_counts["missing_id"] += 1
                continue

            revenue = parse_float(raw_row.get("revenue"))
            if revenue is None or revenue == 0:
                removal_counts["missing_or_zero_revenue"] += 1
                continue

            budget = parse_float(raw_row.get("budget"))
            if budget is None:
                removal_counts["missing_budget"] += 1
                continue

            inflation_factor = inflation_factors.get(year)
            if inflation_factor is None:
                removal_counts["missing_inflation_factor"] += 1
                continue

            row: dict[str, object] = {}
            for column in MOVIE_BASE_COLUMNS:
                row[column] = raw_row.get(column, "")

            row["id"] = float(movie_id)
            row["year"] = year
            row["doy"] = release_dt.timetuple().tm_yday
            row["dow"] = release_dt.weekday()
            row["week"] = int(release_dt.isocalendar().week)

            genre_values = build_genre_values(raw_row.get("genres"))
            row.update(genre_values)

            credit_rows = credits_lookup.get(movie_id, [[None] * TOP_N_CAST])
            for actor_ids in credit_rows:
                out_row = dict(row)
                for col_name, actor_id in zip(ACTOR_COLUMNS, actor_ids):
                    out_row[col_name] = actor_id

                out_row["revenue_inflated"] = revenue * inflation_factor
                out_row["budget_inflated"] = budget * inflation_factor
                out_row["title"] = (raw_row.get("title") or "").strip()
                out_row["keywords"] = keywords_lookup.get(movie_id, "[]")
                out_row["month"] = release_dt.month
                angle = 2.0 * math.pi * out_row["month"] / 12.0
                out_row["month_sin"] = math.sin(angle)
                out_row["month_cos"] = math.cos(angle)

                rows_kept.append(out_row)

    return rows_kept, removal_counts


def main() -> None:
    for path in [MOVIES_METADATA_PATH, CREDITS_PATH, KEYWORDS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw input: {path}")

    rows_kept, removal_counts = build_rows()
    if not rows_kept:
        raise RuntimeError("No rows left after filtering.")

    rows_kept.sort(key=lambda row: (int(row["year"]), str(row["release_date"])))
    max_year = max(int(row["year"]) for row in rows_kept)
    test_start_year = max_year - TEST_YEARS + 1

    train_rows = [row for row in rows_kept if int(row["year"]) < test_start_year]
    test_rows = [row for row in rows_kept if int(row["year"]) >= test_start_year]

    write_csv(OUTPUT_FULL_PATH, rows_kept)
    write_csv(OUTPUT_TRAIN_PATH, train_rows)
    write_csv(OUTPUT_TEST_PATH, test_rows)

    print(f"kept_rows: {len(rows_kept)}")
    for reason in sorted(removal_counts):
        print(f"removed_{reason}: {removal_counts[reason]}")
    print(f"year_range: {int(rows_kept[0]['year'])}-{max_year}")
    print(f"test_years: {test_start_year}-{max_year}")
    print(f"train_rows: {len(train_rows)}")
    print(f"test_rows: {len(test_rows)}")
    print(f"saved_full: {OUTPUT_FULL_PATH.name}")
    print(f"saved_train: {OUTPUT_TRAIN_PATH.name}")
    print(f"saved_test: {OUTPUT_TEST_PATH.name}")


if __name__ == "__main__":
    main()
