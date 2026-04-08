from __future__ import annotations

from typing import Any


def normalize_watchmode_movie(payload: dict[str, Any]) -> dict[str, Any]:
    movie_id = payload.get("id")
    if movie_id is None:
        raise ValueError("Movie payload is missing WatchMode id")

    genres = payload.get("genres") or payload.get("genre_names") or []
    if isinstance(genres, str):
        genres = [part.strip() for part in genres.split(",") if part.strip()]

    release_year = payload.get("year") or payload.get("release_year")
    description = payload.get("plot_overview") or payload.get("description") or ""

    return {
        "id": int(movie_id),
        "watchmode_id": int(movie_id),
        "title": payload.get("title") or "Unknown",
        "movie_type": payload.get("type") or payload.get("movie_type") or "movie",
        "release_year": release_year,
        "release_date": payload.get("release_date"),
        "description": description,
        "poster_url": payload.get("poster") or payload.get("poster_url"),
        "backdrop_url": payload.get("backdrop") or payload.get("backdrop_url"),
        "imdb_id": payload.get("imdb_id"),
        "tmdb_id": payload.get("tmdb_id"),
        "user_rating": payload.get("user_rating"),
        "critic_score": payload.get("critic_score"),
        "runtime_minutes": payload.get("runtime_minutes"),
        "genres": genres,
        "api_payload": payload,
    }


def enrich_movies_with_stats(movies: list[dict[str, Any]], stats_by_movie_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for movie in movies:
        movie_id = int(movie["id"])
        stats = stats_by_movie_id.get(movie_id, {})
        enriched.append(
            {
                **movie,
                "review_count": stats.get("review_count", 0),
                "average_sentiment": stats.get("average_sentiment"),
                "average_rating": stats.get("average_rating"),
                "positive_reviews": stats.get("positive_reviews", 0),
                "negative_reviews": stats.get("negative_reviews", 0),
            }
        )
    return enriched
