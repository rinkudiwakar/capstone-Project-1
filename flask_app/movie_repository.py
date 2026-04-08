from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import urljoin

import requests


class SupabaseMovieRepository:
    def __init__(self, supabase_url: str, service_role_key: str):
        if not supabase_url:
            raise ValueError("SUPABASE_URL is required.")
        if not service_role_key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY is required.")

        self.base_url = supabase_url.rstrip("/")
        self.rest_url = f"{self.base_url}/rest/v1"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
                "Content-Type": "application/json",
            }
        )

    def close(self) -> None:
        self.session.close()

    def init_schema(self) -> None:
        self._request("GET", "movies", params={"select": "watchmode_id", "limit": "1"})
        self._request(
            "GET",
            "global_review_stats",
            params={"select": "total_reviews", "limit": "1"},
        )

    def upsert_movie(self, movie: dict[str, Any]) -> None:
        payload = {
            "watchmode_id": movie["watchmode_id"],
            "title": movie.get("title"),
            "movie_type": movie.get("movie_type"),
            "release_year": movie.get("release_year"),
            "release_date": movie.get("release_date"),
            "description": movie.get("description"),
            "poster_url": movie.get("poster_url"),
            "backdrop_url": movie.get("backdrop_url"),
            "imdb_id": movie.get("imdb_id"),
            "tmdb_id": movie.get("tmdb_id"),
            "user_rating": movie.get("user_rating"),
            "critic_score": movie.get("critic_score"),
            "runtime_minutes": movie.get("runtime_minutes"),
            "genres": movie.get("genres") or [],
            "api_payload": movie.get("api_payload") or {},
        }
        self._request(
            "POST",
            "movies",
            params={"on_conflict": "watchmode_id"},
            json=payload,
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )

    def list_movie_stats(self, movie_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not movie_ids:
            return {}
        values = ",".join(str(int(movie_id)) for movie_id in movie_ids)
        rows = self._request(
            "GET",
            "movie_review_stats",
            params={
                "select": "watchmode_id,review_count,average_sentiment,average_rating,positive_reviews,negative_reviews",
                "watchmode_id": f"in.({values})",
            },
        )
        return {
            int(row["watchmode_id"]): {
                "review_count": row.get("review_count") or 0,
                "average_sentiment": self._to_float(row.get("average_sentiment")),
                "average_rating": self._to_float(row.get("average_rating")),
                "positive_reviews": row.get("positive_reviews") or 0,
                "negative_reviews": row.get("negative_reviews") or 0,
            }
            for row in rows
        }

    def get_movie_snapshot(self, movie_id: int) -> dict[str, Any] | None:
        rows = self._request(
            "GET",
            "movies",
            params={
                "select": "watchmode_id,title,movie_type,release_year,release_date,description,poster_url,backdrop_url,imdb_id,tmdb_id,user_rating,critic_score,runtime_minutes,genres,api_payload",
                "watchmode_id": f"eq.{int(movie_id)}",
                "limit": "1",
            },
        )
        if not rows:
            return None
        return self._serialize_movie_row(rows[0])

    def add_review(
        self,
        movie_id: int,
        review_text: str,
        sentiment: int,
        user_id: str | None,
        display_name: str | None,
        is_anonymous: bool,
        rating: int | None = None,
    ) -> int:
        db_user_id = self._resolve_or_create_user(
            user_id=user_id,
            display_name=display_name,
            create=not is_anonymous,
        )
        rows = self._request(
            "POST",
            "reviews",
            json={
                "watchmode_id": int(movie_id),
                "review_text": review_text,
                "sentiment": int(sentiment),
                "user_id": db_user_id,
                "rating": rating,
                "is_anonymous": is_anonymous,
            },
            headers={"Prefer": "return=representation"},
        )
        return int(rows[0]["id"])

    def get_reviews_for_movie(self, movie_id: int, limit: int = 10) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 100))
        rows = self._request(
            "GET",
            "reviews",
            params={
                "select": "id,watchmode_id,review_text,sentiment,rating,user_id,is_anonymous,created_at,app_users(external_user_id,display_name)",
                "watchmode_id": f"eq.{int(movie_id)}",
                "order": "created_at.desc",
                "limit": str(safe_limit),
            },
        )
        return [self._serialize_review_row(row) for row in rows]

    def get_global_stats(self) -> dict[str, Any]:
        rows = self._request(
            "GET",
            "global_review_stats",
            params={
                "select": "total_reviews,reviewed_movies,overall_sentiment,overall_rating",
                "limit": "1",
            },
        )
        stats = rows[0] if rows else {}
        total_movies = self._head_count(
            "movies",
        )
        return {
            "total_reviews": stats.get("total_reviews") or 0,
            "reviewed_movies": stats.get("reviewed_movies") or 0,
            "overall_sentiment": self._to_float(stats.get("overall_sentiment")) or 0,
            "overall_rating": self._to_float(stats.get("overall_rating")) or 0,
            "total_movies": total_movies,
        }

    def get_random_reviewed_movie_id(self) -> int | None:
        rows = self._request(
            "GET",
            "movies",
            params={"select": "watchmode_id", "limit": "1"},
        )
        if not rows:
            return None
        return int(rows[0]["watchmode_id"])

    def _resolve_or_create_user(
        self,
        user_id: str | None,
        display_name: str | None,
        create: bool,
    ) -> str | None:
        if not user_id and not display_name:
            return None

        normalized_user_id = (user_id or "").strip()
        normalized_display_name = (display_name or "").strip() or None

        if not normalized_user_id and not create:
            return None
        if not normalized_user_id and create:
            normalized_user_id = str(uuid.uuid4())

        if self._is_uuid(normalized_user_id):
            rows = self._request(
                "GET",
                "app_users",
                params={"select": "id,display_name", "id": f"eq.{normalized_user_id}", "limit": "1"},
            )
            if rows:
                self._update_user_display_name(rows[0]["id"], normalized_display_name, rows[0].get("display_name"))
                return rows[0]["id"]
            if create:
                created = self._request(
                    "POST",
                    "app_users",
                    params={"on_conflict": "id"},
                    json={
                        "id": normalized_user_id,
                        "external_user_id": normalized_user_id,
                        "display_name": normalized_display_name,
                    },
                    headers={"Prefer": "return=representation,resolution=merge-duplicates"},
                )
                return created[0]["id"]
            return normalized_user_id

        rows = self._request(
            "GET",
            "app_users",
            params={
                "select": "id,display_name",
                "external_user_id": f"eq.{self._quote_value(normalized_user_id)}",
                "limit": "1",
            },
        )
        if rows:
            self._update_user_display_name(rows[0]["id"], normalized_display_name, rows[0].get("display_name"))
            return rows[0]["id"]
        if not create:
            return None
        created = self._request(
            "POST",
            "app_users",
            params={"on_conflict": "external_user_id"},
            json={
                "external_user_id": normalized_user_id,
                "display_name": normalized_display_name,
            },
            headers={"Prefer": "return=representation,resolution=merge-duplicates"},
        )
        return created[0]["id"]

    def _update_user_display_name(
        self,
        app_user_id: str,
        display_name: str | None,
        current_display_name: str | None,
    ) -> None:
        if not display_name or display_name == current_display_name:
            return
        self._request(
            "PATCH",
            "app_users",
            params={"id": f"eq.{app_user_id}"},
            json={"display_name": display_name},
            headers={"Prefer": "return=minimal"},
        )

    def _serialize_movie_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "watchmode_id": int(row["watchmode_id"]),
            "title": row.get("title"),
            "movie_type": row.get("movie_type"),
            "release_year": row.get("release_year"),
            "release_date": row.get("release_date"),
            "description": row.get("description"),
            "poster_url": row.get("poster_url"),
            "backdrop_url": row.get("backdrop_url"),
            "imdb_id": row.get("imdb_id"),
            "tmdb_id": row.get("tmdb_id"),
            "user_rating": self._to_float(row.get("user_rating")),
            "critic_score": row.get("critic_score"),
            "runtime_minutes": row.get("runtime_minutes"),
            "genres": row.get("genres") or [],
            "api_payload": row.get("api_payload") or {},
        }

    def _serialize_review_row(self, row: dict[str, Any]) -> dict[str, Any]:
        app_user = row.get("app_users") or {}
        return {
            "id": int(row["id"]),
            "movie_id": int(row["watchmode_id"]),
            "review_text": row["review_text"],
            "sentiment": int(row["sentiment"]),
            "rating": row.get("rating"),
            "user_id": row.get("user_id"),
            "external_user_id": app_user.get("external_user_id"),
            "display_name": app_user.get("display_name"),
            "is_anonymous": bool(row["is_anonymous"]),
            "created_at": row.get("created_at"),
        }

    def _request(
        self,
        method: str,
        resource: str,
        *,
        params: dict[str, str] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = self.session.request(
            method=method,
            url=urljoin(f"{self.rest_url}/", resource),
            params=params,
            json=json,
            headers=headers,
            timeout=20,
        )
        if response.status_code >= 400:
            try:
                payload = response.json()
                message = payload.get("message") or payload.get("hint") or str(payload)
            except ValueError:
                message = response.text
            raise RuntimeError(f"Supabase request failed for {resource}: {response.status_code} {message}")
        if not response.text.strip():
            return []
        return response.json()

    def _head_count(self, resource: str) -> int:
        response = self.session.request(
            method="HEAD",
            url=urljoin(f"{self.rest_url}/", resource),
            params={"select": "watchmode_id"},
            headers={"Prefer": "count=exact"},
            timeout=20,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Supabase count request failed for {resource}: {response.status_code} {response.text}")
        content_range = response.headers.get("Content-Range", "")
        if "/" not in content_range:
            return 0
        try:
            return int(content_range.split("/")[-1])
        except ValueError:
            return 0

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _is_uuid(value: str) -> bool:
        try:
            uuid.UUID(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _quote_value(value: str) -> str:
        return value.replace(",", r"\,")
