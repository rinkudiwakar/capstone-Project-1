from __future__ import annotations

import os
from dataclasses import dataclass


TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    watchmode_api_key: str | None
    movie_catalog_limit: int
    watchmode_region: str
    supabase_url: str | None
    supabase_service_role_key: str | None
    flask_port: int
    load_dotenv: bool
    eager_startup: bool
    preload_movies_on_startup: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            watchmode_api_key=os.getenv("WATCHMODE_API_KEY"),
            movie_catalog_limit=max(1, min(int(os.getenv("MOVIE_CATALOG_LIMIT", "24")), 100)),
            watchmode_region=os.getenv("WATCHMODE_REGION", "US").upper(),
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            flask_port=int(os.getenv("PORT", "5000")),
            load_dotenv=os.getenv("FLASK_APP_LOAD_DOTENV", "true").strip().lower() in TRUE_VALUES,
            eager_startup=os.getenv("FLASK_APP_EAGER_STARTUP", "true").strip().lower() in TRUE_VALUES,
            preload_movies_on_startup=os.getenv("FLASK_APP_PRELOAD_MOVIES", "true").strip().lower() in TRUE_VALUES,
        )
