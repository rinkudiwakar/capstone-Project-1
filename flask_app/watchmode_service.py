"""
WatchMode API Integration Service
Fetches real movie data from WatchMode API instead of using hardcoded data.

API Documentation: https://api.watchmode.com/docs
"""

import os
import requests
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class WatchModeAPIError(Exception):
    """Custom exception for WatchMode API errors."""
    pass


class WatchModeService:
    """Service class for interacting with WatchMode API."""
    
    BASE_URL = "https://api.watchmode.com/v1"
    CACHE_DURATION = 3600  # 1 hour cache for title data
    
    def __init__(self, api_key: str = None):
        """
        Initialize WatchMode service with API key.
        
        Args:
            api_key: WatchMode API key. If not provided, will read from WATCHMODE_API_KEY env var.
        
        Raises:
            WatchModeAPIError: If API key is not provided or found.
        """
        self.api_key = api_key or os.getenv("WATCHMODE_API_KEY")
        if not self.api_key:
            raise WatchModeAPIError(
                "WatchMode API key not found. Set WATCHMODE_API_KEY environment variable. "
                "Get a free key at https://api.watchmode.com/requestApiKey/"
            )
        self._cache = {}
        self._cache_timestamps = {}
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_timestamps:
            return False
        return datetime.now() - self._cache_timestamps[key] < timedelta(seconds=self.CACHE_DURATION)
    
    def _get_cached(self, key: str) -> Optional[any]:
        """Get cached data if valid."""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _set_cache(self, key: str, value: any) -> None:
        """Set cache with timestamp."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make HTTP request to WatchMode API.
        
        Args:
            endpoint: API endpoint path (without base URL)
            params: Query parameters
        
        Returns:
            Response JSON data
        
        Raises:
            WatchModeAPIError: If API request fails
        """
        if params is None:
            params = {}
        
        # Add API key to all requests
        params["apiKey"] = self.api_key
        
        try:
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise WatchModeAPIError("WatchMode API request timed out")
        except requests.exceptions.ConnectionError:
            raise WatchModeAPIError("Failed to connect to WatchMode API")
        except requests.exceptions.HTTPError as e:
            response_body = ""
            try:
                response_body = response.text[:500]
            except Exception:
                response_body = ""
            if response.status_code == 401:
                raise WatchModeAPIError("Invalid or missing WatchMode API key")
            elif response.status_code == 404:
                raise WatchModeAPIError("Resource not found on WatchMode API")
            raise WatchModeAPIError(
                f"WatchMode API error: {e}. Response body: {response_body}"
            )
        except Exception as e:
            raise WatchModeAPIError(f"Unexpected error calling WatchMode API: {e}")
    
    def search_titles(self, query: str, search_type: str = "movie") -> List[Dict]:
        """
        Search for movie/TV titles by name.
        
        Args:
            query: Search query (movie/show name)
            search_type: Filter by type - "movie", "tv_series", or None for both
        
        Returns:
            List of title results with id, title, type, year, poster, user_rating
        
        Example:
            results = service.search_titles("Breaking Bad")
        """
        cache_key = f"search_{query}_{search_type}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            params = {
                "search_field": "name",
                "search_value": query,
            }
            if search_type:
                params["types"] = search_type
            
            data = self._make_request("/search", params)
            
            results = []
            if "title_results" in data:
                for title in data["title_results"]:
                    results.append({
                        "id": title.get("id"),
                        "title": title.get("title"),
                        "type": title.get("type"),
                        "year": title.get("year"),
                        "poster": title.get("poster_url"),
                        "imdb_id": title.get("imdb_id"),
                        "plot_overview": title.get("plot_overview"),
                    })
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.error(f"Error searching titles: {e}")
            raise
    
    def get_title_details(self, title_id: int, include_sources: bool = True) -> Dict:
        """
        Get detailed information about a title.
        
        Args:
            title_id: WatchMode title ID
            include_sources: Include streaming source information
        
        Returns:
            Dictionary with title details, cast, ratings, streaming sources
        
        Example:
            details = service.get_title_details(3173903)
        """
        cache_key = f"title_details_{title_id}_{include_sources}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            params = {}
            if include_sources:
                params["append_to_response"] = "sources,cast-crew"
            
            data = self._make_request(f"/title/{title_id}/details", params)
            
            result = {
                "id": data.get("id"),
                "title": data.get("title"),
                "type": data.get("type"),
                "plot_overview": data.get("plot_overview"),
                "year": data.get("year"),
                "release_date": data.get("release_date"),
                "runtime_minutes": data.get("runtime_minutes"),
                "user_rating": data.get("user_rating"),
                "critic_score": data.get("critic_score"),
                "us_rating": data.get("us_rating"),
                "genres": data.get("genre_names", []),
                "poster": data.get("poster"),
                "backdrop": data.get("backdrop"),
                "imdb_id": data.get("imdb_id"),
                "trailer": data.get("trailer"),
            }
            
            # Add cast if available
            if "cast" in data:
                result["cast"] = [
                    {
                        "name": person.get("full_name"),
                        "role": person.get("role"),
                        "type": person.get("type")
                    }
                    for person in data["cast"][:10]  # Limit to 10 cast members
                ]
            
            # Add streaming sources if available
            if "sources" in data:
                result["streaming_sources"] = [
                    {
                        "name": source.get("name"),
                        "type": source.get("type"),
                        "region": source.get("region"),
                        "web_url": source.get("web_url"),
                    }
                    for source in data["sources"]
                ]
            
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error getting title details: {e}")
            raise
    
    def get_streaming_sources(self, title_id: int, region: str = "US") -> List[Dict]:
        """
        Get streaming sources for a title by region.
        
        Args:
            title_id: WatchMode title ID
            region: 2-letter country code (default: US)
        
        Returns:
            List of streaming sources with availability info
        
        Example:
            sources = service.get_streaming_sources(345534)
        """
        try:
            params = {"regions": region}
            data = self._make_request(f"/title/{title_id}/sources", params)
            
            sources = []
            for source in data:
                sources.append({
                    "source_id": source.get("source_id"),
                    "name": source.get("name"),
                    "type": source.get("type"),  # "sub", "rent", "buy", "free", "tve"
                    "region": source.get("region"),
                    "web_url": source.get("web_url"),
                    "price": source.get("price"),
                    "format": source.get("format"),
                })
            
            return sources
        except Exception as e:
            logger.error(f"Error getting streaming sources: {e}")
            raise
    
    def list_titles(self, **filters) -> Tuple[List[Dict], int]:
        """
        List and filter titles with flexible criteria.
        
        Args:
            types: "movie", "tv_series", etc. (comma-separated)
            regions: 2-letter country codes (comma-separated, default: US)
            source_types: "sub", "free", "rent", "buy" (comma-separated)
            source_ids: Specific streaming service IDs (comma-separated)
            genres: Genre IDs (comma-separated)
            user_rating_low: Minimum user rating (0-10)
            user_rating_high: Maximum user rating (0-10)
            sort_by: "release_date_desc", "popularity_desc", etc.
            page: Page number (default: 1)
            limit: Results per page (default: 250, max: 250)
        
        Returns:
            Tuple of (list of titles, total_results)
        
        Example:
            titles, total = service.list_titles(
                types="movie",
                user_rating_low=7,
                sort_by="release_date_desc",
                limit=20
            )
        """
        try:
            # Build query parameters from kwargs
            params = {}
            
            # Map friendly parameter names to API names
            param_mapping = {
                "types": "types",
                "regions": "regions",
                "source_types": "source_types",
                "source_ids": "source_ids",
                "genres": "genres",
                "network_ids": "network_ids",
                "languages": "languages",
                "user_rating_low": "user_rating_low",
                "user_rating_high": "user_rating_high",
                "critic_score_low": "critic_score_low",
                "critic_score_high": "critic_score_high",
                "sort_by": "sort_by",
                "page": "page",
                "limit": "limit",
            }
            
            for key, api_key in param_mapping.items():
                if key in filters:
                    params[api_key] = filters[key]
            
            # Set defaults
            if "limit" not in params:
                params["limit"] = 50
            if "page" not in params:
                params["page"] = 1
            
            data = self._make_request("/list-titles", params)
            
            titles = []
            for title in data.get("titles", []):
                titles.append({
                    "id": title.get("id"),
                    "title": title.get("title"),
                    "type": title.get("type"),
                    "year": title.get("year"),
                    "plot_overview": title.get("plot_overview"),
                    "poster": title.get("poster_url"),
                    "user_rating": title.get("user_rating"),
                    "critic_score": title.get("critic_score"),
                    "genres": title.get("genre_names", []),
                })
            
            return titles, data.get("total_results", 0)
        except Exception as e:
            logger.error(f"Error listing titles: {e}")
            raise
    
    def get_popular_movies(self, limit: int = 50, region: str = "US") -> List[Dict]:
        """
        Get popular movies.
        
        Args:
            limit: Number of movies to return (max 250)
            region: 2-letter country code
        
        Returns:
            List of popular movies
        """
        safe_limit = min(limit, 250)
        strategies = [
            {
                "label": "popular movies by region and popularity",
                "filters": {
                    "types": "movie",
                    "regions": region,
                    "sort_by": "popularity_desc",
                    "limit": safe_limit,
                },
            },
            {
                "label": "popular movies by popularity",
                "filters": {
                    "types": "movie",
                    "sort_by": "popularity_desc",
                    "limit": safe_limit,
                },
            },
            {
                "label": "recent movies by region",
                "filters": {
                    "types": "movie",
                    "regions": region,
                    "sort_by": "release_date_desc",
                    "limit": safe_limit,
                },
            },
            {
                "label": "recent movies",
                "filters": {
                    "types": "movie",
                    "sort_by": "release_date_desc",
                    "limit": safe_limit,
                },
            },
            {
                "label": "basic movie listing",
                "filters": {
                    "types": "movie",
                    "limit": safe_limit,
                },
            },
        ]

        errors = []
        for strategy in strategies:
            try:
                logger.info(f"Trying WatchMode catalog strategy: {strategy['label']}")
                titles, _ = self.list_titles(**strategy["filters"])
                deduped = self._dedupe_titles(titles)
                if deduped:
                    logger.info(
                        "WatchMode catalog strategy succeeded: %s (%s titles)",
                        strategy["label"],
                        len(deduped),
                    )
                    return deduped[:safe_limit]
            except Exception as exc:
                logger.warning(
                    "WatchMode catalog strategy failed: %s - %s",
                    strategy["label"],
                    exc,
                )
                errors.append(f"{strategy['label']}: {exc}")

        logger.info("Falling back to WatchMode search-based catalog bootstrap")
        search_seed_queries = ("avengers", "batman", "star", "love", "night", "mission")
        aggregated: list[Dict] = []
        for query in search_seed_queries:
            try:
                aggregated.extend(self.search_titles(query, search_type="movie"))
            except Exception as exc:
                logger.warning("WatchMode search fallback failed for query '%s': %s", query, exc)

        deduped = self._dedupe_titles(aggregated)
        if deduped:
            logger.info("WatchMode search fallback succeeded with %s titles", len(deduped))
            return deduped[:safe_limit]

        raise WatchModeAPIError(
            "Unable to fetch startup movie catalog from WatchMode. Attempts: "
            + " | ".join(errors)
        )
    
    def get_highly_rated_movies(self, min_rating: float = 7.0, limit: int = 50) -> List[Dict]:
        """
        Get highly-rated movies.
        
        Args:
            min_rating: Minimum user rating (0-10)
            limit: Number of movies to return
        
        Returns:
            List of highly-rated movies
        """
        try:
            titles, _ = self.list_titles(
                types="movie",
                user_rating_low=min_rating,
                sort_by="release_date_desc",
                limit=min(limit, 250)
            )
            return titles
        except Exception as e:
            logger.error(f"Error getting highly-rated movies: {e}")
            raise
    
    def get_api_quota(self) -> Dict:
        """
        Get current API quota and usage information.
        
        Returns:
            Dictionary with quota and quotaUsed
        """
        try:
            data = self._make_request("/status")
            return {
                "quota": data.get("quota"),
                "quota_used": data.get("quotaUsed"),
                "quota_remaining": data.get("quota") - data.get("quotaUsed") if data.get("quota") else None,
            }
        except Exception as e:
            logger.error(f"Error getting API quota: {e}")
            raise
    
    def get_sources_list(self, region: str = "US") -> List[Dict]:
        """
        Get list of all streaming sources (Netflix, Prime, etc.).
        
        Args:
            region: 2-letter country code
        
        Returns:
            List of streaming sources with logos and app store links
        """
        cache_key = f"sources_{region}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            params = {"regions": region}
            data = self._make_request("/sources", params)
            
            sources = []
            for source in data:
                sources.append({
                    "id": source.get("id"),
                    "name": source.get("name"),
                    "type": source.get("type"),
                    "logo_100px": source.get("logo_100px"),
                    "ios_url": source.get("ios_appstore_url"),
                    "android_url": source.get("android_playstore_url"),
                })
            
            self._set_cache(cache_key, sources)
            return sources
        except Exception as e:
            logger.error(f"Error getting sources list: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("WatchMode service cache cleared")

    @staticmethod
    def _dedupe_titles(titles: List[Dict]) -> List[Dict]:
        deduped: List[Dict] = []
        seen_ids = set()
        for title in titles:
            title_id = title.get("id")
            if not title_id or title_id in seen_ids:
                continue
            seen_ids.add(title_id)
            deduped.append(title)
        return deduped


# Singleton instance (created on first use)
_service_instance = None


def get_watchmode_service(api_key: str = None) -> WatchModeService:
    """
    Get or create WatchMode service singleton.
    
    Args:
        api_key: Optional API key to override environment variable
    
    Returns:
        WatchModeService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = WatchModeService(api_key)
    return _service_instance
