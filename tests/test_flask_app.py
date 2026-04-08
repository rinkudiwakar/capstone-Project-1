import json
import unittest
from unittest.mock import Mock, patch

with patch.dict(
    "os.environ",
    {
        "FLASK_APP_LOAD_DOTENV": "false",
        "FLASK_APP_EAGER_STARTUP": "false",
        "FLASK_APP_PRELOAD_MOVIES": "false",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "service-role-key",
        "WATCHMODE_API_KEY": "test-key",
    },
    clear=False,
):
    from flask_app.app import app


class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def build_repo(self):
        repo = Mock()
        repo.get_global_stats.return_value = {
            "total_reviews": 3,
            "reviewed_movies": 2,
            "overall_sentiment": 0.67,
            "overall_rating": 4.5,
            "total_movies": 12,
        }
        repo.get_reviews_for_movie.return_value = [
            {
                "id": 1,
                "movie_id": 101,
                "review_text": "Great movie",
                "sentiment": 1,
                "rating": 5,
                "user_id": None,
                "external_user_id": None,
                "display_name": None,
                "is_anonymous": True,
                "created_at": "2026-04-08T00:00:00+00:00",
            }
        ]
        repo.list_movie_stats.return_value = {
            101: {
                "review_count": 3,
                "average_sentiment": 0.67,
                "average_rating": 4.5,
                "positive_reviews": 2,
                "negative_reviews": 1,
            }
        }
        repo.add_review.return_value = 99
        return repo

    def build_watchmode(self):
        watchmode = Mock()
        watchmode.get_popular_movies.return_value = [
            {
                "id": 101,
                "title": "Test Movie",
                "type": "movie",
                "year": 2024,
                "plot_overview": "A test movie",
                "poster": "https://image.tmdb.org/test.jpg",
                "user_rating": 7.8,
                "critic_score": 81,
                "genre_names": ["Drama"],
            }
        ]
        watchmode.get_title_details.return_value = {
            "id": 101,
            "title": "Test Movie",
            "type": "movie",
            "year": 2024,
            "plot_overview": "A test movie",
            "poster": "https://image.tmdb.org/test.jpg",
            "backdrop": "https://image.tmdb.org/backdrop.jpg",
            "user_rating": 7.8,
            "critic_score": 81,
            "runtime_minutes": 120,
            "genre_names": ["Drama"],
            "release_date": "2024-01-01",
        }
        watchmode.search_titles.return_value = watchmode.get_popular_movies.return_value
        return watchmode

    @patch("flask_app.app.get_repository")
    def test_api_stats_endpoint(self, mock_repo_factory):
        mock_repo_factory.return_value = self.build_repo()
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["total_reviews"], 3)

    @patch("flask_app.app.get_watchmode")
    @patch("flask_app.app.get_repository")
    def test_api_movies_endpoint(self, mock_repo_factory, mock_watchmode_factory):
        mock_repo_factory.return_value = self.build_repo()
        mock_watchmode_factory.return_value = self.build_watchmode()
        response = self.client.get("/api/movies")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], 101)
        self.assertEqual(data[0]["review_count"], 3)

    @patch("flask_app.app.get_watchmode")
    @patch("flask_app.app.get_repository")
    def test_api_movie_detail_endpoint(self, mock_repo_factory, mock_watchmode_factory):
        mock_repo_factory.return_value = self.build_repo()
        mock_watchmode_factory.return_value = self.build_watchmode()
        response = self.client.get("/api/movies/101")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["title"], "Test Movie")
        self.assertEqual(data["review_count"], 3)

    @patch("flask_app.app.build_prediction_payload", return_value=1)
    @patch("flask_app.app.get_watchmode")
    @patch("flask_app.app.get_repository")
    def test_api_predict_sentiment(self, mock_repo_factory, mock_watchmode_factory, _mock_prediction):
        mock_repo_factory.return_value = self.build_repo()
        mock_watchmode_factory.return_value = self.build_watchmode()
        payload = {"text": "This is a great movie that I absolutely loved watching!", "movie_id": 101, "is_anonymous": True}
        response = self.client.post("/api/predict-sentiment", json=payload, content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["sentiment"], 1)
        self.assertEqual(data["review_id"], 99)

    @patch("flask_app.app.build_prediction_payload", side_effect=ValueError("Review is too short. Please write more."))
    @patch("flask_app.app.get_watchmode")
    @patch("flask_app.app.get_repository")
    def test_api_predict_sentiment_short_text(self, mock_repo_factory, mock_watchmode_factory, _mock_prediction):
        mock_repo_factory.return_value = self.build_repo()
        mock_watchmode_factory.return_value = self.build_watchmode()
        payload = {"text": "Bad movie", "movie_id": 101, "is_anonymous": True}
        response = self.client.post("/api/predict-sentiment", json=payload, content_type="application/json")
        self.assertEqual(response.status_code, 400)

    @patch("flask_app.app.get_repository")
    def test_api_health_endpoint(self, mock_repo_factory):
        mock_repo_factory.return_value = self.build_repo()
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["database"]["ready"])

    @patch("flask_app.app.render_template", return_value="ok")
    @patch("flask_app.app.build_prediction_payload", return_value=1)
    def test_post_form_fallback(self, _mock_prediction, _mock_render_template):
        response = self.client.post("/predict", data={"text": "This is a wonderful movie experience!"})
        self.assertEqual(response.status_code, 200)

    @patch("flask_app.app.build_prediction_payload", return_value=1)
    @patch("flask_app.app.get_watchmode")
    @patch("flask_app.app.get_repository")
    def test_api_predict_sentiment_with_display_name(self, mock_repo_factory, mock_watchmode_factory, _mock_prediction):
        repo = self.build_repo()
        mock_repo_factory.return_value = repo
        mock_watchmode_factory.return_value = self.build_watchmode()

        payload = {
            "text": "This is a great movie that I absolutely loved watching!",
            "movie_id": 101,
            "user_id": "user_101",
            "display_name": "Prateek",
            "is_anonymous": False,
            "rating": 5,
        }
        response = self.client.post("/api/predict-sentiment", json=payload, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["display_name"], "Prateek")
        self.assertFalse(data["is_anonymous"])
        repo.add_review.assert_called_once_with(101, payload["text"], 1, "user_101", "Prateek", False, rating=5)

    def test_api_predict_sentiment_named_review_requires_display_name(self):
        payload = {
            "text": "This is a great movie that I absolutely loved watching!",
            "movie_id": 101,
            "is_anonymous": False,
        }
        response = self.client.post("/api/predict-sentiment", json=payload, content_type="application/json")

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("display_name", data["error"])


class WatchModeFallbackTests(unittest.TestCase):
    def test_get_popular_movies_falls_back_to_search_results(self):
        from flask_app.watchmode_service import WatchModeAPIError, WatchModeService

        service = WatchModeService(api_key="test-key")
        with patch.object(service, "list_titles", side_effect=WatchModeAPIError("bad list request")):
            with patch.object(
                service,
                "search_titles",
                side_effect=[
                    [{"id": 1, "title": "Movie One", "type": "movie"}],
                    [{"id": 1, "title": "Movie One", "type": "movie"}, {"id": 2, "title": "Movie Two", "type": "movie"}],
                    [],
                    [],
                    [],
                    [],
                ],
            ):
                movies = service.get_popular_movies(limit=5, region="US")

        self.assertEqual(len(movies), 2)
        self.assertEqual(movies[0]["id"], 1)
        self.assertEqual(movies[1]["id"], 2)

    @patch("flask_app.app.get_repository")
    @patch("flask_app.app.get_watchmode")
    def test_catalog_movies_use_watchmode_catalog_without_media_enrichment(self, mock_watchmode_factory, mock_repo_factory):
        from flask_app.app import get_catalog_movies

        repo = Mock()
        repo.list_movie_stats.return_value = {}
        mock_repo_factory.return_value = repo

        watchmode = Mock()
        watchmode.get_popular_movies.return_value = [
            {
                "id": 101,
                "title": "Posterless Movie",
                "type": "movie",
                "year": 2024,
                "plot_overview": "",
                "poster": None,
            }
        ]
        mock_watchmode_factory.return_value = watchmode

        movies = get_catalog_movies(limit=1)

        self.assertEqual(movies[0]["poster_url"], None)
        watchmode.get_title_details.assert_not_called()


if __name__ == "__main__":
    unittest.main()
