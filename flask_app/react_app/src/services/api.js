import axios from 'axios';

const resolveApiBaseUrl = () => {
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  if (typeof window !== 'undefined' && window.location?.origin) {
    return window.location.origin;
  }
  return 'http://localhost:5000';
};

export const API_BASE_URL = resolveApiBaseUrl();

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API Service
export const movieAPI = {
  // Get all movies
  getAllMovies: async (limit = 20) => {
    try {
      const response = await api.get('/api/movies', {
        params: { limit },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching movies:', error);
      throw error;
    }
  },

  // Get single movie
  getMovieById: async (movieId) => {
    try {
      const response = await api.get(`/api/movies/${movieId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching movie:', error);
      throw error;
    }
  },

  // Search movies
  searchMovies: async (query) => {
    try {
      const response = await api.get('/api/search-movies', {
        params: { q: query },
      });
      return response.data;
    } catch (error) {
      console.error('Error searching movies:', error);
      throw error;
    }
  },

  // Get reviews for a movie
  getMovieReviews: async (movieId, limit = 10) => {
    try {
      const response = await api.get(`/api/reviews/${movieId}`, {
        params: { limit },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching reviews:', error);
      throw error;
    }
  },

  // Get movie sentiment
  getMovieSentiment: async (movieId) => {
    try {
      const response = await api.get(`/api/movies/${movieId}/sentiment`);
      return response.data;
    } catch (error) {
      console.error('Error fetching sentiment:', error);
      throw error;
    }
  },

  // Predict sentiment
  predictSentiment: async (reviewText, movieId, userId = null, isAnonymous = true, rating = null, displayName = null) => {
    try {
      const response = await api.post('/api/predict-sentiment', {
        text: reviewText,
        movie_id: movieId,
        user_id: userId,
        is_anonymous: isAnonymous,
        rating,
        display_name: displayName,
      });
      return response.data;
    } catch (error) {
      console.error('Error predicting sentiment:', error);
      throw error;
    }
  },

  // Get overall stats
  getStats: async () => {
    try {
      const response = await api.get('/api/stats');
      return response.data;
    } catch (error) {
      console.error('Error fetching stats:', error);
      throw error;
    }
  },
};

export default api;
