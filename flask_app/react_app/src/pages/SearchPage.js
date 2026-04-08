import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { movieAPI } from '../services/api';
import MovieCard from '../components/MovieCard';

function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const navigate = useNavigate();

  const handleSearch = async (event) => {
    event.preventDefault();

    if (!query.trim() || query.trim().length < 2) {
      alert('Please enter at least 2 characters');
      return;
    }

    try {
      setLoading(true);
      const data = await movieAPI.searchMovies(query);
      setResults(data);
      setSearched(true);
    } catch (error) {
      console.error('Search failed:', error);
      alert('Search failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleMovieClick = (movieId) => {
    navigate(`/review/${movieId}`);
  };

  return (
    <div className="min-h-screen pt-8 pb-16">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-12">
          <h1 className="section-heading text-slate-950 mb-6 text-center">
            Search every movie
          </h1>

          <form onSubmit={handleSearch} className="max-w-2xl mx-auto">
            <div className="relative glass-panel rounded-full p-2">
              <input
                type="text"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search for any movie title..."
                className="w-full px-6 py-4 bg-transparent rounded-full text-slate-900 placeholder-slate-400 focus:outline-none text-lg"
              />
              <button
                type="submit"
                className="absolute right-2 top-1/2 -translate-y-1/2 px-6 py-3 bg-slate-900 text-white font-semibold rounded-full hover:bg-slate-700 transition"
              >
                Search
              </button>
            </div>
          </form>

          <div className="mt-6 text-center text-slate-500 text-sm">
            Search is open-ended here, so you can look for any movie instead of only the homepage set.
          </div>
        </div>

        {loading && (
          <div className="flex justify-center items-center py-12">
            <div className="text-center">
              <div className="w-12 h-12 border-4 border-slate-200 border-t-slate-900 rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-slate-700">Searching...</p>
            </div>
          </div>
        )}

        {searched && !loading && (
          <div>
            <h2 className="text-2xl font-semibold text-slate-950 mb-6">
              {results.length === 0
                ? `No movies found for "${query}"`
                : `Found ${results.length} movie${results.length !== 1 ? 's' : ''}`}
            </h2>

            {results.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {results.map((movie) => (
                  <MovieCard
                    key={movie.id}
                    movie={movie}
                    onClick={() => handleMovieClick(movie.id)}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <p className="text-slate-500 text-lg mb-4">
                  Try searching with different keywords or browse all movies on the home page.
                </p>
                <a
                  href="/"
                  className="inline-block px-6 py-3 bg-slate-900 text-white font-semibold rounded-full hover:bg-slate-700 transition"
                >
                  Back to home
                </a>
              </div>
            )}
          </div>
        )}

        {!searched && (
          <div className="text-center py-12">
            <p className="text-slate-500 text-lg">
              Start with a title and jump straight into the review flow.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default SearchPage;
