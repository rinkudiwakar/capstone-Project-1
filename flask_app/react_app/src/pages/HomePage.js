import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { movieAPI } from '../services/api';
import MovieCard from '../components/MovieCard';
import LoadingSpinner from '../components/LoadingSpinner';

function HomePage() {
  const [movies, setMovies] = useState([]);
  const [activeMovieIndex, setActiveMovieIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [carouselPaused, setCarouselPaused] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const loadMovies = async () => {
      try {
        setLoading(true);
        const data = await movieAPI.getAllMovies(20);
        setMovies(data);
      } catch (err) {
        setError('Failed to load movies. Please try again.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadMovies();
  }, []);

  useEffect(() => {
    if (carouselPaused || movies.length <= 1) {
      return undefined;
    }

    const timer = setInterval(() => {
      setActiveMovieIndex((currentIndex) => (currentIndex + 1) % movies.length);
    }, 3800);

    return () => clearInterval(timer);
  }, [carouselPaused, movies]);

  const handleMovieClick = (movieId) => {
    setCarouselPaused(true);
    navigate(`/review/${movieId}`);
  };

  if (loading) return <LoadingSpinner />;

  const featuredMovie = movies[activeMovieIndex] || null;
  const featuredMovies = movies.slice(0, 5);

  return (
    <div className="min-h-screen pt-8 pb-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {featuredMovie && (
          <div className="mb-14 animate-fade-in">
            <div className="glass-panel rounded-[2.5rem] overflow-hidden p-8 md:p-10">
              <div className="grid lg:grid-cols-[1.35fr_0.8fr] gap-10 items-start">
                <div>
                  <div className="inline-flex items-center rounded-full bg-slate-900 text-white px-4 py-2 text-xs font-medium tracking-[0.24em] uppercase">
                    Now rotating
                  </div>
                  <h1 className="section-heading text-slate-950 mt-6">
                    A calmer place to discover, rate, and revisit films.
                  </h1>
                  <p className="muted-copy text-lg max-w-2xl mt-5">
                    The featured spotlight keeps moving through top titles until you choose one to review. Search stays open-ended for any movie you want to find.
                  </p>

                  <div className="mt-10 rounded-[2rem] bg-slate-950 text-white p-8 relative overflow-hidden">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_rgba(96,165,250,0.35),_transparent_35%),radial-gradient(circle_at_bottom_left,_rgba(255,255,255,0.08),_transparent_28%)]" />
                    <div className="relative">
                      <p className="text-xs uppercase tracking-[0.26em] text-slate-400">Featured title</p>
                      <h2 className="text-4xl md:text-5xl font-semibold tracking-tight mt-3">
                        {featuredMovie.title}
                      </h2>
                      <p className="text-slate-300 text-lg mt-4 max-w-2xl line-clamp-3">
                        {featuredMovie.description || 'Tap into the conversation around this movie, leave your own review, and let the model read the sentiment.'}
                      </p>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-8">
                        <div className="rounded-2xl bg-white/8 p-4">
                          <div className="text-xs uppercase tracking-[0.22em] text-slate-400">Year</div>
                          <div className="text-2xl font-semibold mt-2">{featuredMovie.release_year || 'Soon'}</div>
                        </div>
                        <div className="rounded-2xl bg-white/8 p-4">
                          <div className="text-xs uppercase tracking-[0.22em] text-slate-400">Reviews</div>
                          <div className="text-2xl font-semibold mt-2">{featuredMovie.review_count || 0}</div>
                        </div>
                        <div className="rounded-2xl bg-white/8 p-4">
                          <div className="text-xs uppercase tracking-[0.22em] text-slate-400">Sentiment</div>
                          <div className="text-2xl font-semibold mt-2">
                            {featuredMovie.average_sentiment ? `${Math.round(featuredMovie.average_sentiment * 100)}%` : 'New'}
                          </div>
                        </div>
                        <div className="rounded-2xl bg-white/8 p-4">
                          <div className="text-xs uppercase tracking-[0.22em] text-slate-400">Rating</div>
                          <div className="text-2xl font-semibold mt-2">
                            {featuredMovie.average_rating ? featuredMovie.average_rating.toFixed(1) : 'N/A'}
                          </div>
                        </div>
                      </div>

                      <div className="mt-8 flex flex-wrap gap-3">
                        {(featuredMovie.genres || []).slice(0, 4).map((genre) => (
                          <span key={genre} className="rounded-full border border-white/15 bg-white/8 px-4 py-2 text-sm text-slate-200">
                            {genre}
                          </span>
                        ))}
                      </div>

                      <button
                        onClick={() => handleMovieClick(featuredMovie.id)}
                        className="mt-8 rounded-full bg-white text-slate-950 px-6 py-3 text-sm font-semibold hover:bg-slate-200 transition"
                      >
                        Rate this movie
                      </button>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Top picks</p>
                      <h3 className="text-2xl font-semibold tracking-tight text-slate-950 mt-2">Keep scrolling</h3>
                    </div>
                    <button
                      onClick={() => setCarouselPaused((current) => !current)}
                      className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-600 hover:bg-slate-50 transition"
                    >
                      {carouselPaused ? 'Resume' : 'Pause'}
                    </button>
                  </div>

                  {featuredMovies.map((movie, index) => {
                    const isActive = movie.id === featuredMovie.id;
                    return (
                      <button
                        key={movie.id}
                        type="button"
                        onClick={() => {
                          setActiveMovieIndex(index);
                          setCarouselPaused(true);
                          handleMovieClick(movie.id);
                        }}
                        className={`w-full text-left rounded-[1.5rem] px-5 py-4 transition border ${
                          isActive
                            ? 'bg-slate-950 text-white border-slate-950'
                            : 'bg-white/70 border-slate-200 text-slate-900 hover:bg-white'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-4">
                          <div>
                            <div className="text-lg font-semibold tracking-tight">{movie.title}</div>
                            <div className={`text-sm mt-1 ${isActive ? 'text-slate-300' : 'text-slate-500'}`}>
                              {movie.release_year || 'Coming soon'} - {(movie.genres || []).slice(0, 2).join(' - ') || 'Discover'}
                            </div>
                          </div>
                          <div className={`text-sm font-medium ${isActive ? 'text-white' : 'text-slate-500'}`}>
                            {movie.average_rating ? `${movie.average_rating.toFixed(1)}/5` : 'Rate'}
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        <div>
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-8">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Browse</p>
              <h2 className="section-heading text-slate-950 mt-2">
                20 movies to review today
              </h2>
            </div>
            <p className="text-slate-500 max-w-lg">
              Home stays focused on a curated list of 20. Use search to reach any movie in the wider catalog.
            </p>
          </div>

          {error && (
            <div className="bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 rounded-2xl mb-6">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {movies.map((movie) => (
              <MovieCard key={movie.id} movie={movie} onClick={() => handleMovieClick(movie.id)} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
