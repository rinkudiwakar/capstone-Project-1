import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { movieAPI } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import SentimentResult from '../components/SentimentResult';

function ReviewPage({ userId, displayName, useAnonymous, onUserSet, onRefreshStats }) {
  const { movieId } = useParams();
  const [movie, setMovie] = useState(null);
  const [reviewText, setReviewText] = useState('');
  const [sentiment, setSentiment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [tempDisplayName, setTempDisplayName] = useState(displayName || '');
  const [charCount, setCharCount] = useState(0);
  const [rating, setRating] = useState(4);

  useEffect(() => {
    setTempDisplayName(displayName || '');
  }, [displayName]);

  useEffect(() => {
    const loadPage = async () => {
      try {
        const [movieData, reviewData] = await Promise.all([
          movieAPI.getMovieById(movieId),
          movieAPI.getMovieReviews(movieId),
        ]);
        setMovie(movieData);
        setReviews(reviewData);
      } catch (err) {
        setError('Failed to load movie. Please try again.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadPage();
  }, [movieId]);

  const handleSubmitReview = async () => {
    const normalizedReview = reviewText.trim();
    const currentDisplayName = (displayName || tempDisplayName || '').trim();

    if (!normalizedReview) {
      setError('Please write a review');
      return;
    }
    if (normalizedReview.length < 20) {
      setError('Review must be at least 20 characters long');
      return;
    }
    if (!useAnonymous && !currentDisplayName) {
      setError('Please enter the name you want to publish with or switch to anonymous mode');
      return;
    }

    try {
      setSubmitting(true);
      setError(null);
      const currentUserId = useAnonymous
        ? null
        : (userId || window.crypto?.randomUUID?.() || `user_${Date.now()}`);

      if (!useAnonymous && currentDisplayName !== displayName) {
        onUserSet(currentDisplayName, currentUserId);
      }

      const result = await movieAPI.predictSentiment(
        normalizedReview,
        movieId,
        currentUserId,
        useAnonymous,
        rating,
        useAnonymous ? null : currentDisplayName,
      );
      setSentiment(result);
      setReviewText('');
      setCharCount(0);

      const [movieData, reviewData] = await Promise.all([
        movieAPI.getMovieById(movieId),
        movieAPI.getMovieReviews(movieId),
        onRefreshStats(),
      ]);
      setMovie(movieData);
      setReviews(reviewData);
    } catch (err) {
      setError(err?.response?.data?.error || 'Failed to submit review. Please try again.');
      console.error(err);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <LoadingSpinner />;

  if (!movie) {
    return (
      <div className="min-h-screen pt-20 flex flex-col items-center justify-center">
        <h1 className="text-2xl text-slate-900 mb-4">Movie not found</h1>
        <Link to="/" className="text-slate-500 hover:text-slate-900 underline">
          Back to home
        </Link>
      </div>
    );
  }

  const initials = movie.title
    .split(' ')
    .slice(0, 2)
    .map((part) => part[0])
    .join('')
    .toUpperCase();

  return (
    <div className="min-h-screen pt-8 pb-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <Link to="/" className="text-slate-500 hover:text-slate-900 transition mb-8 inline-flex items-center">
          Back to Movies
        </Link>

        <div className="glass-panel rounded-[2.5rem] p-6 md:p-10 mb-8">
          <div className="grid md:grid-cols-[1.4fr_0.8fr] gap-8 items-start">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Now reviewing</p>
              <h1 className="text-4xl md:text-5xl font-semibold tracking-tight text-slate-950 mt-3">{movie.title}</h1>
              <p className="text-slate-500 text-lg mt-3">
                {movie.release_year || 'Coming soon'} - {(movie.genres || []).slice(0, 3).join(' - ') || 'Movie'}
              </p>
              <p className="text-slate-600 text-base leading-relaxed mt-6">
                {movie.description || 'No description available yet for this title.'}
              </p>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
                <div className="rounded-2xl bg-slate-950 text-white p-4">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Reviews</div>
                  <div className="text-2xl font-semibold mt-2">{movie.review_count || 0}</div>
                </div>
                <div className="rounded-2xl bg-slate-100 p-4">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Sentiment</div>
                  <div className="text-2xl font-semibold mt-2 text-slate-950">
                    {movie.average_sentiment ? `${(movie.average_sentiment * 100).toFixed(0)}%` : 'N/A'}
                  </div>
                </div>
                <div className="rounded-2xl bg-slate-100 p-4">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Community rating</div>
                  <div className="text-2xl font-semibold mt-2 text-slate-950">
                    {movie.average_rating ? movie.average_rating.toFixed(1) : 'N/A'}
                  </div>
                </div>
                <div className="rounded-2xl bg-slate-100 p-4">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">WatchMode rating</div>
                  <div className="text-2xl font-semibold mt-2 text-slate-950">{movie.user_rating || 'N/A'}</div>
                </div>
              </div>
            </div>

            <div className="rounded-[2rem] bg-[radial-gradient(circle_at_top,_rgba(148,163,184,0.45),_transparent_35%),linear-gradient(135deg,#ffffff,#e2e8f0_60%,#cbd5e1)] p-8 min-h-[300px] flex flex-col justify-between">
              <div className="w-20 h-20 rounded-full bg-white/70 shadow-sm flex items-center justify-center text-2xl font-semibold text-slate-900 animate-drift">
                {initials}
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Why rate here</p>
                <p className="text-slate-700 mt-3 leading-relaxed">
                  Your review, rating, and predicted sentiment are saved together so the next visitor sees both the movie context and the audience pulse around it.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="glass-panel rounded-[2.5rem] p-8 mb-8">
          <h2 className="text-2xl font-semibold tracking-tight text-slate-950 mb-6">Share your review</h2>

          {error && <div className="bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 rounded-2xl mb-6">{error}</div>}
          {sentiment && <SentimentResult sentiment={sentiment} onClose={() => setSentiment(null)} />}

          {!useAnonymous && (
            <div className="mb-6 bg-slate-50 border border-slate-200 rounded-2xl p-4">
              <label className="block text-slate-900 text-sm font-semibold mb-2">Your public name</label>
              <input
                type="text"
                value={tempDisplayName}
                onChange={(event) => setTempDisplayName(event.target.value)}
                placeholder="How should your review appear to others?"
                className="w-full px-4 py-3 bg-white border border-slate-200 rounded-2xl text-slate-900 placeholder-slate-400 focus:outline-none focus:border-slate-900"
              />
              <p className="text-xs text-slate-400 mt-2">
                Choose any display name you want other viewers to see next to your review.
              </p>
            </div>
          )}

          <div className="mb-6">
            <label className="block text-slate-900 text-sm font-semibold mb-3">Your rating</label>
            <div className="flex items-center gap-3 flex-wrap">
              {[1, 2, 3, 4, 5].map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setRating(value)}
                  className={`w-11 h-11 rounded-full border text-sm font-semibold transition ${
                    rating >= value
                      ? 'bg-slate-900 text-white border-slate-900'
                      : 'bg-white text-slate-500 border-slate-200 hover:border-slate-400'
                  }`}
                >
                  {value}
                </button>
              ))}
              <span className="text-sm text-slate-500">Saved along with your review and sentiment.</span>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-slate-900 text-sm font-semibold mb-3">Your honest review</label>
            <textarea
              value={reviewText}
              onChange={(event) => {
                setReviewText(event.target.value);
                setCharCount(event.target.value.length);
              }}
              placeholder="Write what you think about this movie..."
              className="w-full h-40 px-4 py-3 bg-white border border-slate-200 rounded-[1.5rem] text-slate-900 placeholder-slate-400 focus:outline-none focus:border-slate-900 resize-none"
            />
            <div className="flex justify-between items-center mt-2">
              <span className="text-xs text-slate-400">Write at least 20 characters for better analysis</span>
              <span className={`text-sm ${charCount < 20 ? 'text-rose-500' : 'text-emerald-600'}`}>{charCount}/500</span>
            </div>
          </div>

          <button
            onClick={handleSubmitReview}
            disabled={submitting || charCount < 20}
            className={`w-full py-3.5 px-6 font-semibold text-base rounded-full transition duration-200 ${
              submitting || charCount < 20
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-slate-900 text-white hover:bg-slate-700'
            }`}
          >
            {submitting ? 'Saving your review...' : useAnonymous ? 'Save anonymous review' : 'Publish review and rate movie'}
          </button>
        </div>

        {reviews.length > 0 && (
          <div className="glass-panel rounded-[2.5rem] p-8">
            <h2 className="text-2xl font-semibold tracking-tight text-slate-950 mb-6">Recent reviews</h2>
            <div className="space-y-4">
              {reviews.slice(0, 5).map((review) => (
                <div key={review.id} className="bg-white/70 border border-slate-200 rounded-[1.5rem] p-5 transition">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <span className="text-slate-500 text-sm">
                        {review.is_anonymous ? 'Anonymous' : (review.display_name || review.external_user_id || 'Named reviewer')}
                      </span>
                      <div className="text-xs text-slate-400 mt-1">
                        {review.rating ? `${review.rating}/5 rating` : 'No rating provided'}
                      </div>
                    </div>
                    <span
                      className={`text-sm font-semibold px-3 py-1 rounded-full ${
                        review.sentiment === 1 ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'
                      }`}
                    >
                      {review.sentiment === 1 ? 'Positive' : 'Negative'}
                    </span>
                  </div>
                  <p className="text-slate-700 text-sm leading-6">{review.review_text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ReviewPage;
