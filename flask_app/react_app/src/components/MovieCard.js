import React from 'react';

function MovieCard({ movie, onClick }) {
  const sentimentPercentage = movie.average_sentiment ? Math.round(movie.average_sentiment * 100) : null;
  const averageRating = movie.average_rating ? movie.average_rating.toFixed(1) : null;
  const initials = movie.title
    .split(' ')
    .slice(0, 2)
    .map((part) => part[0])
    .join('')
    .toUpperCase();

  return (
    <div
      onClick={onClick}
      className="group cursor-pointer glass-panel rounded-[2rem] overflow-hidden transition duration-300 transform hover:-translate-y-1"
    >
      <div className="relative overflow-hidden h-64 sm:h-72 bg-[radial-gradient(circle_at_top,_rgba(148,163,184,0.38),_transparent_34%),linear-gradient(135deg,#ffffff,#dbeafe_58%,#cbd5e1)]">
        <div className="absolute inset-0 p-6 flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <span className="inline-flex items-center rounded-full bg-white/70 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em] text-slate-500">
              {movie.movie_type || 'movie'}
            </span>
            <div className="w-16 h-16 rounded-full bg-white/45 text-slate-900 flex items-center justify-center text-xl font-semibold shadow-sm animate-drift">
              {initials}
            </div>
          </div>

          <div>
            <h3 className="text-slate-950 font-semibold text-2xl tracking-tight max-w-[12rem]">
              {movie.title}
            </h3>
            <p className="mt-2 text-sm text-slate-600 line-clamp-3">
              {movie.description || 'Open this title to leave a rating, write a review, and see how everyone else felt about it.'}
            </p>
          </div>
        </div>
      </div>

      <div className="p-5 sm:p-6 bg-white/78">
        <div className="flex items-center justify-between text-sm text-slate-500 mb-4">
          <span>{movie.release_year || 'Coming soon'}</span>
          <span>{(movie.genres || []).slice(0, 2).join(' - ') || 'Fresh picks'}</span>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-2xl bg-slate-950 text-white p-3">
            <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Reviews</div>
            <div className="text-xl font-semibold mt-1">{movie.review_count || 0}</div>
          </div>
          <div className="rounded-2xl bg-slate-100 p-3">
            <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Sentiment</div>
            <div className="text-xl font-semibold mt-1 text-slate-950">
              {sentimentPercentage !== null ? `${sentimentPercentage}%` : 'New'}
            </div>
          </div>
          <div className="rounded-2xl bg-slate-100 p-3">
            <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Rating</div>
            <div className="text-xl font-semibold mt-1 text-slate-950">{averageRating || 'N/A'}</div>
          </div>
        </div>

        <button className="mt-5 w-full rounded-full bg-slate-900 text-white py-3 text-sm font-medium tracking-wide transition group-hover:bg-slate-700">
          Open and rate
        </button>
      </div>
    </div>
  );
}

export default MovieCard;
