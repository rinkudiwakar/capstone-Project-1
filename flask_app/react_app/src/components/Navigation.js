import React from 'react';
import { Link } from 'react-router-dom';

function Navigation({ stats, displayName, useAnonymous, onToggleAnonymous }) {
  return (
    <nav className="sticky top-0 z-50 bg-white/70 backdrop-blur-2xl border-b border-slate-200/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-wrap justify-between items-center gap-4 min-h-[5.5rem] py-4">
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="w-11 h-11 rounded-2xl bg-slate-950 text-white flex items-center justify-center shadow-lg">
              <span className="text-sm font-semibold tracking-[0.24em]">MS</span>
            </div>
            <div>
              <div className="text-xl font-semibold tracking-tight text-slate-950">MovieSentiment</div>
              <div className="text-xs uppercase tracking-[0.24em] text-slate-400">Find. Rate. Return.</div>
            </div>
          </Link>

          {stats && (
            <div className="hidden lg:grid grid-cols-4 gap-3">
              <div className="glass-panel rounded-2xl px-4 py-3 min-w-[120px]">
                <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Reviews</div>
                <div className="text-2xl font-semibold text-slate-950">{stats.total_reviews}</div>
              </div>
              <div className="glass-panel rounded-2xl px-4 py-3 min-w-[120px]">
                <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Movies</div>
                <div className="text-2xl font-semibold text-slate-950">{stats.reviewed_movies}</div>
              </div>
              <div className="glass-panel rounded-2xl px-4 py-3 min-w-[120px]">
                <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Positive</div>
                <div className="text-2xl font-semibold text-slate-950">
                  {((stats.overall_sentiment || 0) * 100).toFixed(0)}%
                </div>
              </div>
              <div className="glass-panel rounded-2xl px-4 py-3 min-w-[120px]">
                <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Avg rating</div>
                <div className="text-2xl font-semibold text-slate-950">
                  {stats.overall_rating ? stats.overall_rating.toFixed(1) : '0.0'}
                </div>
              </div>
            </div>
          )}

          <div className="flex items-center gap-3">
            <Link
              to="/search"
              className="px-4 py-2.5 rounded-full bg-slate-900 text-white text-sm font-medium hover:bg-slate-700 transition"
            >
              Search movies
            </Link>

            <button
              onClick={onToggleAnonymous}
              className={`px-4 py-2.5 rounded-full text-sm font-medium transition ${
                useAnonymous
                  ? 'bg-slate-100 text-slate-900 border border-slate-200'
                  : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
              }`}
              title="Toggle between anonymous and registered reviews"
            >
              {useAnonymous ? 'Anonymous mode' : 'Named reviews'}
            </button>

            {!useAnonymous && displayName && (
              <div className="hidden sm:flex items-center rounded-full border border-slate-200 bg-white px-4 py-2 text-xs text-slate-500">
                Publishing as {displayName}
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navigation;
