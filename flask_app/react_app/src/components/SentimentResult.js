import React, { useEffect, useState } from 'react';

function SentimentResult({ sentiment, onClose }) {
  const [show, setShow] = useState(true);
  const isPositive = sentiment.sentiment === 1;

  useEffect(() => {
    const timer = setTimeout(() => {
      setShow(false);
      onClose();
    }, 4000);

    return () => clearTimeout(timer);
  }, [onClose]);

  if (!show) return null;

  return (
    <div className="mb-6 overflow-hidden">
      <div
        className={`animate-slide-in-down rounded-[1.75rem] p-8 border glass-panel ${
          isPositive ? 'border-emerald-200 bg-emerald-50/75' : 'border-rose-200 bg-rose-50/75'
        }`}
      >
        <div className="flex items-center justify-center space-x-4 mb-4">
          <div
            className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl font-semibold ${
              isPositive ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'
            }`}
          >
            {isPositive ? 'P' : 'N'}
          </div>
          <div>
            <h3 className="text-3xl font-semibold text-slate-950 tracking-tight">
              {isPositive ? 'Positive sentiment' : 'Negative sentiment'}
            </h3>
            <p className="text-base text-slate-600">
              {isPositive
                ? 'Your review reads as clearly favorable.'
                : 'Your review reads as critical or disappointed.'}
            </p>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 text-center">
          <div className="bg-white/75 rounded-2xl p-4 border border-slate-200">
            <p className="text-xs text-slate-400 uppercase tracking-[0.24em]">Verdict</p>
            <p className="text-2xl font-semibold text-slate-950 mt-1">
              {isPositive ? 'Loved it' : 'Not for me'}
            </p>
          </div>
          <div className="bg-white/75 rounded-2xl p-4 border border-slate-200">
            <p className="text-xs text-slate-400 uppercase tracking-[0.24em]">Rating</p>
            <p className="text-xl font-semibold text-slate-950 mt-1">
              {sentiment.rating ? `${sentiment.rating}/5` : 'Not rated'}
            </p>
          </div>
        </div>

        <p className="text-center text-sm mt-4 text-slate-500">
          Saved to the platform and closing in a moment.
        </p>
      </div>
    </div>
  );
}

export default SentimentResult;
