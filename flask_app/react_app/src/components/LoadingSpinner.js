import React from 'react';

function LoadingSpinner() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      <div className="glass-panel rounded-[2rem] px-10 py-12 text-center max-w-md w-full">
        <div className="w-14 h-14 border-4 border-slate-200 border-t-slate-900 rounded-full animate-spin mx-auto mb-6"></div>
        <h2 className="text-2xl font-semibold text-slate-950 mb-2">Preparing your next watch</h2>
        <p className="text-slate-500">Loading movies, community reviews, and sentiment insights.</p>
      </div>
    </div>
  );
}

export default LoadingSpinner;
