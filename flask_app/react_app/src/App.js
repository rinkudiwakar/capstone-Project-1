import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ReviewPage from './pages/ReviewPage';
import SearchPage from './pages/SearchPage';
import Navigation from './components/Navigation';
import { movieAPI } from './services/api';

function App() {
  const [stats, setStats] = useState(null);
  const [userId, setUserId] = useState(null);
  const [displayName, setDisplayName] = useState('');
  const [useAnonymous, setUseAnonymous] = useState(true);

  useEffect(() => {
    const storedUserId = localStorage.getItem('movieAppUserId');
    const storedDisplayName = localStorage.getItem('movieAppDisplayName');
    if (storedUserId) {
      setUserId(storedUserId);
    }
    if (storedDisplayName) {
      setDisplayName(storedDisplayName);
    }
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const data = await movieAPI.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const handleSetUser = (name, explicitUserId = null) => {
    const normalizedName = (name || '').trim();
    if (!normalizedName) {
      return;
    }
    const stableUserId = explicitUserId || userId || window.crypto?.randomUUID?.() || `user_${Date.now()}`;
    setUserId(stableUserId);
    setDisplayName(normalizedName);
    localStorage.setItem('movieAppUserId', stableUserId);
    localStorage.setItem('movieAppDisplayName', normalizedName);
  };

  const handleToggleAnonymous = () => {
    setUseAnonymous(!useAnonymous);
  };

  return (
    <Router>
      <div className="min-h-screen app-shell">
        <Navigation
          stats={stats}
          displayName={displayName}
          useAnonymous={useAnonymous}
          onToggleAnonymous={handleToggleAnonymous}
        />

        <Routes>
          <Route
            path="/"
            element={<HomePage onRefreshStats={fetchStats} />}
          />
          <Route
            path="/review/:movieId"
            element={
              <ReviewPage
                userId={userId}
                displayName={displayName}
                useAnonymous={useAnonymous}
                onUserSet={handleSetUser}
                onRefreshStats={fetchStats}
              />
            }
          />
          <Route
            path="/search"
            element={<SearchPage />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
