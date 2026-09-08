import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE = 'http://localhost:5000/api';

const POPULAR_TICKERS = [
  { label: 'NIFTY 50', symbol: '^NSEI' },
  { label: 'S&P 500', symbol: '^GSPC' },
  { label: 'Apple Inc.', symbol: 'AAPL' },
  { label: 'Microsoft', symbol: 'MSFT' },
  { label: 'Reliance Ind.', symbol: 'RELIANCE.NS' },
  { label: 'Tesla Inc.', symbol: 'TSLA' }
];

export default function App() {
  const [selectedTicker, setSelectedTicker] = useState('^NSEI');
  const [inputTicker, setInputTicker] = useState('');
  const [predictionData, setPredictionData] = useState(null); // ✅ Fixed: null instead of []
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch prediction from Express Backend
  const handlePredict = async (tickerToPredict) => {
    const ticker = (tickerToPredict || inputTicker || selectedTicker).trim().toUpperCase();
    if (!ticker) return;

    setSelectedTicker(ticker);
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/predict/${encodeURIComponent(ticker)}`); // ✅ Fixed space
      const data = await response.json();

      if (data.status === 'error') {
        throw new Error(data.message || 'Failed to generate prediction');
      }

      setPredictionData(data);
      fetchHistory();
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch history logs
  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history`);
      const data = await res.json();
      if (Array.isArray(data)) setHistory(data);
    } catch (e) {
      console.warn('Could not fetch history logs:', e);
    }
  };

  // Load initial prediction for ^NSEI on mount
  useEffect(() => {
    handlePredict('^NSEI');
    fetchHistory();
  }, []);

  return (
    <div className="app-container">
      {/* 1. Header */}
      <header className="header">
        <div className="brand">
          <span className="logo-icon">📈</span>
          <div>
            <h1>StockAI Predictor</h1>
            <p className="subtitle">Stacked LSTM Deep Learning & Quantitative Engine</p>
          </div>
        </div>
      </header>

      {/* 2. Ticker Selector & Search Bar */}
      <section className="search-section">
        <div className="quick-buttons">
          {POPULAR_TICKERS.map((t) => (
            <button
              key={t.symbol}
              className={`pill-btn ${selectedTicker === t.symbol ? 'active' : ''}`}
              onClick={() => handlePredict(t.symbol)}
            >
              {t.label} ({t.symbol})
            </button>
          ))}
        </div>

        <form
          className="search-bar"
          onSubmit={(e) => {
            e.preventDefault();
            handlePredict(inputTicker);
          }}
        >
          <input
            type="text"
            placeholder="Enter custom ticker (e.g. NVDA, TCS.NS, GOOGL)..."
            value={inputTicker}
            onChange={(e) => setInputTicker(e.target.value)}
          />
          <button type="submit" className="predict-btn" disabled={loading}>
            {loading ? 'Analyzing...' : 'Run Forecast ⚡'}
          </button>
        </form>
      </section>

      {/* 3. Error Banner */}
      {error && (
        <div className="error-banner">
          ⚠️ <strong>Error:</strong> {error}
        </div>
      )}

      {/* 4. Main Dashboard Grid */}
      {loading ? (
        <div className="loader-container">
          <div className="spinner"></div>
          <p>Running LSTM feature extraction & neural inference...</p>
        </div>
      ) : predictionData ? (
        <main className="dashboard-grid">
          {/* Card A: Prediction Badge */}
          <div className="card prediction-card">
            <div className="card-header">
              <h2>Tomorrow's AI Forecast</h2>
              <span className="ticker-badge">{predictionData.ticker}</span>
            </div>

            <div className={`direction-banner ${predictionData.prediction === 'UP' ? 'up' : 'down'}`}>
              <span className="arrow">{predictionData.prediction === 'UP' ? '▲' : '▼'}</span>
              <div>
                <h3>{predictionData.prediction === 'UP' ? 'BULLISH (UP)' : 'BEARISH (DOWN)'}</h3>
                <p>Confidence: <strong>{predictionData.confidence}%</strong></p>
              </div>
            </div>

            <div className="probability-bar-container">
              <div className="bar-labels">
                <span>Bearish (0%)</span>
                <span>Probability: {predictionData.probability_up}%</span>
                <span>Bullish (100%)</span>
              </div>
              <div className="progress-bg">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.min(Math.max(predictionData.probability_up || 50, 5), 100)}%` }}
                ></div>
              </div>
            </div>

            <div className="meta-info">
              <div>
                <span>Latest Close:</span>
                <strong>${predictionData.latest_close?.toLocaleString()}</strong>
              </div>
              <div>
                <span>Market Date:</span>
                <strong>{predictionData.latest_date}</strong>
              </div>
              <div>
                <span>Model Loaded:</span>
                <code>{predictionData.model_used}</code>
              </div>
            </div>
          </div>

          {/* Card B: Technical Indicators */}
          <div className="card indicators-card">
            <div className="card-header">
              <h2>Technical Indicators Context</h2>
              <span className="status-pill">Stationary Signals</span>
            </div>

            <div className="indicators-grid">
              <div className="indicator-box">
                <span className="ind-label">RSI (14 Days)</span>
                <span className="ind-value">{predictionData.indicators?.rsi_14}</span>
                <span className={`ind-tag ${predictionData.indicators?.rsi_14 > 70 ? 'overbought' : predictionData.indicators?.rsi_14 < 30 ? 'oversold' : 'neutral'}`}>
                  {predictionData.indicators?.rsi_14 > 70 ? 'Overbought (>70)' : predictionData.indicators?.rsi_14 < 30 ? 'Oversold (<30)' : 'Neutral'}
                </span>
              </div>

              <div className="indicator-box">
                <span className="ind-label">MACD Line</span>
                <span className="ind-value">{predictionData.indicators?.macd}</span>
                <span className={`ind-tag ${predictionData.indicators?.macd > 0 ? 'bullish' : 'bearish'}`}>
                  {predictionData.indicators?.macd > 0 ? 'Bullish Trend' : 'Bearish Trend'}
                </span>
              </div>

              <div className="indicator-box">
                <span className="ind-label">SMA-20 Trend</span>
                <span className="ind-value">${predictionData.indicators?.sma_20}</span>
                <span className="ind-tag neutral">Short-Term Baseline</span>
              </div>

              <div className="indicator-box">
                <span className="ind-label">EMA-50 Trend</span>
                <span className="ind-value">${predictionData.indicators?.ema_50}</span>
                <span className="ind-tag neutral">Intermediate Baseline</span>
              </div>
            </div>
          </div>
        </main>
      ) : null}

      {/* 5. History Table */}
      {history.length > 0 && (
        <section className="history-section card">
          <h2>🕒 Recent Predictions Log (MongoDB)</h2>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Forecast</th>
                  <th>Confidence</th>
                  <th>Latest Price</th>
                  <th>Market Date</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 6).map((item, idx) => (
                  <tr key={idx}>
                    <td><strong>{item.ticker}</strong></td>
                    <td>
                      <span className={`badge ${item.prediction === 'UP' ? 'badge-up' : 'badge-down'}`}>
                        {item.prediction}
                      </span>
                    </td>
                    <td>{item.confidence}%</td>
                    <td>${item.latest_close?.toFixed(2)}</td>
                    <td>{item.latest_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 6. Footer */}
      <footer className="footer">
        <p>⚠️ <strong>Disclaimer:</strong> This application is developed for academic, educational, and research demonstration purposes. It does not constitute financial investment advice.</p>
      </footer>
    </div>
  );
}