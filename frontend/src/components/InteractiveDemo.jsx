import React, { useState } from 'react';
import { Search, Image as ImageIcon, SlidersHorizontal, Loader2 } from 'lucide-react';

export default function InteractiveDemo() {
  const [query, setQuery] = useState('');
  const [threshold, setThreshold] = useState(0.5);
  const [category, setCategory] = useState('etri');
  const [loading, setLoading] = useState(false);
  const [resultImage, setResultImage] = useState(null);
  const [error, setError] = useState('');

  const handleQuery = async () => {
    if (!query) return;
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:8000/api/query_gif', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text_query: query,
          threshold: parseFloat(threshold),
          category: category
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to process query');
      }

      setResultImage(`data:image/gif;base64,${data.image_base64}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="demo-container">
      <div className="page-header">
        <h1 className="page-title">Interactive Inference</h1>
        <p className="page-subtitle">Query the 3D scene using natural language</p>
      </div>

      <div className="grid-2">
        {/* Controls Panel */}
        <div className="glass-panel card">
          <div className="card-title">Query Parameters</div>
          
          <div className="form-group">
            <label><Search size={14} style={{display:'inline', marginRight:'4px'}}/> Text Query</label>
            <input 
              type="text" 
              value={query} 
              onChange={e => setQuery(e.target.value)} 
              placeholder="e.g., 'a red apple' or 'car'"
            />
          </div>

          <div className="form-group">
            <label>
              <SlidersHorizontal size={14} style={{display:'inline', marginRight:'4px'}}/> 
              Relevancy Threshold: {threshold}
            </label>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.01" 
              value={threshold} 
              onChange={e => setThreshold(e.target.value)} 
            />
          </div>

          <div className="form-group">
            <label><ImageIcon size={14} style={{display:'inline', marginRight:'4px'}}/> Dataset Category</label>
            <select value={category} onChange={e => setCategory(e.target.value)}>
              <option value="etri">ETRI Dataset (Driving)</option>
              <option value="lerf">LERF Dataset (Indoor)</option>
            </select>
          </div>

          <button 
            className="btn-primary" 
            onClick={handleQuery} 
            disabled={loading || !query}
          >
            {loading ? <><Loader2 className="spinner" size={18} /> Processing (Takes ~10s)...</> : 'Run Inference (GIF Mode)'}
          </button>
          
          {error && <div style={{ color: 'var(--danger-color)', marginTop: '16px', fontSize: '0.9rem' }}>{error}</div>}
        </div>

        {/* Results Panel */}
        <div className="glass-panel card">
          <div className="card-title">Query Result</div>
          <div className={`image-preview-container ${loading ? 'loading-pulse' : ''}`}>
            {resultImage ? (
              <img src={resultImage} alt="Query Result" />
            ) : (
              <div className="placeholder-text">
                <ImageIcon size={48} opacity={0.5} />
                <span>No results yet. Enter a query to begin.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
