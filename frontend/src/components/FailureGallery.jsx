import React, { useState, useEffect } from 'react';
import { AlertTriangle, Info, Image as ImageIcon } from 'lucide-react';

export default function FailureGallery() {
  const [failures, setFailures] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/api/failures')
      .then(res => res.json())
      .then(data => {
        setFailures(data.failures || []);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch failure cases:", err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="monitor-container" style={{ padding: '24px' }}>
      <div className="page-header" style={{ marginBottom: '32px' }}>
        <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <AlertTriangle size={28} color="#f56565" />
          Failure Cases Gallery
        </h1>
        <p className="page-subtitle">Analysis of low relevancy scores and model hallucinations</p>
      </div>

      {loading ? (
        <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '40px' }}>Loading failure data...</div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
          {failures.map((item) => (
            <div key={item.id} className="glass-panel card" style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              
              {/* Image or Placeholder */}
              <div style={{ 
                height: '180px', 
                background: 'linear-gradient(135deg, rgba(20,20,30,1) 0%, rgba(30,20,20,1) 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderBottom: '1px solid var(--border-color)',
                position: 'relative',
                overflow: 'hidden'
              }}>
                {item.image_url ? (
                  <img src={item.image_url} alt={item.query} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <ImageIcon size={48} color="rgba(255,255,255,0.1)" />
                )}
                
                <div style={{ 
                  position: 'absolute', 
                  top: '12px', 
                  right: '12px', 
                  background: 'rgba(245, 101, 101, 0.9)', 
                  color: '#fff', 
                  padding: '4px 8px', 
                  borderRadius: '12px',
                  fontSize: '0.75rem',
                  fontWeight: 'bold',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                }}>
                  Score: {item.score.toFixed(2)}
                </div>
              </div>

              {/* Details */}
              <div style={{ padding: '20px', flexGrow: 1 }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '8px', color: 'var(--text-main)', lineHeight: '1.4' }}>
                  "{item.query}"
                </h3>
                
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'flex-start', 
                  gap: '8px', 
                  background: 'rgba(0,0,0,0.2)', 
                  padding: '12px', 
                  borderRadius: '8px',
                  marginTop: '16px'
                }}>
                  <Info size={16} color="var(--text-muted)" style={{ marginTop: '2px', flexShrink: 0 }} />
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5', margin: 0 }}>
                    {item.issue}
                  </p>
                </div>
              </div>
              
              <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border-color)', fontSize: '0.75rem', color: 'rgba(255,255,255,0.3)', textAlign: 'right' }}>
                {new Date(item.timestamp).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
