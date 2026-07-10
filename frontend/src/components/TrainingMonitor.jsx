import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Zap, Cpu, Layers, Timer } from 'lucide-react';

export default function TrainingMonitor() {
  const [data, setData] = useState([]);
  const [status, setStatus] = useState('Initializing...');

  useEffect(() => {
    // Fetch mock training metrics from the backend
    fetch('http://localhost:8000/api/training/metrics')
      .then(res => res.json())
      .then(result => {
        setData(result.metrics);
        setStatus(result.status);
      })
      .catch(err => {
        console.error("Failed to fetch metrics:", err);
        setStatus("Offline");
      });
  }, []);

  const currentLoss = data.length > 0 ? data[data.length - 1].loss.toFixed(4) : "0.0000";
  const currentPsnr = data.length > 0 ? data[data.length - 1].psnr.toFixed(2) : "0.00";
  const currentGpu = data.length > 0 ? data[data.length - 1].gpu_usage.toFixed(1) : "0.0";
  const currentGaussian = data.length > 0 ? data[data.length - 1].gaussian_count.toLocaleString() : "0";
  const currentSpeed = data.length > 0 ? data[data.length - 1].iterations_per_sec.toFixed(1) : "0.0";

  return (
    <div className="monitor-container">
      <div className="page-header">
        <h1 className="page-title">Training Overview</h1>
        <p className="page-subtitle">Real-time metrics for LangSplat optimization</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '24px', marginBottom: '24px' }}>
        <div className="stat-box glass-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={20} color="var(--accent-color)" />
            <div className="stat-label">Current Loss</div>
          </div>
          <div className="stat-value">{currentLoss}</div>
          <div className="stat-label" style={{ marginTop: '8px', color: 'var(--success-color)' }}>↓ Improving</div>
        </div>

        <div className="stat-box glass-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Zap size={20} color="var(--accent-color)" />
            <div className="stat-label">PSNR</div>
          </div>
          <div className="stat-value">{currentPsnr} dB</div>
          <div className="stat-label" style={{ marginTop: '8px', color: 'var(--success-color)' }}>↑ Improving</div>
        </div>

        <div className="stat-box glass-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={20} color="var(--accent-color)" />
            <div className="stat-label">GPU Usage</div>
          </div>
          <div className="stat-value">{currentGpu}%</div>
          <div className="stat-label" style={{ marginTop: '8px', color: '#a0aec0' }}>Stable</div>
        </div>

        <div className="stat-box glass-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={20} color="var(--accent-color)" />
            <div className="stat-label">Gaussian Count</div>
          </div>
          <div className="stat-value">{currentGaussian}</div>
          <div className="stat-label" style={{ marginTop: '8px', color: '#a0aec0' }}>Growing</div>
        </div>

        <div className="stat-box glass-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Timer size={20} color="var(--accent-color)" />
            <div className="stat-label">Speed</div>
          </div>
          <div className="stat-value">{currentSpeed} it/s</div>
          <div className="stat-label" style={{ marginTop: '8px', color: '#a0aec0' }}>Stable</div>
        </div>
      </div>

      <div className="glass-panel card" style={{ height: '400px', marginBottom: '24px' }}>
        <div className="card-title">Training Loss Over Time</div>
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="step" stroke="#a0aec0" />
              <YAxis stroke="#a0aec0" />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--border-color)', borderRadius: '8px' }} 
                itemStyle={{ color: 'var(--accent-color)' }}
              />
              <Line type="monotone" dataKey="loss" stroke="var(--accent-color)" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="placeholder-text" style={{ height: '100%', justifyContent: 'center' }}>
            {status === "Offline" ? "Backend Offline" : "Loading metrics..."}
          </div>
        )}
      </div>
      
      <div className="glass-panel card" style={{ height: '400px' }}>
        <div className="card-title">PSNR Over Time</div>
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="step" stroke="#a0aec0" />
              <YAxis stroke="#a0aec0" domain={['auto', 'auto']} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--border-color)', borderRadius: '8px' }} 
                itemStyle={{ color: 'var(--success-color)' }}
              />
              <Line type="monotone" dataKey="psnr" stroke="var(--success-color)" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="placeholder-text" style={{ height: '100%', justifyContent: 'center' }}>
            {status === "Offline" ? "Backend Offline" : "Loading metrics..."}
          </div>
        )}
      </div>
    </div>
  );
}
