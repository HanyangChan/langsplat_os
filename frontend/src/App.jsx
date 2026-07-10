import React, { useState } from 'react';
import { LayoutDashboard, Compass, Settings as SettingsIcon, Zap, AlertTriangle } from 'lucide-react';
import TrainingMonitor from './components/TrainingMonitor';
import InteractiveDemo from './components/InteractiveDemo';
import FailureGallery from './components/FailureGallery';
import Settings from './components/Settings';

function App() {
  const [activeTab, setActiveTab] = useState('monitor');

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="logo-container">
          <Zap className="logo-icon" size={28} />
          <span>LangSplat OS</span>
        </div>
        
        <nav className="nav-links">
          <div 
            className={`nav-item ${activeTab === 'monitor' ? 'active' : ''}`}
            onClick={() => setActiveTab('monitor')}
          >
            <LayoutDashboard size={20} />
            Training Monitor
          </div>
          <div 
            className={`nav-item ${activeTab === 'demo' ? 'active' : ''}`}
            onClick={() => setActiveTab('demo')}
          >
            <Compass size={20} />
            Interactive Demo
          </div>
          <div 
            className={`nav-item ${activeTab === 'failures' ? 'active' : ''}`}
            onClick={() => setActiveTab('failures')}
          >
            <AlertTriangle size={20} />
            Failure Cases
          </div>
          <div 
            className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <SettingsIcon size={20} />
            Settings
          </div>
        </nav>
        
        <div style={{ marginTop: 'auto', padding: '16px', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          <div style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '8px' }}>System Status</div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>GPU:</span> <span style={{ color: 'var(--success-color)' }}>Online</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>VRAM:</span> <span>12GB / 24GB</span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'monitor' && <TrainingMonitor />}
        {activeTab === 'demo' && <InteractiveDemo />}
        {activeTab === 'failures' && <FailureGallery />}
        {activeTab === 'settings' && <Settings />}
      </main>
    </div>
  );
}

export default App;
