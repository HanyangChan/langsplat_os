import React, { useState } from 'react';
import { Save, HardDrive, Cpu, Sliders, Palette, RefreshCw } from 'lucide-react';

export default function Settings() {
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Mock settings state
  const [settings, setSettings] = useState({
    device: 'gpu',
    modelCheckpoint: 'langsplat-base-001.pt',
    datasetPath: '/data/lerf_ovs',
    maxPoints: 500000,
    defaultThreshold: 0.5,
    theme: 'dark'
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  const handleSave = () => {
    setSaving(true);
    // Simulate API call
    setTimeout(() => {
      setSaving(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    }, 800);
  };

  return (
    <div className="monitor-container" style={{ padding: '24px', maxWidth: '900px', margin: '0 auto' }}>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-subtitle">Configure system preferences and model parameters</p>
        </div>
        <button 
          onClick={handleSave} 
          className="btn" 
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            background: saved ? 'var(--success-color)' : 'var(--accent-color)', 
            color: '#fff', 
            border: 'none', 
            padding: '10px 20px', 
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.3s'
          }}
          disabled={saving}
        >
          {saving ? <RefreshCw size={18} className="spin" /> : <Save size={18} />}
          {saving ? 'Saving...' : saved ? 'Saved!' : 'Save Changes'}
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        
        {/* Hardware & Computation */}
        <section className="glass-panel card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
            <Cpu size={24} color="var(--accent-color)" />
            <h2 style={{ fontSize: '1.25rem', color: 'var(--text-main)', margin: 0 }}>Compute & Model</h2>
          </div>
          
          <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div className="form-group" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <label style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Inference Device</label>
              <select 
                name="device" 
                value={settings.device} 
                onChange={handleChange}
                style={{ padding: '10px', borderRadius: '6px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: 'var(--text-main)' }}
              >
                <option value="gpu">GPU (CUDA)</option>
                <option value="mps">Mac (MPS)</option>
                <option value="cpu">CPU</option>
              </select>
            </div>
            
            <div className="form-group" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <label style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Active Checkpoint</label>
              <select 
                name="modelCheckpoint" 
                value={settings.modelCheckpoint} 
                onChange={handleChange}
                style={{ padding: '10px', borderRadius: '6px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: 'var(--text-main)' }}
              >
                <option value="langsplat-base-001.pt">langsplat-base-001.pt</option>
                <option value="langsplat-ft-002.pt">langsplat-ft-002.pt (Fine-tuned)</option>
                <option value="clip-resnet50.pt">clip-resnet50.pt</option>
              </select>
            </div>
          </div>
        </section>

        {/* Paths & Storage */}
        <section className="glass-panel card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
            <HardDrive size={24} color="#f6ad55" />
            <h2 style={{ fontSize: '1.25rem', color: 'var(--text-main)', margin: 0 }}>Data & Storage</h2>
          </div>
          
          <div className="form-group" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <label style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Dataset Root Directory</label>
            <input 
              type="text" 
              name="datasetPath" 
              value={settings.datasetPath} 
              onChange={handleChange}
              style={{ padding: '10px', borderRadius: '6px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: 'var(--text-main)', width: '100%' }}
            />
            <p style={{ margin: 0, fontSize: '0.8rem', color: 'rgba(255,255,255,0.3)' }}>Absolute path to the pre-processed LERF/OVS datasets.</p>
          </div>
        </section>

        {/* Rendering Limits */}
        <section className="glass-panel card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
            <Sliders size={24} color="#68d391" />
            <h2 style={{ fontSize: '1.25rem', color: 'var(--text-main)', margin: 0 }}>Rendering Limits</h2>
          </div>
          
          <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div className="form-group" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <label style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Max Rendered Points</label>
              <input 
                type="number" 
                name="maxPoints" 
                value={settings.maxPoints} 
                onChange={handleChange}
                style={{ padding: '10px', borderRadius: '6px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: 'var(--text-main)' }}
              />
            </div>
            
            <div className="form-group" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <label style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Default Relevancy Threshold</label>
              <input 
                type="number" 
                step="0.1"
                min="0"
                max="1"
                name="defaultThreshold" 
                value={settings.defaultThreshold} 
                onChange={handleChange}
                style={{ padding: '10px', borderRadius: '6px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: 'var(--text-main)' }}
              />
            </div>
          </div>
        </section>
        
        {/* Appearance */}
        <section className="glass-panel card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
            <Palette size={24} color="#b794f4" />
            <h2 style={{ fontSize: '1.25rem', color: 'var(--text-main)', margin: 0 }}>Appearance</h2>
          </div>
          
          <div className="form-group" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <label style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Interface Theme</label>
            <div style={{ display: 'flex', gap: '16px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-main)' }}>
                <input type="radio" name="theme" value="dark" checked={settings.theme === 'dark'} onChange={handleChange} />
                Dark Mode
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-main)' }}>
                <input type="radio" name="theme" value="light" checked={settings.theme === 'light'} onChange={handleChange} />
                Light Mode
              </label>
            </div>
          </div>
        </section>

      </div>
      
      {/* Required style for the spinner */}
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
      `}} />
    </div>
  );
}
