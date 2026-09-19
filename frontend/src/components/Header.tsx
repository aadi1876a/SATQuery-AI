import React, { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { Satellite, Radio, Clock, Globe, ArrowLeft, FlaskConical } from 'lucide-react';
import './Header.css';

interface HeaderProps {
  isDemoMode?: boolean;
  onDemoModeChange?: (value: boolean) => void;
  onBackToLanding?: () => void;
}

const Header: React.FC<HeaderProps> = ({ isDemoMode, onDemoModeChange, onBackToLanding }) => {
  const [status, setStatus] = useState<'connected' | 'disconnected' | 'processing'>('processing');
  const [time, setTime] = useState('');
  const [date, setDate] = useState('');

  // UTC clock
  useEffect(() => {
    const tick = () => {
      const now = new Date();
      const utcH = String(now.getUTCHours()).padStart(2, '0');
      const utcM = String(now.getUTCMinutes()).padStart(2, '0');
      const utcS = String(now.getUTCSeconds()).padStart(2, '0');
      setTime(`${utcH}:${utcM}:${utcS}`);
      const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
      setDate(`${now.getUTCDate()} ${months[now.getUTCMonth()]} ${now.getUTCFullYear()}`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // Backend health check
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await apiClient.checkHealth();
        setStatus(res.status === 'ok' ? 'connected' : 'disconnected');
      } catch {
        setStatus('disconnected');
      }
    };
    checkStatus();
    const id = setInterval(checkStatus, 30000);
    return () => clearInterval(id);
  }, []);

  const statusLabel = status === 'connected'
    ? 'ONLINE'
    : status === 'disconnected'
    ? 'OFFLINE'
    : 'LINKING...';

  return (
    <header className="hud-header">
      {/* Left: branding */}
      <div className="hud-brand">
        {onBackToLanding && (
          <button className="hud-back-btn" onClick={onBackToLanding} title="Back to Home">
            <ArrowLeft size={16} />
          </button>
        )}
        <div className="hud-logo-wrap">
          <Satellite size={22} className="hud-logo-icon" />
          <div className="hud-logo-ring" />
        </div>
        <div className="hud-title-block">
          <span className="hud-title">SATQUERY AI</span>
          <span className="hud-subtitle">Multimodal Satellite Intelligence · SIH 2024</span>
        </div>
      </div>

      {/* Center: system status pills */}
      <div className="hud-center">
        <div className="hud-pill hud-pill--teal">
          <span className="hud-pill-dot" />
          <span>OPTICAL</span>
        </div>
        <div className="hud-divider" />
        <div className="hud-pill hud-pill--amber">
          <span className="hud-pill-dot hud-pill-dot--amber" />
          <span>SAR</span>
        </div>
        <div className="hud-divider" />
        <div className="hud-pill hud-pill--teal">
          <span className="hud-pill-dot" />
          <span>VLM</span>
        </div>
        <div className="hud-divider" />
        <div className="hud-pill hud-pill--teal">
          <span className="hud-pill-dot" />
          <span>GROUNDING</span>
        </div>
      </div>

      {/* Right: system info */}
      <div className="hud-right">
        {/* UTC clock */}
        <div className="hud-clock">
          <Clock size={12} className="hud-clock-icon" />
          <div>
            <span className="hud-clock-time">{time}</span>
            <span className="hud-clock-label">UTC · {date}</span>
          </div>
        </div>

        {/* Backend status */}
        <div className={`hud-status hud-status--${status}`}>
          <Radio size={13} />
          <div className="hud-status-info">
            <span className="hud-status-label">{statusLabel}</span>
            <span className="hud-status-sub">BACKEND</span>
          </div>
          <span className={`status-dot ${status}`} />
        </div>

        {/* Demo mode toggle (in analysis mode) */}
        {onDemoModeChange && (
          <label
            className={`hud-demo-toggle ${isDemoMode ? 'active' : ''}`}
            htmlFor="demo-mode-toggle-header"
            title="Enable demo mode to simulate AI responses"
          >
            <FlaskConical size={11} />
            <input
              type="checkbox"
              id="demo-mode-toggle-header"
              checked={isDemoMode}
              onChange={e => onDemoModeChange(e.target.checked)}
            />
            DEMO
          </label>
        )}

        {/* Coordinates badge */}
        <div className="hud-coords">
          <Globe size={11} />
          <span>20.5937°N · 78.9629°E</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
