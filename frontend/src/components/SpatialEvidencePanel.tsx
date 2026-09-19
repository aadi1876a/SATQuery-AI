import React from 'react';
import { SpatialEvidence } from '../types/api';
import { Target, Crosshair } from 'lucide-react';
import './SpatialEvidencePanel.css';

interface SpatialEvidencePanelProps {
  evidences: SpatialEvidence[];
  activeIndex: number | null;
  onSelect: (index: number) => void;
}

const SpatialEvidencePanel: React.FC<SpatialEvidencePanelProps> = ({ evidences, activeIndex, onSelect }) => {
  if (!evidences || evidences.length === 0) return null;

  return (
    <div className="panel" style={{ flexShrink: 0 }}>
      <div className="panel-header justify-between">
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Target size={13} /> SPATIAL EVIDENCE
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', letterSpacing: '0.1em', color: 'var(--text-dim)' }}>
          {evidences.length} OBJECTS
        </span>
      </div>

      <div style={{ padding: '0.625rem', display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
        {evidences.map((ev, idx) => {
          const isActive = activeIndex === idx;
          return (
            <div
              key={idx}
              className={`evidence-item ${isActive ? 'active' : ''} animate-fade-in`}
              onClick={() => onSelect(idx)}
              style={{ animationDelay: `${idx * 0.08}s`, animationFillMode: 'both' }}
            >
              <div className="evidence-inner">
                <div className="evidence-row">
                  <span className="evidence-label">
                    {ev.label ?? `OBJECT ${idx + 1}`}
                  </span>
                  <span className="evidence-type-badge">{ev.type.toUpperCase()}</span>
                </div>
                {ev.coords && (
                  <div className="evidence-coords">
                    <Crosshair size={10} />
                    <span>[{ev.coords.map(c => Math.round(c)).join(', ')}]</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default SpatialEvidencePanel;
