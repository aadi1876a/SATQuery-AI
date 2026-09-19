import React from 'react';
import { QueryResponse } from '../types/api';
import { CheckCircle2, XCircle, AlertTriangle, Download, BrainCircuit, Activity } from 'lucide-react';
import { apiClient } from '../api/client';
import './ResultPanel.css';

interface ResultPanelProps {
  result: QueryResponse | null;
  isLoading?: boolean;
}

const ResultPanel: React.FC<ResultPanelProps> = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <div className="panel flex flex-col items-center justify-center gap-4" style={{ minHeight: 200 }}>
        <Activity size={28} style={{ color: 'var(--accent-teal)', animation: 'pulse 1.5s ease-in-out infinite' }} />
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', letterSpacing: '0.12em', color: 'var(--accent-teal)', textTransform: 'uppercase' }}>
          PROCESSING TELEMETRY...
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="panel flex flex-col items-center justify-center" style={{ minHeight: 200 }}>
        <BrainCircuit size={22} style={{ color: 'var(--text-dim)', opacity: 0.3, marginBottom: '0.5rem' }} />
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', letterSpacing: '0.12em', color: 'var(--text-dim)', textTransform: 'uppercase' }}>
          SYSTEM STANDBY
        </p>
      </div>
    );
  }

  const statusColors: Record<string, string> = {
    success:  'var(--accent-teal)',
    rejected: 'var(--accent-amber)',
    error:    'var(--accent-error)',
  };
  const statusColor = statusColors[result.status] ?? 'var(--text-muted)';

  const StatusIcon = result.status === 'success'
    ? CheckCircle2
    : result.status === 'rejected'
    ? AlertTriangle
    : XCircle;

  return (
    <div className={`panel result-panel status-${result.status}`}>
      {/* Header */}
      <div className="panel-header justify-between">
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <BrainCircuit size={14} /> AI INTELLIGENCE
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', color: statusColor }}>
          <StatusIcon size={13} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', letterSpacing: '0.1em' }}>
            {result.status.toUpperCase()}
          </span>
        </span>
      </div>

      <div style={{ padding: '0.875rem', display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>

        {/* Answer block */}
        {result.status === 'success' && result.answer && (
          <div className="animate-fade-in">
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', letterSpacing: '0.12em', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
              ANALYSIS RESULT
            </p>
            <div className="result-answer-block">
              {result.answer}
            </div>
          </div>
        )}

        {/* Rejected */}
        {result.status === 'rejected' && result.reason && (
          <div className="animate-fade-in" style={{ padding: '0.75rem', border: '1px solid rgba(232,168,87,0.3)', borderRadius: 'var(--radius-sm)', background: 'rgba(232,168,87,0.05)' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color: 'var(--accent-amber)', letterSpacing: '0.1em', marginBottom: '0.375rem' }}>REQUEST REJECTED</p>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-main)' }}>{result.reason}</p>
          </div>
        )}

        {/* Error */}
        {result.status === 'error' && result.reason && (
          <div className="animate-fade-in" style={{ padding: '0.75rem', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 'var(--radius-sm)', background: 'rgba(239,68,68,0.05)' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color: 'var(--accent-error)', letterSpacing: '0.1em', marginBottom: '0.375rem' }}>SYSTEM ERROR</p>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-main)' }}>{result.reason}</p>
          </div>
        )}

        {/* Confidence + Evidence stats */}
        {result.status === 'success' && (result.confidence !== undefined || result.visual_evidence?.length) && (
          <div className="animate-fade-in-delayed flex gap-3">
            {/* Circular confidence meter */}
            {result.confidence !== undefined && (() => {
              const r = 28;
              const circ = 2 * Math.PI * r;
              const dash = circ * result.confidence;
              return (
                <div style={{ flex: 1, background: 'rgba(6,10,14,0.5)', border: '1px solid var(--border-panel)', borderRadius: 'var(--radius-sm)', padding: '0.75rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.375rem' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', letterSpacing: '0.1em', color: 'var(--text-dim)', alignSelf: 'flex-start', textTransform: 'uppercase' }}>CONFIDENCE</span>
                  <div style={{ position: 'relative', width: 64, height: 64 }}>
                    <svg viewBox="0 0 64 64" className="confidence-svg" style={{ width: 64, height: 64, transform: 'rotate(-90deg)' }}>
                      <circle cx="32" cy="32" r={r} strokeWidth="4" fill="transparent" stroke="rgba(255,255,255,0.06)" />
                      <circle
                        cx="32" cy="32" r={r}
                        strokeWidth="4" fill="transparent"
                        stroke="var(--accent-teal)"
                        strokeDasharray={circ}
                        strokeDashoffset={circ - dash}
                        strokeLinecap="round"
                        className="confidence-track"
                      />
                    </svg>
                    <span style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-teal)' }}>
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              );
            })()}

            {/* Evidence count */}
            {result.visual_evidence && result.visual_evidence.length > 0 && (
              <div style={{ flex: 1, background: 'rgba(6,10,14,0.5)', border: '1px solid var(--border-panel)', borderRadius: 'var(--radius-sm)', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', letterSpacing: '0.1em', color: 'var(--text-dim)', textTransform: 'uppercase' }}>DETECTIONS</span>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="evidence-count-badge">{result.visual_evidence.length}</span>
                  <span className="evidence-count-label">OBJECTS<br/>FOUND</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Download report */}
        {result.report_download_url && (
          <button
            className="btn btn-outline w-full"
            onClick={() => apiClient.downloadReport(result.report_download_url!)}
          >
            <Download size={13} /> EXPORT MISSION REPORT
          </button>
        )}
      </div>
    </div>
  );
};

export default ResultPanel;
