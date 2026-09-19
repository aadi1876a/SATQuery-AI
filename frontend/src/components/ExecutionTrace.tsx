import React from 'react';
import { ExecutionTrace as TraceType } from '../types/api';
import { GitBranch, ShieldCheck, Search, Settings2, BrainCircuit, Activity } from 'lucide-react';
import './ExecutionTrace.css';

interface ExecutionTraceProps {
  trace: TraceType | null;
  isLoading?: boolean;
}

const ExecutionTrace: React.FC<ExecutionTraceProps> = ({ trace, isLoading }) => {
  if (isLoading) {
    return (
      <div className="panel flex flex-col items-center justify-center gap-4" style={{ minHeight: 250 }}>
        <Activity size={28} style={{ color: 'var(--accent-teal)', animation: 'pulse 1.5s ease-in-out infinite' }} />
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', letterSpacing: '0.12em', color: 'var(--accent-teal)', textTransform: 'uppercase' }}>
          ESTABLISHING PIPELINE...
        </p>
      </div>
    );
  }

  if (!trace) {
    return (
      <div className="panel flex flex-col items-center justify-center gap-2" style={{ minHeight: 250 }}>
        <GitBranch size={22} style={{ color: 'var(--text-dim)', opacity: 0.4 }} />
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', letterSpacing: '0.12em', color: 'var(--text-dim)', textTransform: 'uppercase', textAlign: 'center' }}>
          NO TRACE DATA<br />
          <span style={{ opacity: 0.5, fontSize: '0.55rem' }}>AWAITING EXECUTION</span>
        </p>
      </div>
    );
  }

  const steps = [
    {
      title: 'INPUT VALIDATION',
      icon: ShieldCheck,
      content: trace.input_validation.is_valid
        ? 'All checks passed'
        : `Rejected: ${trace.input_validation.message ?? 'unknown'}`,
      status: trace.input_validation.is_valid ? 'success' : 'error',
    },
    {
      title: 'TASK CLASSIFIER',
      icon: Search,
      content: `Detected: ${trace.task_detected.toUpperCase()}`,
      status: 'success',
    },
    {
      title: 'MODEL ROUTER',
      icon: Settings2,
      content: trace.tools_selected.join(' · '),
      status: 'success',
    },
    {
      title: 'EXECUTION',
      icon: BrainCircuit,
      content: 'Inference complete',
      status: 'success',
    },
  ];

  return (
    <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header justify-between">
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <GitBranch size={14} /> EXECUTION TRACE
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color: 'var(--accent-teal)', letterSpacing: '0.1em' }}>
          {steps.length} STEPS
        </span>
      </div>

      <div style={{ flex: 1, padding: '1rem', overflowY: 'auto' }}>
        <div className="trace-pipeline">
          {steps.map((step, idx) => (
            <div
              key={idx}
              className="trace-step animate-fade-in"
              style={{ animationDelay: `${idx * 0.15}s`, animationFillMode: 'both' }}
            >
              <div className={`trace-node ${step.status}`}>
                <step.icon size={13} />
              </div>
              <div className="trace-content">
                <div className="trace-title">{step.title}</div>
                <div className="trace-body">{step.content}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ExecutionTrace;
