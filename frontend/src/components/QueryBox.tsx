import React, { useState } from 'react';
import { Send, Loader } from 'lucide-react';
import './QueryBox.css';

interface QueryBoxProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  disabled: boolean;
  placeholder?: string;
  isDemoMode?: boolean;
}

const QueryBox: React.FC<QueryBoxProps> = ({ onSubmit, isLoading, disabled, placeholder = "Ask something about the satellite image...", isDemoMode }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = () => {
    if (query.trim() && !isLoading && !disabled) {
      onSubmit(query.trim());
      setQuery('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleSubmit();
    }
  };

  return (
    <div className="query-box-container card">
      <div className="card-header justify-between">
        <span>AI Query</span>
        {isDemoMode && (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.55rem',
            letterSpacing: '0.1em',
            color: 'var(--accent-amber)',
            border: '1px solid rgba(232,168,87,0.35)',
            background: 'rgba(232,168,87,0.08)',
            borderRadius: '2px',
            padding: '0.15rem 0.4rem',
          }}>
            DEMO MODE ACTIVE
          </span>
        )}
      </div>
      <div className="query-input-wrapper">
        <textarea
          className="query-textarea"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || isLoading}
          rows={3}
        />
        <div className="query-footer flex justify-between items-center">
          <span className="query-hint">
            {disabled ? 'Upload an image to enable queries' : 'Ctrl+Enter to submit'}
          </span>
          <button
            className="btn btn-primary btn-submit"
            onClick={handleSubmit}
            disabled={!query.trim() || disabled || isLoading}
          >
            {isLoading ? <Loader size={18} className="spin" /> : <Send size={18} />}
            {isLoading ? 'Processing' : 'Ask SatQuery'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default QueryBox;
