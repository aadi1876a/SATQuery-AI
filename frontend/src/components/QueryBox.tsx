import React, { useState } from 'react';
import { Send, Loader } from 'lucide-react';
import './QueryBox.css';

interface QueryBoxProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  disabled: boolean;
  placeholder?: string;
}

const QueryBox: React.FC<QueryBoxProps> = ({ onSubmit, isLoading, disabled, placeholder = "Ask something about the satellite image..." }) => {
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
      <div className="card-header">AI Query</div>
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
          <span className="query-hint">Press Ctrl + Enter to submit</span>
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
