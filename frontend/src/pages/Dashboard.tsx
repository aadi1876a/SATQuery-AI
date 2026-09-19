import React, { useState } from 'react';
import Header from '../components/Header';
import ImageUpload from '../components/ImageUpload';
import ImageViewer from '../components/ImageViewer';
import QueryBox from '../components/QueryBox';
import ResultPanel from '../components/ResultPanel';
import ExecutionTrace from '../components/ExecutionTrace';
import SpatialEvidencePanel from '../components/SpatialEvidencePanel';
import { EarthScene } from '../components/3d/EarthScene';
import { ImageObject, QueryResponse, TaskType } from '../types/api';
import { apiClient } from '../api/client';
import './Dashboard.css';

type AppState = 'landing' | 'analysis';

const capabilities = [
  { label: 'OPTICAL', color: 'amber', angle: 0 },
  { label: 'SAR', color: 'teal', angle: 51 },
  { label: 'VLM', color: 'teal', angle: 102 },
  { label: 'VQA', color: 'amber', angle: 154 },
  { label: 'GROUNDING', color: 'teal', angle: 205 },
  { label: 'CHANGE DET.', color: 'amber', angle: 257 },
  { label: 'FUSION', color: 'teal', angle: 308 },
];

const DEMO_RESPONSE: QueryResponse = {
  status: 'success',
  query: 'demo query',
  answer: 'Agricultural fields identified in the lower-left quadrant. A large water body (reservoir) is visible near coordinates (200, 200). Road network detected running northeast. Vegetation NDVI index suggests moderate crop health.',
  confidence: 0.91,
  visual_evidence: [
    { type: 'bbox', coords: [40,  80,  220, 240], label: 'agricultural field' },
    { type: 'bbox', coords: [250, 200, 380, 320], label: 'reservoir' },
    { type: 'bbox', coords: [10,  10,  500, 40],  label: 'road network' },
  ],
  execution_trace: {
    task_detected: 'vqa',
    input_validation: { is_valid: true },
    tools_selected: ['vlm_vqa', 'grounding_model', 'ndvi_estimator'],
    parameters_used: { resolution: 'high', confidence_threshold: 0.5 },
  },
};

const Dashboard: React.FC = () => {
  const [appState, setAppState] = useState<AppState>('landing');
  const [selectedImage, setSelectedImage] = useState<ImageObject | null>(null);
  const [, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<number | null>(null);
  const [isDemoMode, setIsDemoMode] = useState(false);

  const handleStartAnalysis = () => setAppState('analysis');

  const handleImageSelected = (image: ImageObject | null, file: File | null) => {
    setSelectedImage(image);
    setSelectedFile(file);
    setResult(null);
    setActiveEvidence(null);
  };

  const handleQuerySubmit = async (query: string) => {
    if (!selectedImage) return;
    setIsLoading(true);
    setResult(null);
    setActiveEvidence(null);

    try {
      const response = await apiClient.submitQuery([selectedImage], query, 'vqa' as TaskType);
      setResult(response);
    } catch (error) {
      console.error('API Error:', error);
      if (isDemoMode) {
        await new Promise(r => setTimeout(r, 1800));
        setResult({ ...DEMO_RESPONSE, query });
      } else {
        setResult({
          status: 'error',
          query,
          reason: 'Backend connection failed. Enable Demo Mode to simulate a response.',
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="dashboard-layout">
      {/* ── 3D Earth Background ── */}
      <EarthScene isAnalyzing={appState === 'analysis'} />

      {/* Dark panel overlay during analysis */}
      {appState === 'analysis' && (
        <div
          className="absolute inset-0 z-10 pointer-events-none"
          style={{
            background: 'linear-gradient(to right, rgba(6,10,14,0.97) 0%, rgba(6,10,14,0.55) 50%, rgba(6,10,14,0.97) 100%)',
          }}
        />
      )}

      {/* ── UI Layer ── */}
      <div className="relative z-20 flex flex-col" style={{ height: '100%' }}>
        <Header />

        <main className="dashboard-main">
          {/* ════ LANDING VIEW ════ */}
          {appState === 'landing' && (
            <div className="landing-container animate-fade-in">
              <div className="landing-hero">
                <h1 className="landing-title">SATQUERY AI</h1>
                <p className="landing-subtitle">Multimodal Satellite Intelligence Platform</p>
              </div>

              {/* Floating capability badges */}
              <div className="capability-orbit">
                {capabilities.map((cap, i) => {
                  const rad = (cap.angle * Math.PI) / 180;
                  const rx = 230, ry = 70;
                  const x = Math.cos(rad) * rx;
                  const y = Math.sin(rad) * ry;
                  return (
                    <span
                      key={cap.label}
                      className="cap-badge"
                      style={{
                        transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`,
                        animationDelay: `${i * 0.7}s`,
                        borderColor: cap.color === 'amber'
                          ? 'rgba(232,168,87,0.3)'
                          : 'rgba(55,230,180,0.15)',
                        color: cap.color === 'amber'
                          ? 'var(--accent-amber)'
                          : 'var(--text-muted)',
                      }}
                    >
                      {cap.label}
                    </span>
                  );
                })}
              </div>

              <div className="landing-cta">
                <button
                  id="start-analysis-btn"
                  className="btn-launch"
                  onClick={handleStartAnalysis}
                >
                  <span>▶</span> LAUNCH ANALYSIS
                </button>

                <label
                  className={`demo-toggle ${isDemoMode ? 'active' : ''}`}
                  htmlFor="demo-mode-toggle"
                >
                  <input
                    type="checkbox"
                    id="demo-mode-toggle"
                    checked={isDemoMode}
                    onChange={e => setIsDemoMode(e.target.checked)}
                  />
                  DEMO MODE
                </label>
              </div>
            </div>
          )}

          {/* ════ ANALYSIS VIEW ════ */}
          {appState === 'analysis' && (
            <div className="analysis-workspace animate-fade-in-delayed">
              {/* LEFT column */}
              <div className="analysis-left">
                <ImageUpload
                  onImageSelected={handleImageSelected}
                  selectedImage={selectedImage}
                />

                {/* Image Viewer panel */}
                <div className="panel flex flex-col" style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
                  <div className="panel-header justify-between">
                    <span>IMAGE INTELLIGENCE VIEWER</span>
                    {isLoading && (
                      <span style={{ color: 'var(--accent-teal)', fontSize: '0.6rem', letterSpacing: '0.1em' }}>
                        ◉ ANALYZING…
                      </span>
                    )}
                  </div>
                  <div style={{ flex: 1, position: 'relative', overflow: 'hidden', padding: '0.5rem' }}>
                    <ImageViewer
                      image={selectedImage}
                      evidences={result?.visual_evidence}
                      activeEvidenceIndex={activeEvidence}
                      isLoading={isLoading}
                    />
                  </div>
                </div>

                <QueryBox
                  onSubmit={handleQuerySubmit}
                  isLoading={isLoading}
                  disabled={!selectedImage}
                  placeholder="Ask a question, e.g. 'Identify all agricultural fields' or 'Count roads'"
                />
              </div>

              {/* RIGHT column */}
              <div className="analysis-right">
                <ResultPanel result={result} isLoading={isLoading} />

                {result?.visual_evidence && result.visual_evidence.length > 0 && (
                  <SpatialEvidencePanel
                    evidences={result.visual_evidence}
                    activeIndex={activeEvidence}
                    onSelect={setActiveEvidence}
                  />
                )}

                <div style={{ flex: 1 }}>
                  <ExecutionTrace trace={result?.execution_trace ?? null} isLoading={isLoading} />
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
