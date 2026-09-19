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
import { Zap, ArrowRight, ChevronRight, MapPin, BarChart2, Eye, Layers } from 'lucide-react';
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

// Real satellite / earth imagery from public sources
const SATELLITE_CARDS = [
  {
    img: 'https://images.unsplash.com/photo-1614730321146-b6fa6a46bcb4?w=500&q=80',
    label: 'Optical Analysis',
    desc: 'Sentinel-2 multispectral imagery',
    badge: 'OPTICAL',
    badgeColor: 'amber',
  },
  {
    img: 'https://images.unsplash.com/photo-1446776653964-20c1d3a81b06?w=500&q=80',
    label: 'SAR Imaging',
    desc: 'Sentinel-1 C-band synthetic aperture radar',
    badge: 'SAR',
    badgeColor: 'teal',
  },
  {
    img: 'https://images.unsplash.com/photo-1541873676-a18131494184?w=500&q=80',
    label: 'Change Detection',
    desc: 'Multi-temporal land use analysis',
    badge: 'CHANGE',
    badgeColor: 'teal',
  },
];

const FEATURE_ITEMS = [
  {
    icon: Eye,
    title: 'Visual Question Answering',
    desc: 'Ask natural language questions about satellite imagery and get precise AI-powered answers.',
    details: 'The VQA model allows you to interact with satellite imagery as if you were talking to an expert geospatial analyst. You can ask for object counts, land use classification, or specific details like "Are there any vessels in this harbor?". It uses a multimodal vision-language model trained specifically on optical and SAR satellite data.',
  },
  {
    icon: MapPin,
    title: 'Object Grounding',
    desc: 'Detect and localize objects — roads, buildings, crops — with spatial bounding boxes.',
    details: 'Object grounding translates text prompts into precise spatial coordinates. By typing "find all storage tanks", the system will return bounding boxes around every detected tank in the image. This is essential for inventory tracking, urban planning, and rapid infrastructure monitoring.',
  },
  {
    icon: BarChart2,
    title: 'Change Detection',
    desc: 'Compare multi-temporal images to detect land use changes, urban sprawl, and flooding.',
    details: 'Upload two images of the same area from different dates to automatically highlight changes. The AI identifies new construction, deforestation, or flooded areas, making it a critical tool for disaster response, environmental monitoring, and tracking urban expansion.',
  },
  {
    icon: Layers,
    title: 'SAR + Optical Fusion',
    desc: 'Combine radar and optical data for robust all-weather intelligence through cloud cover.',
    details: 'Optical sensors are frequently blocked by clouds, but Synthetic Aperture Radar (SAR) sees right through them. Our fusion engine aligns Sentinel-1 SAR and Sentinel-2 optical data to provide continuous, all-weather monitoring. This ensures you never lose visibility of your critical areas of interest.',
  },
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
  const [selectedFeature, setSelectedFeature] = useState<typeof FEATURE_ITEMS[0] | null>(null);

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
        <Header
          isDemoMode={isDemoMode}
          onDemoModeChange={setIsDemoMode}
          onBackToLanding={appState === 'analysis' ? () => setAppState('landing') : undefined}
        />

        <main className="dashboard-main">
          {/* ════ LANDING VIEW ════ */}
          {appState === 'landing' && (
            <div className="landing-container animate-fade-in">

              {/* Hero section */}
              <div className="landing-hero">
                <div className="landing-eyebrow">
                  <span className="eyebrow-dot" />
                  Smart India Hackathon 2024
                </div>
                <h1 className="landing-title">SATQUERY AI</h1>
                <p className="landing-subtitle">Multimodal Satellite Intelligence Platform</p>
                <p className="landing-desc">
                  Ask questions about satellite imagery in plain English. Powered by Vision-Language Models,
                  SAR processing, and multi-modal AI fusion.
                </p>
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

              {/* CTA */}
              <div className="landing-cta">
                <button
                  id="start-analysis-btn"
                  className="btn-launch"
                  onClick={handleStartAnalysis}
                >
                  <Zap size={18} />
                  LAUNCH ANALYSIS
                  <ArrowRight size={16} />
                </button>

                <label
                  className={`demo-toggle ${isDemoMode ? 'active' : ''}`}
                  htmlFor="demo-mode-toggle-landing"
                >
                  <input
                    type="checkbox"
                    id="demo-mode-toggle-landing"
                    checked={isDemoMode}
                    onChange={e => setIsDemoMode(e.target.checked)}
                  />
                  DEMO MODE
                </label>
              </div>

              {/* Satellite imagery cards */}
              <div className="sat-cards-row">
                {SATELLITE_CARDS.map((card, i) => (
                  <div
                    key={card.label}
                    className="sat-card animate-fade-in"
                    style={{ animationDelay: `${0.3 + i * 0.15}s`, animationFillMode: 'both' }}
                  >
                    <div className="sat-card-img-wrap">
                      <img
                        src={card.img}
                        alt={card.label}
                        className="sat-card-img"
                        loading="lazy"
                      />
                      <div className="sat-card-overlay" />
                      <span className={`sat-card-badge badge-${card.badgeColor}`}>
                        {card.badge}
                      </span>
                    </div>
                    <div className="sat-card-body">
                      <div className="sat-card-title">{card.label}</div>
                      <div className="sat-card-desc">{card.desc}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Feature grid */}
              <div className="feature-grid">
                {FEATURE_ITEMS.map((feat, i) => (
                  <div
                    key={feat.title}
                    className="feature-item animate-fade-in"
                    style={{ animationDelay: `${0.5 + i * 0.1}s`, animationFillMode: 'both' }}
                    onClick={() => setSelectedFeature(feat)}
                  >
                    <div className="feature-icon-wrap">
                      <feat.icon size={18} />
                    </div>
                    <div className="feature-content">
                      <div className="feature-title">{feat.title}</div>
                      <div className="feature-desc">{feat.desc}</div>
                    </div>
                    <ChevronRight size={14} className="feature-arrow" />
                  </div>
                ))}
              </div>

              {/* Feature Modal */}
              {selectedFeature && (
                <div className="feature-modal-overlay animate-fade-in" onClick={() => setSelectedFeature(null)}>
                  <div className="feature-modal card" onClick={e => e.stopPropagation()}>
                    <div className="card-header justify-between">
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <selectedFeature.icon size={16} /> FEATURE DETAILS
                      </span>
                      <button className="btn-icon" onClick={() => setSelectedFeature(null)}>
                        ✕
                      </button>
                    </div>
                    <div className="feature-modal-body">
                      <h2 className="feature-modal-title">{selectedFeature.title}</h2>
                      <p className="feature-modal-desc">{selectedFeature.details}</p>
                      
                      <button 
                        className="btn btn-primary w-full"
                        style={{ marginTop: '1.5rem' }}
                        onClick={() => {
                          setSelectedFeature(null);
                          handleStartAnalysis();
                        }}
                      >
                        TRY THIS FEATURE
                      </button>
                    </div>
                  </div>
                </div>
              )}

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
                  isDemoMode={isDemoMode}
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
