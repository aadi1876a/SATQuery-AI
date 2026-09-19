import React, { useEffect, useRef, useState } from 'react';
import { ImageObject, SpatialEvidence } from '../types/api';
import { ZoomIn, ZoomOut, Maximize } from 'lucide-react';
import './ImageViewer.css';

interface ImageViewerProps {
  image: ImageObject | null;
  evidences?: SpatialEvidence[];
  activeEvidenceIndex?: number | null;
  isLoading?: boolean;
}

const ImageViewer: React.FC<ImageViewerProps> = ({ image, evidences = [], activeEvidenceIndex = null, isLoading = false }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [imgElement, setImgElement] = useState<HTMLImageElement | null>(null);

  // Load image
  useEffect(() => {
    if (!image?.thumbnail_path) {
      setImgElement(null);
      return;
    }
    const img = new Image();
    img.src = image.thumbnail_path;
    img.onload = () => {
      setImgElement(img);
      fitToScreen(img);
    };
  }, [image]);

  const fitToScreen = (img: HTMLImageElement) => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const scaleX = container.clientWidth / img.width;
    const scaleY = container.clientHeight / img.height;
    const initialScale = Math.min(scaleX, scaleY) * 0.95; 
    
    setScale(initialScale);
    
    const scaledWidth = img.width * initialScale;
    const scaledHeight = img.height * initialScale;
    
    setOffset({
      x: (container.clientWidth - scaledWidth) / 2,
      y: (container.clientHeight - scaledHeight) / 2
    });
  };

  const handleFit = () => {
    if (imgElement) fitToScreen(imgElement);
  };

  const handleZoom = (direction: 'in' | 'out') => {
    setScale(prev => {
      const newScale = direction === 'in' ? prev * 1.2 : prev / 1.2;
      return Math.max(0.1, Math.min(newScale, 10)); 
    });
  };

  // Mouse drag for pan
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  // Draw loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !imgElement) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(scale, scale);
    
    // Draw base image
    ctx.drawImage(imgElement, 0, 0);
    
    // Dim image slightly when loading
    if (isLoading) {
      ctx.fillStyle = 'rgba(0, 5, 15, 0.5)';
      ctx.fillRect(0, 0, imgElement.width, imgElement.height);
    }
    
    // Draw Evidences
    evidences.forEach((ev, index) => {
      if (ev.type === 'bbox' && ev.coords && ev.coords.length === 4) {
        const [x1, y1, x2, y2] = ev.coords;
        const width = x2 - x1;
        const height = y2 - y1;
        
        const isActive = activeEvidenceIndex === index;
        
        ctx.beginPath();
        ctx.rect(x1, y1, width, height);
        ctx.lineWidth = isActive ? 3 / scale : 2 / scale;
        ctx.strokeStyle = isActive ? '#00d4ff' : '#10b981';
        
        // Add glow effect for active
        if (isActive) {
          ctx.shadowColor = '#00d4ff';
          ctx.shadowBlur = 15 / scale;
        } else {
          ctx.shadowBlur = 0;
        }
        ctx.stroke();
        
        ctx.shadowBlur = 0; // reset
        
        if (isActive) {
          ctx.fillStyle = 'rgba(0, 212, 255, 0.15)';
          ctx.fill();
        }

        if (ev.label) {
          ctx.font = `bold ${14 / scale}px 'Inter', sans-serif`;
          const textWidth = ctx.measureText(ev.label).width;
          
          ctx.fillStyle = isActive ? '#00d4ff' : '#10b981';
          ctx.fillRect(x1, y1 - (22 / scale), textWidth + (12 / scale), 22 / scale);
          
          ctx.fillStyle = '#000';
          ctx.fillText(ev.label.toUpperCase(), x1 + (6 / scale), y1 - (6 / scale));
        }
      }
    });

    ctx.restore();
  }, [imgElement, offset, scale, evidences, activeEvidenceIndex, isLoading]);

  if (!image) {
    return (
      <div className="viewer-empty">
        <p>NO SOURCE DATA ACQUIRED</p>
        <p style={{ fontSize: '0.6rem', color: 'var(--text-dim)', letterSpacing: '0.05em', marginTop: '0.25rem' }}>Upload a satellite image to begin analysis</p>
      </div>
    );
  }

  return (
    <div className="viewer-container" ref={containerRef}>
      
      {/* AI scan animation */}
      {isLoading && (
        <div className="viewer-scan-overlay">
          <div className="viewer-scan-line" />
        </div>
      )}

      {/* Zoom/Fit toolbar */}
      <div className="viewer-toolbar">
        <button className="btn-icon" onClick={() => handleZoom('in')} title="Zoom In"><ZoomIn size={14} /></button>
        <button className="btn-icon" onClick={() => handleZoom('out')} title="Zoom Out"><ZoomOut size={14} /></button>
        <button className="btn-icon" onClick={handleFit} title="Fit to Screen"><Maximize size={14} /></button>
      </div>
      
      <canvas
        ref={canvasRef}
        className="viewer-canvas"
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
    </div>
  );
};

export default ImageViewer;
