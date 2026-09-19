import React, { useRef, useState } from 'react';
import { UploadCloud, FileImage, X } from 'lucide-react';
import { ImageObject } from '../types/api';
import './ImageUpload.css';

interface ImageUploadProps {
  onImageSelected: (image: ImageObject | null, file: File | null) => void;
  selectedImage: ImageObject | null;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelected, selectedImage }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processFile = (file: File) => {
    // In a real scenario, we might immediately upload to backend and get ImageObject back.
    // Since we don't have a working backend right now, we create a mock ImageObject for the UI.
    const mockImage: ImageObject = {
      image_id: `img_${Date.now()}`,
      file_path: file.name,
      modality: 'optical',
      format: file.type.split('/')[1] || 'tiff',
      bands: 3,
      resolution_m: 10,
      crs: 'EPSG:32644',
      width: 1024,
      height: 1024,
      acquisition_date: new Date().toISOString().split('T')[0],
      // We can use object url as a temporary display path
      thumbnail_path: URL.createObjectURL(file) 
    };
    
    onImageSelected(mockImage, file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onImageSelected(null, null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  if (selectedImage) {
    return (
      <div className="card image-metadata-card">
        <div className="card-header justify-between">
          <span>Active Image</span>
          <button className="btn-icon" onClick={handleClear}><X size={16} /></button>
        </div>
        <div className="metadata-content flex items-center gap-4">
          <div className="metadata-thumb">
            <FileImage size={24} className="text-muted" />
          </div>
          <div className="metadata-details">
            <div className="metadata-title">{selectedImage.file_path}</div>
            <div className="metadata-grid">
              <div><span className="label">Modality:</span> {selectedImage.modality.toUpperCase()}</div>
              <div><span className="label">Format:</span> {selectedImage.format.toUpperCase()}</div>
              <div><span className="label">Resolution:</span> {selectedImage.resolution_m}m</div>
              <div><span className="label">Bands:</span> {selectedImage.bands}</div>
              <div><span className="label">Size:</span> {selectedImage.width}x{selectedImage.height}</div>
              <div><span className="label">CRS:</span> {selectedImage.crs || 'N/A'}</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`upload-zone ${isDragging ? 'dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleChange}
        accept="image/*,.tiff,.tif"
        className="hidden"
      />
      <UploadCloud size={48} className="upload-icon" />
      <h3>Drop satellite imagery here</h3>
      <p>Supports TIFF, PNG, JPEG (Optical & SAR)</p>
      <button className="btn btn-primary">Browse Files</button>
    </div>
  );
};

export default ImageUpload;
