export type Modality = 'optical' | 'sar';

export type TaskType =
  | 'vqa'
  | 'captioning'
  | 'grounding'
  | 'change_vqa'
  | 'fusion_analysis';

export interface ImageObject {
  image_id: string;
  file_path: string;
  modality: Modality;
  format: string;
  bands: number;
  resolution_m: number;
  crs?: string;
  bbox?: [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]
  acquisition_date?: string;
  width: number;
  height: number;
  thumbnail_path?: string;
}

export type SpatialEvidenceType = 'bbox' | 'mask' | 'none';

export interface SpatialEvidence {
  type: SpatialEvidenceType;
  coords?: number[]; // [x1, y1, x2, y2]
  mask_path?: string;
  label?: string;
}

export interface ToolInput {
  task: TaskType;
  query?: string;
  images: ImageObject[];
  params?: Record<string, any>;
}

export interface ToolOutput {
  status: 'success' | 'error' | 'rejected';
  text_answer?: string;
  spatial_evidence?: SpatialEvidence[];
  confidence?: number;
  raw_output_path?: string;
  model_used?: string;
  error_message?: string;
}

export interface InputValidation {
  is_valid: boolean;
  message?: string;
}

export interface ExecutionTrace {
  task_detected: TaskType;
  input_validation: InputValidation;
  tools_selected: string[];
  parameters_used: Record<string, any>;
}

export interface QueryResponse {
  status: 'success' | 'error' | 'rejected';
  query: string;
  answer?: string;
  visual_evidence?: SpatialEvidence[];
  confidence?: number;
  execution_trace?: ExecutionTrace;
  report_download_url?: string;
  reason?: string;
}
