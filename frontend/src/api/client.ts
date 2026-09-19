import { ImageObject, QueryResponse, TaskType } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiClient {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json() as T;
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);
      throw error;
    }
  }

  async checkHealth(): Promise<{ status: string }> {
    try {
      return await this.request<{ status: string }>('/health');
    } catch {
      return { status: 'error' };
    }
  }

  async uploadImage(file: File): Promise<ImageObject> {
    const formData = new FormData();
    formData.append('file', file);
    
    // We don't set Content-Type header here, fetch does it automatically with boundary for FormData
    const url = `${API_BASE_URL}/upload`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload Error: ${response.status} ${response.statusText}`);
    }

    return await response.json() as ImageObject;
  }

  async submitQuery(
    images: ImageObject[], 
    query: string, 
    task: TaskType | 'auto'
  ): Promise<QueryResponse> {
    // If auto is selected, the backend should detect the task
    const endpoint = '/query';
    const body = {
      images: images.map(img => img.image_id), // Send IDs to backend
      query,
      task: task === 'auto' ? undefined : task
    };

    return this.request<QueryResponse>(endpoint, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async downloadReport(reportUrl: string): Promise<void> {
    const url = reportUrl.startsWith('http') ? reportUrl : `${API_BASE_URL}${reportUrl}`;
    window.open(url, '_blank');
  }
}

export const apiClient = new ApiClient();
