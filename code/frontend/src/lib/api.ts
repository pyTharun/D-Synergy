// ============================================================
// API CLIENT
// Centralized fetch functions for the FastAPI backend
// ============================================================

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${endpoint}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

// ============================================================
// TYPES
// ============================================================

export interface Dataset {
  key: string;
  display_name: string;
  csv_available: boolean;
  model_trained: boolean;
  status: 'trained' | 'available' | 'missing';
}

export interface DatasetStats {
  dataset: string;
  display_name: string;
  total_rows: number;
  valid_rows: number;
  missing_rows: number;
  duplicate_rows: number;
  unique_drug_pairs: number;
  unique_cell_lines: number;
  valid_drug1_smiles: number;
  valid_drug2_smiles: number;
  score_stats: {
    min: number | null;
    max: number | null;
    mean: number | null;
    median: number | null;
    std: number | null;
  };
  score_distribution: {
    counts: number[];
    bin_edges: number[];
  };
  model_trained: boolean;
  model_info: ModelMetrics;
}

export interface DatasetSample {
  drugname1: string;
  drugname2: string;
  drugname1_full: string;
  drugname2_full: string;
  cellline: string;
  score: number;
}

export interface PredictRequest {
  dataset: string;
  drug1: string;
  drug2: string;
  cellline: string;
}

export interface PredictResponse {
  predicted_score: number;
  dataset: string;
  display_name: string;
  model_name: string;
  morgan_radius: number;
  fingerprint_size: number;
  cellline: string;
  interpretation: string;
}

export interface ModelMetrics {
  dataset: string;
  display_name: string;
  model_name: string;
  model_type: string;
  morgan_radius: number;
  fingerprint_size: number;
  training_samples: number;
  testing_samples: number;
  trained_at: string;
  metrics: {
    MAE: number;
    MSE: number;
    RMSE: number;
    R2: number;
  };
  dataset_stats: Record<string, unknown>;
}

export interface TrainRequest {
  dataset: string;
  model_type: string;
  test_size: number;
  morgan_radius: number;
  fp_size: number;
  random_state: number;
}

export interface TrainJobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  message: string;
  result: {
    metrics?: { MAE: number; RMSE: number; R2: number; MSE: number };
    training_samples?: number;
    testing_samples?: number;
    model_name?: string;
    trained_at?: string;
  };
}

// ============================================================
// API FUNCTIONS
// ============================================================

export const api = {
  // Health
  health: () => apiFetch<{ status: string; loaded_models: string[] }>('/health'),

  // Datasets
  listDatasets: () =>
    apiFetch<{ datasets: Dataset[] }>('/datasets').then((r) => r.datasets),

  datasetStats: (dataset: string) =>
    apiFetch<DatasetStats>(`/dataset-stats?dataset=${dataset}`),

  datasetSamples: (dataset: string, page = 1, limit = 50, search = '') =>
    apiFetch<{
      dataset: string;
      total: number;
      page: number;
      limit: number;
      total_pages: number;
      rows: DatasetSample[];
    }>(`/dataset-samples?dataset=${dataset}&page=${page}&limit=${limit}&search=${encodeURIComponent(search)}`),

  cellLines: (dataset: string) =>
    apiFetch<{ dataset: string; cell_lines: string[]; count: number }>(`/cell-lines?dataset=${dataset}`),

  // Prediction
  predict: (req: PredictRequest) =>
    apiFetch<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  // Models
  listModels: () =>
    apiFetch<{ models: ModelMetrics[] }>('/models').then((r) => r.models),

  modelMetrics: (dataset: string) =>
    apiFetch<ModelMetrics>(`/model-metrics?dataset=${dataset}`),

  modelActuals: (dataset: string) =>
    apiFetch<{ dataset: string; actual: number[]; predicted: number[]; count: number }>(
      `/model-actuals?dataset=${dataset}`
    ),

  modelErrors: (dataset: string) =>
    apiFetch<{ dataset: string; bin_edges: number[]; counts: number[] }>(
      `/model-errors?dataset=${dataset}`
    ),

  // Training
  supportedModels: () =>
    apiFetch<{ models: { key: string; name: string }[] }>('/supported-models').then(
      (r) => r.models
    ),

  startTraining: (req: TrainRequest) =>
    apiFetch<{ job_id: string; status: string; message: string }>('/train', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  trainingStatus: (jobId: string) =>
    apiFetch<TrainJobStatus>(`/train/status/${jobId}`),
};
