export interface SearchRequest {
  query: string;
  parameters?: {
    candidates_k: number;    // 50-500
    alpha: number;          // 0-1 (step 0.1)
    top_k: number;          // 5-50
    similarity_threshold: number; // 0.1-1 (step 0.05)
  };
}

export interface SearchHit {
  service: string;
  description: string;
  classification: string;
  sector: string;
}

export interface SearchResponse {
  hits_kept: SearchHit[];
  total_time: number;
  query: string;
  parameters_used: {
    candidates_k: number;
    alpha: number;
    top_k: number;
    similarity_threshold: number;
  };
}