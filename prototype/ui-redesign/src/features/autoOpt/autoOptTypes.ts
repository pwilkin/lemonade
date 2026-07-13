export type AutoOptBudget = 'quick' | 'standard' | 'thorough';
export type AutoOptRunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
export type AutoOptStageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
export type AutoOptParallelMode = 'single' | 'parallel';
export type AutoOptKvCacheQuant = 'none' | 'q8_0' | 'q5_1' | 'q4_0';
export type AutoOptRamHeadroom = 'normal' | 'reduced' | 'minimal' | 'disabled';

export interface AutoOptParallelAnswer {
  mode: AutoOptParallelMode;
  slots?: number;
  dedicated?: boolean;
}

export interface AutoOptAnswers {
  parallel: AutoOptParallelAnswer;
  kv_cache_quant: AutoOptKvCacheQuant;
  ram_headroom: AutoOptRamHeadroom;
  use_vision?: boolean;
  allow_network: boolean;
  backends_to_consider?: string[];
}

export interface AutoOptStartRequest {
  model: string;
  budget: AutoOptBudget;
  allow_unload: boolean;
  answers: AutoOptAnswers;
}

/** Flat answer shape consumed by the synthesis engine (C++ WizardAnswers). */
export interface WizardAnswers {
  parallel: boolean;
  slots: number;
  dedicated_slots: boolean;
  kv_cache_quant: AutoOptKvCacheQuant;
  ram_headroom: AutoOptRamHeadroom;
  use_vision?: boolean;
  allow_network: boolean;
  backends_to_consider: string[];
}

export function wizardAnswersFrom(answers: AutoOptAnswers): WizardAnswers {
  const parallel = answers.parallel?.mode === 'parallel';
  return {
    parallel,
    slots: Math.max(1, answers.parallel?.slots ?? (parallel ? 2 : 1)),
    dedicated_slots: answers.parallel?.dedicated ?? true,
    kv_cache_quant: answers.kv_cache_quant || 'none',
    ram_headroom: answers.ram_headroom || 'normal',
    use_vision: answers.use_vision,
    allow_network: answers.allow_network !== false,
    backends_to_consider: answers.backends_to_consider || [],
  };
}

export interface AutoOptProgress {
  stage: string;
  stage_index: number;
  stage_count: number;
  detail?: string;
}

export interface AutoOptStage {
  name: string;
  status: AutoOptStageStatus;
  duration_ms?: number;
  error?: string;
  data?: Record<string, unknown>;
}

export interface HardwareGpu {
  vendor: string;
  name: string;
  family: string;
  vram_gb: number;
}

export interface HardwareSnapshot {
  gpus: HardwareGpu[];
  has_igpu: boolean;
  ram_is_vram: boolean;
  host_ram_gb: number;
  installed_backends: string[];
  os: string;
}

export interface ModelFacts {
  architecture: string;
  block_count: number;
  expert_count: number;
  full_attention_interval: number;
  swa_layer_count: number;
  n_ctx_train: number;
  kv_bytes_per_token: number;
  is_moe: boolean;
  is_hybrid_or_recurrent: boolean;
  has_mtp: boolean;
  has_vision: boolean;
  base_model_repo: string;
  checkpoint: string;
}

export interface FitDevice {
  device: string;
  model_mib: number;
  ctx_mib: number;
  compute_mib: number;
}

export interface FitEstimate {
  backend: string;
  fit_target_mib: number;
  extra_args: string;
  fitted_args: string;
  fitted_ctx: number;
  fitted_ngl: number;
  fitted_ncmoe: number;
  devices: FitDevice[];
  fits_fully: boolean;
  ok: boolean;
  error: string;
}

export interface BenchParams {
  d?: number;
  b?: number;
  ub?: number;
  ctk?: string;
  ctv?: string;
  spec_n?: number;
  ladder?: boolean;
}

export interface BenchPoint {
  backend: string;
  params: BenchParams;
  pp_avg_ts: number;
  tg_avg_ts: number;
  n_depth: number;
  ok: boolean;
  error?: string;
}

export interface SamplingDefaults {
  temperature?: number;
  top_p?: number;
  min_p?: number;
  top_k?: number;
  source: string;
}

export interface AutoOptExpected {
  pp_ts?: number;
  tg_ts?: number;
  vram_mib?: number;
}

export interface AutoOptRecommendation {
  label: string;
  tradeoff?: string;
  llamacpp_backend: string;
  ctx_size: number;
  mmproj_enabled: boolean;
  llamacpp_args: string;
  rationale: string[];
  expected?: AutoOptExpected;
}

export interface AutoOptResult {
  primary: AutoOptRecommendation;
  alternatives: AutoOptRecommendation[];
  sampling_defaults?: SamplingDefaults;
}

export interface AutoOptMeasurements {
  fit: FitEstimate[];
  bench: BenchPoint[];
}

export interface AutoOptRunRecord {
  id: string;
  model: string;
  checkpoint: string;
  budget: AutoOptBudget;
  answers: AutoOptAnswers;
  allow_unload: boolean;
  status: AutoOptRunStatus;
  created_at: string;
  finished_at?: string;
  summary?: string;
  error?: string;
  lemonade_version?: string;
  progress?: AutoOptProgress;
  stages: AutoOptStage[];
  measurements: AutoOptMeasurements;
  result?: AutoOptResult;
}

export function isAutoOptRunActive(run: Pick<AutoOptRunRecord, 'status'>): boolean {
  return run.status === 'queued' || run.status === 'running';
}
