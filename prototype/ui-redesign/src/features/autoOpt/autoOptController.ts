import api from '../../api';
import {
  checkpointRepoId,
  pickBenchWinner,
  roundDownCtx,
  synthesize,
  validationFlagSubset,
} from './autoOptSynthesize';
import {
  AutoOptResult,
  AutoOptStageStatus,
  AutoOptStartRequest,
  BenchPoint,
  FitEstimate,
  HardwareSnapshot,
  ModelFacts,
  SamplingDefaults,
  wizardAnswersFrom,
} from './autoOptTypes';

export const AUTOOPT_STAGES = [
  'snapshot', 'model_facts', 'hf_metadata', 'fit_probes', 'bench_matrix',
  'load_validation', 'synthesize',
] as const;

export const UNSUPPORTED_SERVER_MESSAGE =
  'This server does not support the llama.cpp tool endpoints — update lemond.';

export class AutoOptUnsupportedError extends Error {
  constructor() {
    super(UNSUPPORTED_SERVER_MESSAGE);
    this.name = 'AutoOptUnsupportedError';
  }
}

export interface ControllerCallbacks {
  stage(name: string, status: AutoOptStageStatus, patch?: { duration_ms?: number; error?: string; data?: Record<string, unknown> }): void;
  progress(detail: string): void;
  fit(estimate: FitEstimate): void;
  bench(points: BenchPoint[]): void;
}

export interface ControllerOutcome {
  result: AutoOptResult;
  summary: string;
}

const RECURRENT_ARCHS = new Set([
  'mamba', 'mamba2', 'rwkv6', 'rwkv6qwen2', 'rwkv7', 'arwkv7',
  'jamba', 'falcon-h1', 'granitehybrid', 'nemotron-h', 'lfm2', 'plamo2',
]);

function isAbort(err: unknown): boolean {
  return (err as { name?: string } | null)?.name === 'AbortError';
}

function isNotFound(err: unknown): boolean {
  return (err as { status?: number } | null)?.status === 404;
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err || 'Unknown error');
}

function parseGb(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const match = /([\d.]+)/.exec(value);
    const n = match ? Number(match[1]) : NaN;
    if (Number.isFinite(n)) return value.toLowerCase().includes('mb') ? n / 1024 : n;
  }
  return 0;
}

function snapshotFromSystemInfo(info: Record<string, unknown>): HardwareSnapshot {
  const hw: HardwareSnapshot = {
    gpus: [],
    has_igpu: false,
    ram_is_vram: false,
    host_ram_gb: parseGb(info['Physical Memory']),
    installed_backends: [],
    os: String(info['OS Version'] || ''),
  };
  const devices = (info.devices || {}) as Record<string, unknown>;
  // system-info carries no explicit iGPU flag; an APU reports a small dedicated
  // carve-out (vram_gb) next to a large shared GTT budget (virtual_mem_gb).
  let sharedMemoryGpu = false;
  const readGpus = (list: unknown, vendor: string) => {
    if (!Array.isArray(list)) return;
    for (const raw of list) {
      const g = raw as Record<string, unknown>;
      if (g?.available !== true) continue;
      hw.gpus.push({
        vendor,
        name: String(g.name || ''),
        family: String(g.family || ''),
        vram_gb: Number(g.vram_gb) || 0,
      });
      const virtualGb = Number(g.virtual_mem_gb);
      if (Number.isFinite(virtualGb) && virtualGb > (Number(g.vram_gb) || 0)) sharedMemoryGpu = true;
    }
  };
  readGpus(devices.amd_gpu, 'amd');
  readGpus(devices.nvidia_gpu, 'nvidia');
  hw.has_igpu = sharedMemoryGpu;
  hw.ram_is_vram = sharedMemoryGpu && hw.gpus.length === 1;

  const recipes = (info.recipes || {}) as Record<string, { backends?: Record<string, { state?: unknown }> }>;
  const backends = recipes.llamacpp?.backends || {};
  for (const [name, backendInfo] of Object.entries(backends)) {
    const state = String(backendInfo?.state || '');
    if (state === 'installed' || state === 'update_available') hw.installed_backends.push(name);
  }
  return hw;
}

function factsFromModelDetail(detail: Record<string, unknown>): ModelFacts {
  const metadata = (detail.metadata || null) as Record<string, unknown> | null;
  if (!metadata) throw new Error('model detail carries no GGUF metadata');
  const labels = Array.isArray(detail.labels) ? detail.labels.map(l => String(l).toLowerCase()) : [];
  const architecture = String(metadata.architecture || '');
  const fullAttentionInterval = Number(metadata.full_attention_interval) || 0;
  const expertCount = Number(metadata.expert_count) || 0;
  return {
    architecture,
    block_count: Number(metadata.block_count) || 0,
    expert_count: expertCount,
    full_attention_interval: fullAttentionInterval,
    swa_layer_count: Number(metadata.swa_layer_count) || 0,
    n_ctx_train: Number(metadata.context_length) || 0,
    kv_bytes_per_token: Number(metadata.kv_bytes_per_token) || 0,
    is_moe: expertCount > 1,
    is_hybrid_or_recurrent: fullAttentionInterval > 0 || RECURRENT_ARCHS.has(architecture),
    has_mtp: labels.includes('mtp'),
    has_vision: labels.includes('vision'),
    base_model_repo: String(metadata.base_model_repo || ''),
    checkpoint: String(detail.checkpoint || ''),
  };
}

function hfRepoIdFromUrl(urlOrId: string): string {
  const marker = 'huggingface.co/';
  const pos = urlOrId.indexOf(marker);
  let id = pos === -1 ? urlOrId : urlOrId.slice(pos + marker.length);
  while (id.endsWith('/')) id = id.slice(0, -1);
  return id;
}

async function fetchHfJson(url: string, signal: AbortSignal): Promise<Record<string, unknown> | null> {
  const resp = await fetch(url, { signal });
  if (!resp.ok) return null;
  try {
    return await resp.json() as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function resolveBaseModelRepo(
  ggufBaseModelRepo: string,
  checkpoint: string,
  signal: AbortSignal,
): Promise<string> {
  if (ggufBaseModelRepo) return hfRepoIdFromUrl(ggufBaseModelRepo);
  const repo = checkpointRepoId(checkpoint);
  if (!repo) return '';
  const card = await fetchHfJson(`https://huggingface.co/api/models/${repo}`, signal);
  const baseModel = (card?.cardData as Record<string, unknown> | undefined)?.base_model;
  if (typeof baseModel === 'string') return baseModel;
  if (Array.isArray(baseModel) && typeof baseModel[0] === 'string') return baseModel[0];
  return '';
}

function samplingFromGenerationConfig(gc: Record<string, unknown>, base: string): SamplingDefaults {
  const sd: SamplingDefaults = { source: `hf:${base}/generation_config.json` };
  if (typeof gc.temperature === 'number') sd.temperature = gc.temperature;
  if (typeof gc.top_p === 'number') sd.top_p = gc.top_p;
  if (typeof gc.min_p === 'number') sd.min_p = gc.min_p;
  if (typeof gc.top_k === 'number') sd.top_k = gc.top_k;
  return sd;
}

function mtpProbePrompt(): string {
  return 'The quick brown fox jumps over the lazy dog. '.repeat(120);
}

export async function executeAutoOptRun(
  request: AutoOptStartRequest,
  signal: AbortSignal,
  cb: ControllerCallbacks,
): Promise<ControllerOutcome> {
  const answers = wizardAnswersFrom(request.answers);
  const budget = request.budget;
  const withBench = budget !== 'quick';

  const throwIfAborted = () => {
    if (signal.aborted) throw Object.assign(new Error('cancelled by user'), { name: 'AbortError' });
  };

  const runStage = async (name: string, body: () => Promise<void>): Promise<boolean> => {
    throwIfAborted();
    cb.stage(name, 'running');
    const t0 = performance.now();
    let error = '';
    let ok = true;
    try {
      await body();
    } catch (err) {
      if (isAbort(err)) {
        cb.stage(name, 'failed', { duration_ms: Math.round(performance.now() - t0) });
        throw err;
      }
      if (isNotFound(err)) {
        cb.stage(name, 'failed', { duration_ms: Math.round(performance.now() - t0), error: UNSUPPORTED_SERVER_MESSAGE });
        throw new AutoOptUnsupportedError();
      }
      ok = false;
      error = errorMessage(err);
    }
    cb.stage(name, ok ? 'completed' : 'failed', {
      duration_ms: Math.round(performance.now() - t0),
      ...(error ? { error } : {}),
    });
    return ok;
  };

  const fits: FitEstimate[] = [];
  const bench: BenchPoint[] = [];
  let hw: HardwareSnapshot | null = null;
  let facts: ModelFacts | null = null;
  let sampling: SamplingDefaults | undefined;

  // ── snapshot ─────────────────────────────────────────────────────────
  if (!await runStage('snapshot', async () => {
    hw = snapshotFromSystemInfo(await api.systemInfo());
  })) {
    throw new Error('could not inspect system hardware');
  }

  // ── model_facts ──────────────────────────────────────────────────────
  if (!await runStage('model_facts', async () => {
    const detail = await api.modelDetail(request.model) as unknown as Record<string, unknown>;
    if (String(detail.recipe || '') !== 'llamacpp') throw new Error('model recipe is not llamacpp');
    if (detail.downloaded !== true) throw new Error('model is not downloaded');
    facts = factsFromModelDetail(detail);
  })) {
    throw new Error('could not read model metadata');
  }
  const hardware = hw!;
  const modelFacts = facts!;

  // ── hf_metadata (direct Hugging Face fetch; soft-fail) ───────────────
  if (answers.allow_network) {
    await runStage('hf_metadata', async () => {
      const base = await resolveBaseModelRepo(modelFacts.base_model_repo, modelFacts.checkpoint, signal);
      if (!base) throw new Error('base model not resolvable');
      const gc = await fetchHfJson(`https://huggingface.co/${base}/resolve/main/generation_config.json`, signal);
      if (!gc) throw new Error(`no generation_config.json on ${base}`);
      sampling = samplingFromGenerationConfig(gc, base);
      cb.stage('hf_metadata', 'running', { data: { base_model: base, generation_config_found: true } });
    });
  } else {
    cb.stage('hf_metadata', 'skipped');
  }
  throwIfAborted();

  // For benchmark tiers the probes own the GPU: unload everything (consent
  // enforced by the wizard) and never reload — apply/load does that.
  if (withBench && request.allow_unload) {
    await api.unloadModel();
  }

  // ── fit_probes ───────────────────────────────────────────────────────
  const candidates: string[] = [];
  for (const b of hardware.installed_backends) {
    if (b === 'cpu' || b === 'system' || b === 'metal') continue;
    if (answers.backends_to_consider.length > 0 && !answers.backends_to_consider.includes(b)) continue;
    candidates.push(b);
  }
  if (candidates.length === 0 && hardware.installed_backends.length > 0) {
    candidates.push(hardware.installed_backends[0]);
  }

  await runStage('fit_probes', async () => {
    for (const backend of candidates) {
      cb.progress(`llama-fit-params on ${backend}`);
      let fit: FitEstimate;
      try {
        fit = await api.llamacppFitParams({ model: request.model, backend, fit_target_mib: 1024 }, signal);
      } catch (err) {
        if (isAbort(err) || isNotFound(err)) throw err;
        fit = {
          backend, fit_target_mib: 1024, extra_args: '', fitted_args: '', fitted_ctx: 0,
          fitted_ngl: -1, fitted_ncmoe: 0, devices: [], fits_fully: false, ok: false,
          error: errorMessage(err),
        };
      }
      cb.fit(fit);
      fits.push(fit);
      if (modelFacts.has_vision && answers.use_vision === false) {
        try {
          const noMmproj = await api.llamacppFitParams(
            { model: request.model, backend, args: '--no-mmproj', fit_target_mib: 1024 }, signal);
          cb.fit(noMmproj);
          fits.push(noMmproj);
        } catch (err) {
          if (isAbort(err) || isNotFound(err)) throw err;
        }
      }
      throwIfAborted();
    }
    if (!fits.some(f => f.ok)) throw new Error('all fit probes failed');
  });
  // Fit failures degrade to heuristics rather than failing the run.
  throwIfAborted();

  // ── bench_matrix ─────────────────────────────────────────────────────
  if (withBench) {
    await runStage('bench_matrix', async () => {
      const primaryFit = fits.find(f => f.ok && f.extra_args === '') || null;
      const depths: number[] = [0];
      if (primaryFit && primaryFit.fitted_ctx >= 32768 && budget === 'thorough') {
        depths.push(30000);
      } else if (primaryFit && primaryFit.fitted_ctx >= 8192 && budget === 'thorough') {
        depths.push(Math.floor(0.8 * primaryFit.fitted_ctx));
      }

      const runBench = async (backend: string, body: { d: number | number[]; b?: number; ub?: number }, ladder: boolean) => {
        const points = await api.llamacppBench(
          { model: request.model, backend, ...body },
          { signal, onProgress: detail => cb.progress(detail) },
        );
        const tagged = ladder
          ? points.map(p => ({ ...p, params: { ...(p.params || {}), ladder: true } }))
          : points;
        cb.bench(tagged);
        bench.push(...tagged);
      };

      // Backend duel (single-candidate boxes get a plain baseline).
      if (candidates.length > 1) {
        for (const backend of candidates) {
          cb.progress(`llama-bench duel on ${backend}`);
          await runBench(backend, { d: depths }, false);
          throwIfAborted();
        }
      } else if (candidates.length === 1) {
        cb.progress(`llama-bench baseline on ${candidates[0]}`);
        await runBench(candidates[0], { d: depths }, false);
      }
      throwIfAborted();

      // Batch ladder and MTP sweep run on the provisional duel winner.
      let bb = candidates[0] || 'vulkan';
      if (candidates.length > 1) {
        const winner = pickBenchWinner(bench, candidates);
        if (winner) bb = winner;
      }
      if (hardware.ram_is_vram && hardware.host_ram_gb >= 32) {
        const rungs = budget === 'thorough' ? [512, 1024, 2048, 4096, 8192] : [512, 2048, 8192];
        for (const r of rungs) {
          cb.progress(`batch ladder -b ${r}`);
          await runBench(bb, { d: 0, b: r, ub: r }, true);
          throwIfAborted();
        }
      }

      // MTP draft-length sweep (llama-bench has no spec flags): real loads
      // with explicit args, decode speed from llama-server's timings.
      if (modelFacts.has_mtp) {
        const ns = budget === 'thorough' ? [1, 2, 3, 4, 5, 6] : [2, 3, 4];
        for (const n of ns) {
          cb.progress(`MTP draft sweep n=${n}`);
          try {
            await api.loadModel(request.model, {
              llamacpp_args: `--spec-type draft-mtp --spec-draft-n-max ${n} --spec-draft-p-min 0.75`,
              save_options: false,
            });
            let tokS: number | undefined;
            for (let attempt = 0; attempt < 2; attempt++) {
              const resp = await api.chatCompletionRaw(
                request.model,
                [{ role: 'user', content: mtpProbePrompt() }],
                { max_tokens: 128, temperature: 0 },
                signal,
              );
              const timings = resp?.timings as Record<string, unknown> | undefined;
              const measured = Number(timings?.predicted_per_second);
              if (Number.isFinite(measured) && measured > 0) tokS = measured;
            }
            if (tokS !== undefined) {
              const point: BenchPoint = {
                backend: bb, params: { spec_n: n }, pp_avg_ts: 0, tg_avg_ts: tokS, n_depth: 0, ok: true,
              };
              cb.bench([point]);
              bench.push(point);
            }
          } catch (err) {
            if (isAbort(err)) throw err;
          }
          throwIfAborted();
        }
        await api.unloadModel(request.model).catch(() => {});
      }
    });
  } else {
    cb.stage('bench_matrix', 'skipped');
  }
  throwIfAborted();

  // ── synthesize (+ load_validation ctx adjustment, lever 12) ──────────
  let result: AutoOptResult | null = null;
  if (!await runStage('synthesize', async () => {
    result = synthesize(hardware, modelFacts, answers, fits, bench, sampling);
  })) {
    throw new Error('preset synthesis failed');
  }
  const synthesized = result!;

  if (withBench) {
    await runStage('load_validation', async () => {
      const tokens = validationFlagSubset(synthesized.primary.ctx_size, synthesized.primary.llamacpp_args);
      cb.progress('validating recommended flags with llama-fit-params');
      const v = await api.llamacppFitParams({
        model: request.model,
        backend: synthesized.primary.llamacpp_backend,
        args: tokens,
        fit_target_mib: 1024,
      }, signal);
      if (v.ok && v.fitted_ctx > 0 && v.fitted_ctx < synthesized.primary.ctx_size) {
        synthesized.primary.ctx_size = roundDownCtx(v.fitted_ctx);
        synthesized.primary.rationale.push(
          `Context reduced to ${synthesized.primary.ctx_size} after full-flag validation.`);
      }
      cb.fit(v);
    });
  } else {
    cb.stage('load_validation', 'skipped');
  }

  const summary = `${synthesized.primary.llamacpp_backend} · ctx ${synthesized.primary.ctx_size}`
    + (synthesized.primary.llamacpp_args ? ` · ${synthesized.primary.llamacpp_args}` : '');
  return { result: synthesized, summary };
}

/**
 * Distinguishes "endpoints missing" (404) from every other reply. A server
 * with the tool endpoints answers the empty probe body with 400.
 */
export async function probeToolEndpoints(): Promise<boolean> {
  try {
    await api.llamacppFitParams({ model: '', backend: '' });
    return true;
  } catch (err) {
    return !isNotFound(err);
  }
}
