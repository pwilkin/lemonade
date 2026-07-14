import api from '../../api';
import {
  checkpointRepoId,
  computeFitEstimate,
  pickBenchWinner,
  roundDownCtx,
  roundUpCtx,
  synthesize,
} from './autoOptSynthesize';
import {
  AutoOptRecommendation,
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

// Execution order == array order (review #6a): synthesize precedes the
// real-load test-by-failure that finalizes the recommendation.
export const AUTOOPT_STAGES = [
  'snapshot', 'model_facts', 'hf_metadata', 'fit_estimate', 'bench_matrix',
  'synthesize', 'load_test',
] as const;

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

const MEASURE_RUNS = 2;              // measured runs per config; report the mean
const MAX_LOAD_BACKOFFS = 3;
const DEPTH_SENTENCE = 'The quick brown fox jumps over the lazy dog. ';
const DEPTH_TOKENS_PER_SENTENCE = 11;

function isAbort(err: unknown): boolean {
  return (err as { name?: string } | null)?.name === 'AbortError';
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err || 'Unknown error');
}

function abortError(): Error {
  return Object.assign(new Error('cancelled by user'), { name: 'AbortError' });
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
      const virtualGb = Number(g.virtual_mem_gb);
      const vramGb = Number(g.vram_gb) || 0;
      const shared = Number.isFinite(virtualGb) && virtualGb > vramGb;
      hw.gpus.push({
        vendor,
        name: String(g.name || ''),
        family: String(g.family || ''),
        vram_gb: shared ? virtualGb : vramGb,
      });
      if (shared) sharedMemoryGpu = true;
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

/** Available VRAM budget in MiB. Unified-memory boxes draw from system RAM. */
function availableMib(hw: HardwareSnapshot): number {
  if (hw.ram_is_vram) return Math.max(1024, hw.host_ram_gb * 0.9) * 1024;
  const gpuGb = hw.gpus.reduce((sum, g) => sum + (g.vram_gb || 0), 0);
  if (gpuGb > 0) return gpuGb * 0.92 * 1024;
  return Math.max(1024, hw.host_ram_gb * 0.7) * 1024;
}

function factsFromModelDetail(detail: Record<string, unknown>): ModelFacts {
  const metadata = (detail.metadata && typeof detail.metadata === 'object' && !Array.isArray(detail.metadata))
    ? detail.metadata as Record<string, unknown>
    : null;
  const labels = Array.isArray(detail.labels) ? detail.labels.map(l => String(l).toLowerCase()) : [];
  const architecture = String(metadata?.architecture || '');
  const fullAttentionInterval = Number(metadata?.full_attention_interval) || 0;
  const expertCount = Number(metadata?.expert_count) || 0;
  // GGUF file size in GiB (models/{id} `size`); fall back to metadata if present.
  const sizeGb = Number(detail.size) || Number(metadata?.file_size_gb) || 0;
  return {
    architecture,
    block_count: Number(metadata?.block_count) || 0,
    expert_count: expertCount,
    full_attention_interval: fullAttentionInterval,
    swa_layer_count: Number(metadata?.swa_layer_count) || 0,
    n_ctx_train: Number(metadata?.context_length) || Number(detail.max_context_window) || 0,
    kv_bytes_per_token: Number(metadata?.kv_bytes_per_token) || 0,
    weights_mib: sizeGb * 1024,
    is_moe: expertCount > 1,
    is_hybrid_or_recurrent: fullAttentionInterval > 0 || RECURRENT_ARCHS.has(architecture),
    has_mtp: labels.includes('mtp'),
    base_model_repo: String(metadata?.base_model_repo || ''),
    checkpoint: String(detail.checkpoint || ''),
    metadata_present: !!metadata,
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

async function resolveBaseModelRepo(ggufBaseModelRepo: string, checkpoint: string, signal: AbortSignal): Promise<string> {
  if (ggufBaseModelRepo) return hfRepoIdFromUrl(ggufBaseModelRepo);
  const repo = checkpointRepoId(checkpoint);
  if (!repo) return '';
  const card = await fetchHfJson(`https://huggingface.co/api/models/${repo}`, signal);
  const baseModel = (card?.cardData as Record<string, unknown> | undefined)?.base_model;
  if (typeof baseModel === 'string') return baseModel;
  if (Array.isArray(baseModel) && typeof baseModel[0] === 'string') return baseModel[0];
  return '';
}

function samplingFromConfig(gc: Record<string, unknown>, base: string, file: string): SamplingDefaults {
  const sd: SamplingDefaults = { source: `hf:${base}/${file}` };
  if (typeof gc.temperature === 'number') sd.temperature = gc.temperature;
  if (typeof gc.top_p === 'number') sd.top_p = gc.top_p;
  if (typeof gc.min_p === 'number') sd.min_p = gc.min_p;
  if (typeof gc.top_k === 'number') sd.top_k = gc.top_k;
  return sd;
}

function hasSamplingValues(sd: SamplingDefaults): boolean {
  return sd.temperature !== undefined || sd.top_p !== undefined
    || sd.min_p !== undefined || sd.top_k !== undefined;
}

async function fetchSamplingDefaults(base: string, signal: AbortSignal): Promise<SamplingDefaults | null> {
  const gc = await fetchHfJson(`https://huggingface.co/${base}/resolve/main/generation_config.json`, signal);
  if (gc) {
    const sd = samplingFromConfig(gc, base, 'generation_config.json');
    if (hasSamplingValues(sd)) return sd;
  }
  const cfg = await fetchHfJson(`https://huggingface.co/${base}/resolve/main/config.json`, signal);
  if (cfg) {
    const nested = cfg.generation_config;
    if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
      const sd = samplingFromConfig(nested as Record<string, unknown>, base, 'config.json');
      if (hasSamplingValues(sd)) return sd;
    }
    const top = samplingFromConfig(cfg, base, 'config.json');
    if (hasSamplingValues(top)) return top;
  }
  return null;
}

function depthPrompt(depthTokens: number): string {
  if (depthTokens <= 0) return 'Reply with a one-word greeting.';
  const repeats = Math.max(1, Math.ceil(depthTokens / DEPTH_TOKENS_PER_SENTENCE));
  return DEPTH_SENTENCE.repeat(repeats) + '\nReply with a one-word summary of the text above.';
}

interface Metrics {
  ttftMs: number;
  tps: number;
  tokens: number;
}

function extractMetrics(resp: Record<string, unknown>): Metrics {
  const m: Metrics = { ttftMs: 0, tps: 0, tokens: 0 };
  const usage = resp?.usage as Record<string, unknown> | undefined;
  const timings = resp?.timings as Record<string, unknown> | undefined;
  if (usage) {
    const ttft = Number(usage.prefill_duration_ttft);
    const tps = Number(usage.decoding_speed_tps);
    if (Number.isFinite(ttft) && ttft > 0) m.ttftMs = ttft * 1000;
    if (Number.isFinite(tps) && tps > 0) m.tps = tps;
    m.tokens += Number(usage.prompt_tokens) || 0;
    m.tokens += Number(usage.completion_tokens) || 0;
  }
  if (m.ttftMs <= 0 && timings) {
    const ttft = Number(timings.prompt_ms);
    if (Number.isFinite(ttft) && ttft > 0) m.ttftMs = ttft;
  }
  if (m.tps <= 0 && timings) {
    const tps = Number(timings.predicted_per_second);
    if (Number.isFinite(tps) && tps > 0) m.tps = tps;
  }
  if (timings) {
    m.tokens += Number(timings.prompt_n) || 0;
    m.tokens += Number(timings.predicted_n) || 0;
  }
  return m;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);

export async function executeAutoOptRun(
  request: AutoOptStartRequest,
  signal: AbortSignal,
  cb: ControllerCallbacks,
): Promise<ControllerOutcome> {
  const answers = wizardAnswersFrom(request.answers);
  const budget = request.budget;
  const withBench = budget !== 'quick';

  const throwIfAborted = () => { if (signal.aborted) throw abortError(); };

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
    if (!facts.metadata_present) {
      cb.stage('model_facts', 'running', { data: { note: 'model metadata unavailable — using coarse fit heuristics' } });
    }
  })) {
    throw new Error('could not read model metadata');
  }
  const hardware = hw!;
  const modelFacts = facts!;

  // ── hf_metadata (direct Hugging Face fetch) ──────────────────────────
  if (answers.allow_network) {
    await runStage('hf_metadata', async () => {
      const base = await resolveBaseModelRepo(modelFacts.base_model_repo, modelFacts.checkpoint, signal);
      if (!base) {
        cb.stage('hf_metadata', 'running', { data: { note: 'no sampling defaults published (base model not resolvable)' } });
        return;
      }
      const found = await fetchSamplingDefaults(base, signal);
      if (found) {
        sampling = found;
        cb.stage('hf_metadata', 'running', { data: { base_model: base, sampling_source: found.source } });
      } else {
        cb.stage('hf_metadata', 'running', { data: { base_model: base, note: 'no sampling defaults published' } });
      }
    });
  } else {
    cb.stage('hf_metadata', 'skipped');
  }
  throwIfAborted();

  // Candidate backends (GPU variants only).
  const candidates: string[] = [];
  for (const b of hardware.installed_backends) {
    if (b === 'cpu' || b === 'system' || b === 'metal') continue;
    if (answers.backends_to_consider.length > 0 && !answers.backends_to_consider.includes(b)) continue;
    candidates.push(b);
  }
  if (candidates.length === 0 && hardware.installed_backends.length > 0) {
    candidates.push(hardware.installed_backends[0]);
  }

  // ── fit_estimate (heuristic, no fit-params endpoint) ─────────────────
  await runStage('fit_estimate', async () => {
    const avail = availableMib(hardware);
    for (const backend of candidates) {
      const fit = computeFitEstimate({
        backend,
        availableMib: avail,
        weightsMib: modelFacts.weights_mib,
        kvBytesPerToken: modelFacts.kv_bytes_per_token,
        blockCount: modelFacts.block_count,
        isMoe: modelFacts.is_moe,
        nCtxTrain: modelFacts.n_ctx_train,
        degraded: !modelFacts.metadata_present,
      });
      cb.fit(fit);
      fits.push(fit);
    }
    if (fits.length === 0) throw new Error('no GPU backend available to fit against');
  });
  throwIfAborted();

  // For benchmark tiers the probes own the GPU: unload everything (consent
  // enforced by the wizard) and never reload — apply/load does that.
  if (withBench && request.allow_unload) {
    await api.unloadModel().catch(() => {});
  }

  // Base args for measurement, mirroring the recommended non-batch flags so the
  // duel reflects the config the preset will actually use.
  const kv = answers.kv_cache_quant;
  const baseMeasureArgs = kv !== 'none' ? `-ctk ${kv} -ctv ${kv}` : '';
  const primaryFit = fits.find(f => f.ok && f.backend === candidates[0]) || fits[0] || null;
  const effectiveCtx = primaryFit
    ? (primaryFit.fitted_ctx > 0 ? primaryFit.fitted_ctx : modelFacts.n_ctx_train)
    : modelFacts.n_ctx_train;

  const ctxForDepth = (depth: number): number => {
    if (depth <= 0) return Math.min(4096, roundUpCtx(2048));
    return roundUpCtx(depth + 2048);
  };

  // One measured configuration: reload before EVERY run to clear the prompt
  // cache, then time a single non-streaming completion. A load failure is a
  // first-class outcome (ok:false); the caller may back off.
  const measureConfig = async (
    backend: string,
    label: string,
    ctx: number,
    extraArgs: string,
    params: BenchPoint['params'],
    depthTokens: number,
  ): Promise<BenchPoint> => {
    const point: BenchPoint = {
      backend, label, ctx_size: ctx, llamacpp_args: extraArgs, params,
      ttft_ms: 0, tps: 0, vram_gb: -1, ok: false,
    };
    const ttfts: number[] = [];
    const tpsList: number[] = [];
    const vrams: number[] = [];
    const prompt = depthPrompt(depthTokens);
    try {
      for (let i = 0; i < MEASURE_RUNS; i++) {
        throwIfAborted();
        await api.unloadModel().catch(() => {});
        await api.benchLoadModel(request.model, { backend, ctx_size: ctx, llamacpp_args: extraArgs }, signal);
        throwIfAborted();
        const resp = await api.chatCompletionRaw(
          request.model, [{ role: 'user', content: prompt }],
          { max_completion_tokens: 128, temperature: 0 }, signal);
        const m = extractMetrics(resp);
        const runOk = m.ttftMs > 0 || m.tps > 0 || m.tokens > 0;
        if (runOk) {
          if (m.ttftMs > 0) ttfts.push(m.ttftMs);
          if (m.tps > 0) tpsList.push(m.tps);
          point.ok = true;
        }
        const stats = await api.systemStats().catch(() => null);
        if (stats && typeof stats.vram_gb === 'number' && stats.vram_gb > 0) vrams.push(stats.vram_gb);
      }
    } catch (err) {
      if (isAbort(err)) throw err;
      point.ok = false;
      point.error = errorMessage(err);
    } finally {
      await api.unloadModel().catch(() => {});
    }
    point.ttft_ms = mean(ttfts);
    point.tps = mean(tpsList);
    point.vram_gb = vrams.length ? Math.max(...vrams) : -1;
    return point;
  };

  const pushBench = (point: BenchPoint) => { cb.bench([point]); bench.push(point); };

  // ── bench_matrix ─────────────────────────────────────────────────────
  if (withBench) {
    await runStage('bench_matrix', async () => {
      const depths: number[] = [0];
      if (effectiveCtx >= 32768) depths.push(30000);
      else if (effectiveCtx >= 8192) depths.push(Math.floor(0.8 * effectiveCtx));

      // Backend duel at each depth (single-candidate boxes get a baseline).
      const duelBackends = candidates.length ? candidates : ['vulkan'];
      for (const backend of duelBackends) {
        for (const depth of depths) {
          cb.progress(`Benchmarking ${backend} at depth ${depth}`);
          const point = await measureConfig(
            backend, `${backend} · d${depth}`, ctxForDepth(depth), baseMeasureArgs, { d: depth }, depth);
          pushBench(point);
          throwIfAborted();
        }
      }

      // Batch ladder + MTP sweep run on the provisional duel winner.
      let winner = duelBackends[0];
      const measured = candidates.length > 1 ? pickBenchWinner(bench, candidates) : '';
      if (measured) winner = measured;

      if (hardware.ram_is_vram && hardware.host_ram_gb >= 32) {
        const rungs = budget === 'thorough' ? [512, 1024, 2048, 4096, 8192] : [512, 2048, 8192];
        for (const r of rungs) {
          cb.progress(`Batch ladder -b ${r} on ${winner}`);
          const ladderArgs = `${baseMeasureArgs} -b ${r} -ub ${r}`.trim();
          const point = await measureConfig(
            winner, `${winner} · b${r}`, ctxForDepth(0), ladderArgs, { ladder: true, b: r, ub: r, d: 0 }, 0);
          pushBench(point);
          throwIfAborted();
        }
      }

      // MTP draft-length sweep: reload per draft length (measureConfig reloads
      // internally per run), timings from llama-server.
      if (modelFacts.has_mtp) {
        const ns = budget === 'thorough' ? [1, 2, 3, 4, 5, 6] : [2, 3, 4];
        for (const n of ns) {
          cb.progress(`MTP draft sweep n=${n} on ${winner}`);
          const specArgs = `${baseMeasureArgs} --spec-type draft-mtp --spec-draft-n-max ${n} --spec-draft-p-min 0.75`.trim();
          const point = await measureConfig(
            winner, `${winner} · mtp${n}`, ctxForDepth(0), specArgs, { spec_n: n, d: 0 }, 0);
          pushBench(point);
          throwIfAborted();
        }
      }
    });
  } else {
    cb.stage('bench_matrix', 'skipped');
  }
  throwIfAborted();

  // ── synthesize ───────────────────────────────────────────────────────
  let result: AutoOptResult | null = null;
  if (!await runStage('synthesize', async () => {
    result = synthesize(hardware, modelFacts, answers, fits, bench, sampling);
  })) {
    throw new Error('preset synthesis failed');
  }
  const synthesized = result!;

  // ── load_test (test-by-failure) ──────────────────────────────────────
  // Actually load the recommended config; on OOM/error, back off (ctx down,
  // then more offload) and retry, recording each step in the rationale.
  // Skipped for Fast Scan, which promises no loads and no evictions.
  if (withBench) {
    await runStage('load_test', async () => {
      await loadTestWithBackoff(request.model, synthesized.primary, modelFacts, cb, throwIfAborted, signal);
    });
  } else {
    cb.stage('load_test', 'skipped');
    synthesized.primary.rationale.push(
      'Fast Scan: configuration derived from heuristics and not load-validated. '
      + 'Run a Benchmark to confirm it loads on this hardware.');
  }

  const summary = `${synthesized.primary.llamacpp_backend} · ctx ${synthesized.primary.ctx_size}`
    + (synthesized.primary.llamacpp_args ? ` · ${synthesized.primary.llamacpp_args}` : '');
  return { result: synthesized, summary };
}

function reduceCtx(primary: AutoOptRecommendation): boolean {
  const lower = roundDownCtx(Math.max(2048, primary.ctx_size - 1));
  if (lower >= primary.ctx_size) return false;
  primary.ctx_size = lower;
  return true;
}

function increaseOffload(primary: AutoOptRecommendation, mf: ModelFacts): boolean {
  const args = primary.llamacpp_args;
  if (mf.is_moe) {
    const m = /--n-cpu-moe (\d+)/.exec(args);
    if (m) {
      const next = Math.min(mf.block_count || Number(m[1]) + 4, Number(m[1]) + 4);
      if (next > Number(m[1])) {
        primary.llamacpp_args = args.replace(/--n-cpu-moe \d+/, `--n-cpu-moe ${next}`);
        return true;
      }
    }
    if (!/--cpu-moe/.test(args)) {
      primary.llamacpp_args = args.replace(/--n-cpu-moe \d+/, '--cpu-moe').trim();
      if (!/--cpu-moe/.test(primary.llamacpp_args)) primary.llamacpp_args = `--cpu-moe ${args}`.trim();
      return true;
    }
    return false;
  }
  const m = /-ngl (\d+)/.exec(args);
  const current = m ? Number(m[1]) : (mf.block_count || 32);
  const next = Math.max(0, Math.floor(current / 2));
  if (m) {
    if (next >= current) return false;
    primary.llamacpp_args = args.replace(/-ngl \d+/, `-ngl ${next}`);
  } else {
    primary.llamacpp_args = `-ngl ${next} ${args}`.trim();
  }
  return true;
}

async function loadTestWithBackoff(
  model: string,
  primary: AutoOptRecommendation,
  mf: ModelFacts,
  cb: ControllerCallbacks,
  throwIfAborted: () => void,
  signal: AbortSignal,
): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt <= MAX_LOAD_BACKOFFS; attempt++) {
    throwIfAborted();
    cb.progress(`Load test: ctx ${primary.ctx_size}${primary.llamacpp_args ? ` · ${primary.llamacpp_args}` : ''}`);
    try {
      await api.unloadModel().catch(() => {});
      await api.benchLoadModel(model, {
        backend: primary.llamacpp_backend,
        ctx_size: primary.ctx_size,
        llamacpp_args: primary.llamacpp_args,
      }, signal);
      await api.unloadModel().catch(() => {});
      if (attempt > 0) {
        primary.rationale.push(`Load test passed after ${attempt} backoff${attempt > 1 ? 's' : ''} `
          + `(final: ctx ${primary.ctx_size}${primary.llamacpp_args ? `, ${primary.llamacpp_args}` : ''}).`);
      } else {
        primary.rationale.push('Load test passed: the recommended configuration loads on this hardware.');
      }
      return;
    } catch (err) {
      if (isAbort(err)) throw err;
      lastError = errorMessage(err);
      await api.unloadModel().catch(() => {});
      if (attempt === MAX_LOAD_BACKOFFS) break;
      // Back off deterministically: shrink context first, then add offload.
      const before = { ctx: primary.ctx_size, args: primary.llamacpp_args };
      if (!reduceCtx(primary)) {
        if (!increaseOffload(primary, mf)) break;
      }
      cb.progress(`Load failed (${lastError}); backing off `
        + `(ctx ${before.ctx}→${primary.ctx_size})`);
      primary.rationale.push(`Load failed at ctx ${before.ctx}${before.args ? `, ${before.args}` : ''} `
        + `(${lastError}); backed off to ctx ${primary.ctx_size}`
        + `${primary.llamacpp_args !== before.args ? `, ${primary.llamacpp_args}` : ''}.`);
    }
  }
  throw new Error(`the recommended configuration would not load after ${MAX_LOAD_BACKOFFS} backoffs: ${lastError}`);
}
