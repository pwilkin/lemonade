import {
  AutoOptRecommendation,
  AutoOptResult,
  BenchPoint,
  FitEstimate,
  HardwareSnapshot,
  ModelFacts,
  SamplingDefaults,
  WizardAnswers,
} from './autoOptTypes';

const CTX_STEPS = [262144, 131072, 98304, 65536, 49152, 32768, 24576, 16384, 12288, 8192, 6144, 4096, 2048];

export function roundDownCtx(ctx: number): number {
  for (const s of CTX_STEPS) {
    if (ctx >= s) return s;
  }
  return ctx >= 1024 ? Math.floor(ctx) : 1024;
}

export function kvQuantFactor(q: string): number {
  if (q === 'q8_0') return 0.5;
  if (q === 'q5_1') return 0.375;
  if (q === 'q4_0') return 0.28125;
  return 1.0;
}

export function fitTotalMib(fit: FitEstimate): number {
  let total = 0;
  for (const d of fit.devices) {
    if (!d.device.startsWith('Host')) total += d.model_mib + d.ctx_mib + d.compute_mib;
  }
  return total;
}

function findBench(
  bench: BenchPoint[],
  backend: string,
  depth: number,
  extraKey?: 'spec_n' | 'b',
  extraVal = 0,
): BenchPoint | null {
  for (const p of bench) {
    if (!p.ok || p.backend !== backend || p.n_depth !== depth) continue;
    if (extraKey) {
      if (p.params?.[extraKey] !== extraVal) continue;
    } else if (p.params?.spec_n !== undefined || p.params?.ladder !== undefined) {
      continue;
    }
    return p;
  }
  return null;
}

function findFit(fits: FitEstimate[], backend: string, extraArgs = ''): FitEstimate | null {
  for (const f of fits) {
    if (f.ok && f.backend === backend && f.extra_args === extraArgs) return f;
  }
  return null;
}

function pickBackend(
  bench: BenchPoint[],
  candidates: string[],
): { backend: string; deep: BenchPoint | null } {
  let best = '';
  let bestScore = -1;
  let bestDeep: BenchPoint | null = null;
  for (const c of candidates) {
    const d0 = findBench(bench, c, 0);
    if (!d0) continue;
    let deep: BenchPoint | null = null;
    for (const bp of bench) {
      if (bp.ok && bp.backend === c && bp.n_depth > 0
          && bp.params?.ladder === undefined && bp.params?.spec_n === undefined) {
        deep = bp;
      }
    }
    const pp0 = d0.pp_avg_ts;
    const tg = deep ? deep.tg_avg_ts : d0.tg_avg_ts;
    const ppd = deep ? deep.pp_avg_ts : d0.pp_avg_ts;
    const score = 0.35 * pp0 + 0.45 * tg * 10 + 0.20 * ppd;
    if (score > bestScore) {
      bestScore = score;
      best = c;
      bestDeep = deep || d0;
    }
  }
  return { backend: best, deep: best ? bestDeep : null };
}

export function pickBenchWinner(bench: BenchPoint[], candidates: string[]): string {
  return pickBackend(bench, candidates).backend;
}

function fmt1(v: number): string {
  return v.toFixed(1);
}

/** "https://huggingface.co/Qwen/Qwen3-32B" or "repo:variant" -> "org/repo" ('' when unresolvable). */
export function checkpointRepoId(checkpoint: string): string {
  const marker = 'huggingface.co/';
  const pos = checkpoint.indexOf(marker);
  let id = pos === -1 ? checkpoint : checkpoint.slice(pos + marker.length);
  while (id.endsWith('/')) id = id.slice(0, -1);
  const colon = id.indexOf(':');
  if (colon !== -1) id = id.slice(0, colon);
  return id.includes('/') ? id : '';
}

/**
 * llama-fit-params only understands memory-relevant llama flags; host-side
 * flags (--cache-ram, --spec-*, --direct-io) make it exit with a usage error,
 * so load validation probes with the memory subset only.
 */
export function validationFlagSubset(ctxSize: number, llamacppArgs: string): string[] {
  const memFlags = new Set([
    '-c', '-ctk', '-ctv', '-np', '-kvu', '-no-kvu', '-b', '-ub',
    '--cpu-moe', '--n-cpu-moe', '-ngl', '--split-mode', '-sm',
  ]);
  const tokens: string[] = ['-c', String(ctxSize)];
  let takeValue = false;
  for (const cur of llamacppArgs.split(/\s+/).filter(Boolean)) {
    if (takeValue) {
      tokens.push(cur);
      takeValue = false;
      continue;
    }
    if (memFlags.has(cur)) {
      tokens.push(cur);
      takeValue = cur !== '-kvu' && cur !== '-no-kvu' && cur !== '--cpu-moe';
    }
  }
  return tokens;
}

// ── The 12-lever synthesis (1:1 port of the C++ engine) ────────────────

export function synthesize(
  hw: HardwareSnapshot,
  mf: ModelFacts,
  ans: WizardAnswers,
  fits: FitEstimate[],
  bench: BenchPoint[],
  sampling?: SamplingDefaults,
): AutoOptResult {
  const p: AutoOptRecommendation = {
    label: 'Recommended',
    llamacpp_backend: '',
    ctx_size: -1,
    mmproj_enabled: true,
    llamacpp_args: '',
    rationale: [],
  };
  const result: AutoOptResult = { primary: p, alternatives: [] };
  const args: string[] = [];

  // Lever 6 (+3 constraint): backend selection — measured duel when bench
  // data exists, install-order heuristic otherwise.
  const candidates: string[] = [];
  for (const b of hw.installed_backends) {
    if (b === 'cpu' || b === 'system') continue;
    if (ans.backends_to_consider.length > 0 && !ans.backends_to_consider.includes(b)) continue;
    candidates.push(b);
  }
  if (candidates.length === 0) candidates.push('vulkan');

  let backend = candidates[0];
  if (candidates.length > 1) {
    const { backend: measured, deep: bestDeep } = pickBackend(bench, candidates);
    if (measured) backend = measured;
    if (bestDeep) {
      const others = candidates.filter(c => c !== backend).join(', ');
      p.rationale.push(`${backend} chosen over ${others}: best measured decode/prefill balance `
        + `on this model (${fmt1(bestDeep.tg_avg_ts)} tok/s decode at depth ${bestDeep.n_depth}).`);
    } else {
      p.rationale.push(`${backend} chosen (first installed GPU backend; enable a benchmark `
        + 'budget to measure alternatives).');
    }
  }
  p.llamacpp_backend = backend;
  const isRocm = backend.startsWith('rocm');
  const isCuda = backend === 'cuda';

  // Lever 10: mmap is broken on ROCm; direct I/O is the faster workaround.
  if (isRocm) {
    args.push('--direct-io');
    p.rationale.push('--direct-io: works around broken mmap on ROCm with faster cold loads '
      + 'than --no-mmap.');
  }

  // Lever 8: vision projector.
  if (mf.has_vision) {
    if (ans.use_vision === false) {
      p.mmproj_enabled = false;
      let freed = 0;
      const withMm = findFit(fits, backend);
      const withoutMm = findFit(fits, backend, '--no-mmproj');
      if (withMm && withoutMm) freed = fitTotalMib(withMm) - fitTotalMib(withoutMm);
      p.rationale.push('Vision projector disabled per your answer'
        + (freed > 0 ? ` — frees ~${freed} MiB for context.` : ' — its memory goes to context instead.'));
    } else {
      p.rationale.push('Vision projector kept loaded (image input enabled).');
    }
  }

  // Lever 11: fit strategy for models that don't fully fit.
  const fit = findFit(fits, backend);
  let usedCpuMoe = false;
  if (fit && !fit.fits_fully) {
    if (mf.is_moe && fit.fitted_ncmoe > 0) {
      args.push(`--n-cpu-moe ${fit.fitted_ncmoe}`);
      usedCpuMoe = true;
      p.rationale.push(`Model exceeds GPU memory: expert tensors of the first ${fit.fitted_ncmoe} `
        + 'layers stay on CPU (--n-cpu-moe) — attention and shared tensors keep GPU speed.');
    } else if (mf.is_moe) {
      args.push('--cpu-moe');
      usedCpuMoe = true;
      p.rationale.push('Model exceeds GPU memory: all expert tensors on CPU (--cpu-moe); '
        + 'the GPU keeps the non-expert layers.');
    } else if (fit.fitted_ngl >= 0) {
      p.rationale.push(`Model exceeds GPU memory: llama.cpp will offload ${fit.fitted_ngl} `
        + 'layers to GPU and run the rest on CPU (expect reduced speed).');
    }
  }

  // Lever 2 + ctx: KV-cache quantization (user pick is a constraint) and the
  // largest context that fits under it.
  const kv = ans.kv_cache_quant;
  if (kv !== 'none') {
    args.push(`-ctk ${kv} -ctv ${kv}`);
    const note = kv === 'q8_0'
      ? 'roughly doubles usable context with negligible quality loss'
      : (kv === 'q5_1'
        ? 'about 2.7x context capacity with a slight quality cost'
        : 'about 3.5x context capacity; quality measurably degrades on very long contexts');
    p.rationale.push(`KV cache quantized to ${kv}: ${note}.`);
  }
  let ctx = mf.n_ctx_train > 0 ? mf.n_ctx_train : 32768;
  if (fit && fit.fitted_ctx > 0) {
    let fitCtx = fit.fitted_ctx;
    if (kv !== 'none') fitCtx = Math.floor(fitCtx / kvQuantFactor(kv));
    ctx = Math.min(ctx, fitCtx);
  }
  p.ctx_size = roundDownCtx(ctx);
  if (mf.n_ctx_train > 0 && p.ctx_size >= mf.n_ctx_train) {
    p.ctx_size = mf.n_ctx_train;
    p.rationale.push(`Context ${p.ctx_size}: the model's full trained window fits.`);
  } else {
    p.rationale.push(`Context ${p.ctx_size}: the largest standard size that fits in memory`
      + (kv !== 'none' ? ' with the quantized KV cache.' : '.'));
  }

  // Lever 1: prompt-cache checkpoints scale (user pick constrained by the
  // hybrid/recurrent bump).
  let headroom = ans.ram_headroom;
  if (mf.is_hybrid_or_recurrent && headroom === 'disabled') {
    headroom = 'minimal';
    p.rationale.push('Prompt-cache checkpoints kept at minimal instead of disabled: this '
      + 'architecture (hybrid/recurrent) must re-process the whole prompt on any cache miss.');
  }
  if (headroom === 'reduced') args.push('--cache-ram 4096 -ctxcp 16');
  else if (headroom === 'minimal') args.push('--cache-ram 2048 -ctxcp 8');
  else if (headroom === 'disabled') args.push('--cache-ram 0 -ctxcp 0');
  if (headroom !== 'normal') {
    p.rationale.push(`Prompt-cache RAM capped (${headroom})`
      + (hw.ram_is_vram
        ? ' — on this machine system RAM and GPU memory share one pool.'
        : ' to keep system RAM free.'));
  }

  // Lever 4: parallel slots.
  if (ans.parallel && ans.slots > 1) {
    const np = Math.min(Math.max(ans.slots, 2), 8);
    if (ans.dedicated_slots) {
      args.push(`-np ${np} -no-kvu`);
      p.rationale.push(`${np} parallel slots with dedicated context: each request is `
        + `guaranteed ${Math.floor(p.ctx_size / np)} tokens (${p.ctx_size} / ${np}).`);
    } else {
      args.push(`-np ${np} -kvu`);
      p.rationale.push(`${np} parallel slots sharing one context pool: long requests can use `
        + 'most of the window, but concurrent long requests race for it.');
    }
  }

  // Lever 5: speculative decoding.
  args.push('--spec-default');
  p.rationale.push('--spec-default: n-gram speculative decoding is effectively free.');
  if (mf.has_mtp) {
    let bestN = 3;
    let bestTs = -1;
    for (const bp of bench) {
      if (!bp.ok || bp.params?.spec_n === undefined) continue;
      if (bp.tg_avg_ts > bestTs) {
        bestTs = bp.tg_avg_ts;
        bestN = bp.params.spec_n;
      }
    }
    args.push(`--spec-type draft-mtp --spec-draft-n-max ${bestN}`);
    p.rationale.push(bestTs > 0
      ? `MTP draft length ${bestN} measured fastest on this machine (${fmt1(bestTs)} tok/s).`
      : 'Model has MTP heads: draft-based speculative decoding enabled (default draft length 3).');
  }

  // Lever 9: batch/ubatch ladder for unified-memory boxes with RAM to spare.
  const bigIgpu = hw.ram_is_vram && hw.host_ram_gb >= 32;
  {
    let bestB = 0;
    let bestPp = -1;
    let basePp = -1;
    for (const bp of bench) {
      if (!bp.ok || bp.params?.ladder === undefined || bp.backend !== backend || bp.n_depth !== 0) continue;
      const b = bp.params.b ?? 0;
      if (b === 512) basePp = bp.pp_avg_ts;
      if (bp.pp_avg_ts > bestPp) {
        bestPp = bp.pp_avg_ts;
        bestB = b;
      }
    }
    if (bestB > 512 && basePp > 0 && bestPp > basePp * 1.05) {
      args.push(`-b ${bestB} -ub ${bestB}`);
      p.rationale.push(`Batch size ${bestB} measured ${fmt1((bestPp / basePp - 1) * 100)}% `
        + 'faster prefill than the default 512 (costs some memory for compute buffers).');
    } else if (bench.length === 0 && bigIgpu) {
      args.push('-b 2048 -ub 2048');
      p.rationale.push('Batch size 2048: unified-memory machines with ample RAM prefill much '
        + 'faster with larger batches (heuristic — run a standard/thorough pass to measure).');
    }
  }

  // Lever 3: tensor parallelism across identical GPUs (CUDA-only today).
  if (isCuda) {
    let sameFamily = 0;
    for (let i = 0; i < hw.gpus.length; i++) {
      for (let j = i + 1; j < hw.gpus.length; j++) {
        if (hw.gpus[i].vendor === 'nvidia' && hw.gpus[i].family === hw.gpus[j].family) sameFamily++;
      }
    }
    if (sameFamily > 0) {
      args.push('--split-mode tensor');
      p.rationale.push('Two or more identical NVIDIA GPUs: tensor parallelism speeds up decode '
        + '(prefill gets slightly slower).');
    }
  }

  // Lever 7: sampling defaults pass through (request-time, not load flags).
  result.sampling_defaults = sampling;
  if (sampling) {
    p.rationale.push('Sampling defaults (temperature/top-p/top-k) taken from the base model\'s '
      + `generation_config (${sampling.source}).`);
  }

  const join = (v: string[]) => v.join(' ');
  p.llamacpp_args = join(args);
  if (fit) {
    const d0 = findBench(bench, backend, 0);
    p.expected = { vram_mib: fitTotalMib(fit) };
    if (d0) {
      p.expected.pp_ts = d0.pp_avg_ts;
      p.expected.tg_ts = d0.tg_avg_ts;
    }
  }

  // ── Alternatives ───────────────────────────────────────────────────
  if (kv !== 'none') {
    const alt: AutoOptRecommendation = {
      ...p,
      label: 'Maximum quality',
      tradeoff: 'smaller context window',
      rationale: ['Unquantized f16 KV cache: no quality risk; context shrinks to what fits.'],
      llamacpp_args: join(args.filter(a => !a.startsWith('-ctk'))),
      ctx_size: roundDownCtx(Math.max(Math.floor(p.ctx_size * kvQuantFactor(kv)), 4096)),
      expected: undefined,
    };
    result.alternatives.push(alt);
  }
  if (kv !== 'q4_0') {
    const a2: string[] = [];
    let replaced = false;
    for (const a of args) {
      if (a.startsWith('-ctk')) {
        a2.push('-ctk q4_0 -ctv q4_0');
        replaced = true;
      } else {
        a2.push(a);
      }
    }
    if (!replaced) a2.unshift('-ctk q4_0 -ctv q4_0');
    const altCtx = Math.floor(p.ctx_size * kvQuantFactor(kv) / kvQuantFactor('q4_0'));
    const alt: AutoOptRecommendation = {
      ...p,
      label: 'Maximum context',
      tradeoff: 'quality degrades on very long contexts',
      rationale: ['q4_0 KV cache: about 3.5x the context capacity of f16; noticeable quality '
        + 'loss past ~32k tokens of active context.'],
      llamacpp_args: join(a2),
      ctx_size: roundDownCtx(Math.min(altCtx, mf.n_ctx_train > 0 ? mf.n_ctx_train : altCtx)),
      expected: undefined,
    };
    result.alternatives.push(alt);
  }
  if (usedCpuMoe) {
    const alt: AutoOptRecommendation = {
      ...p,
      label: 'Conservative offload',
      tradeoff: 'slower, but immune to memory pressure',
      rationale: ['All expert tensors on CPU (--cpu-moe): the safest fit when other '
        + 'applications compete for GPU memory.'],
      llamacpp_args: join(args.map(a => a.startsWith('--n-cpu-moe') ? '--cpu-moe' : a)),
      expected: undefined,
    };
    result.alternatives.push(alt);
  }

  return result;
}
