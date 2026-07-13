const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const webpack = require('webpack');

async function bundleSynthesize() {
  const outputPath = fs.mkdtempSync(path.join(os.tmpdir(), 'lemonade-autoopt-synth-'));
  const config = {
    mode: 'development',
    target: 'node',
    entry: path.resolve(__dirname, '../src/features/autoOpt/autoOptSynthesize.ts'),
    output: {
      path: outputPath,
      filename: 'autoOptSynthesize.cjs',
      library: { type: 'commonjs2' },
    },
    resolve: { extensions: ['.ts', '.tsx', '.js'] },
    module: {
      rules: [{ test: /\.tsx?$/, use: 'ts-loader', exclude: /node_modules/ }],
    },
    optimization: { minimize: false },
  };

  await new Promise((resolve, reject) => {
    webpack(config, (error, stats) => {
      if (error) return reject(error);
      if (stats?.hasErrors()) return reject(new Error(stats.toString({ all: false, errors: true })));
      resolve();
    });
  });

  return { outputPath, modulePath: path.join(outputPath, 'autoOptSynthesize.cjs') };
}

function igpuHw() {
  return {
    gpus: [{ vendor: 'amd', name: 'Radeon 8060S', family: 'gfx1151', vram_gb: 96.0 }],
    has_igpu: true,
    ram_is_vram: true,
    host_ram_gb: 128,
    installed_backends: ['vulkan', 'rocm-stable'],
    os: 'linux',
  };
}

function dgpuHw() {
  return {
    gpus: [{ vendor: 'nvidia', name: 'RTX 4090', family: 'sm_89', vram_gb: 24.0 }],
    has_igpu: false,
    ram_is_vram: false,
    host_ram_gb: 64,
    installed_backends: ['cuda'],
    os: 'linux',
  };
}

function denseModel() {
  return {
    architecture: 'qwen3',
    block_count: 36,
    expert_count: 0,
    full_attention_interval: 0,
    swa_layer_count: 0,
    n_ctx_train: 131072,
    kv_bytes_per_token: 90112,
    is_moe: false,
    is_hybrid_or_recurrent: false,
    has_mtp: false,
    has_vision: false,
    base_model_repo: '',
    checkpoint: '',
  };
}

function answers(overrides = {}) {
  return {
    parallel: false,
    slots: 1,
    dedicated_slots: true,
    kv_cache_quant: 'none',
    ram_headroom: 'normal',
    use_vision: undefined,
    allow_network: true,
    backends_to_consider: [],
    ...overrides,
  };
}

function fitting(backend, ctx, extra = '') {
  return {
    backend,
    fit_target_mib: 1024,
    extra_args: extra,
    fitted_args: `-c ${ctx} -ngl -1`,
    fitted_ctx: ctx,
    fitted_ngl: -1,
    fitted_ncmoe: 0,
    devices: [{ device: 'Vulkan0', model_mib: 17408, ctx_mib: 6144, compute_mib: 910 }],
    fits_fully: true,
    ok: true,
    error: '',
  };
}

function hasArg(preset, needle) {
  return preset.llamacpp_args.includes(needle);
}

let failures = 0;
function check(name, ok) {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}`);
  if (!ok) failures += 1;
}

(async () => {
  const { outputPath, modulePath } = await bundleSynthesize();

  try {
    const engine = require(modulePath);
    const { synthesize, checkpointRepoId, validationFlagSubset } = engine;

    // ── checkpoint_repo_id ────────────────────────────────────────────
    check('ckpt: plain repo', checkpointRepoId('Qwen/Qwen3-32B') === 'Qwen/Qwen3-32B');
    check('ckpt: variant suffix stripped',
      checkpointRepoId('ilintar/Agents-A1-GGUF:IQ4_XS') === 'ilintar/Agents-A1-GGUF');
    check('ckpt: url form',
      checkpointRepoId('https://huggingface.co/Qwen/Qwen3-32B') === 'Qwen/Qwen3-32B');
    check('ckpt: url with trailing slash',
      checkpointRepoId('https://huggingface.co/Qwen/Qwen3-32B/') === 'Qwen/Qwen3-32B');
    check('ckpt: bare name rejected', checkpointRepoId('just-a-name') === '');
    check('ckpt: empty rejected', checkpointRepoId('') === '');

    // ── validation_flag_subset ────────────────────────────────────────
    {
      const v = validationFlagSubset(
        65536,
        '-ctk q8_0 -ctv q8_0 --cache-ram 4096 -ctxcp 16 --spec-default '
        + '--spec-type draft-mtp --spec-draft-n-max 3 --direct-io -np 4 -no-kvu '
        + '--n-cpu-moe 12 -b 2048 -ub 2048');
      const joined = v.join(' ');
      check('valid: ctx first', v.length >= 2 && v[0] === '-c' && v[1] === '65536');
      check('valid: value flags kept with values',
        joined.includes('-ctk q8_0')
        && joined.includes('--n-cpu-moe 12')
        && joined.includes('-b 2048 -ub 2048')
        && joined.includes('-np 4'));
      check('valid: bare flags kept without eating values', joined.includes('-no-kvu --n-cpu-moe'));
      check('valid: host-side flags excluded',
        !joined.includes('cache-ram') && !joined.includes('ctxcp')
        && !joined.includes('spec') && !joined.includes('direct-io'));
      check('valid: spec values not orphaned',
        !joined.includes('draft-mtp') && !joined.includes(' 3'));

      const empty = validationFlagSubset(8192, '');
      check('valid: empty args -> ctx only', empty.length === 2 && empty[1] === '8192');
    }

    // ── lever 1: cache ram scale ──────────────────────────────────────
    {
      let r = synthesize(igpuHw(), denseModel(), answers({ ram_headroom: 'reduced' }),
        [fitting('vulkan', 65536)], [], undefined);
      check('L1: reduced scale', hasArg(r.primary, '--cache-ram 4096 -ctxcp 16'));

      r = synthesize(igpuHw(), denseModel(), answers({ ram_headroom: 'normal' }),
        [fitting('vulkan', 65536)], [], undefined);
      check('L1: normal omits flags', !hasArg(r.primary, '--cache-ram'));
    }

    // ── lever 1: hybrid bump ──────────────────────────────────────────
    {
      const mf = { ...denseModel(), full_attention_interval: 4, is_hybrid_or_recurrent: true };
      const r = synthesize(igpuHw(), mf, answers({ ram_headroom: 'disabled' }),
        [fitting('vulkan', 65536)], [], undefined);
      check('L1: hybrid never disabled', hasArg(r.primary, '--cache-ram 2048 -ctxcp 8'));
      check('L1: hybrid bump explained',
        r.primary.rationale.some(s => s.includes('hybrid/recurrent')));
    }

    // ── lever 2: kv quant + alternatives ──────────────────────────────
    {
      const r = synthesize(igpuHw(), denseModel(), answers({ kv_cache_quant: 'q8_0' }),
        [fitting('vulkan', 32768)], [], undefined);
      check('L2: ctk/ctv emitted', hasArg(r.primary, '-ctk q8_0 -ctv q8_0'));
      check('L2: ctx doubled by q8_0', r.primary.ctx_size === 65536);
      check('L2: max-quality alternative present',
        r.alternatives.length > 0 && r.alternatives[0].label === 'Maximum quality'
        && !r.alternatives[0].llamacpp_args.includes('-ctk'));
      check('L2: max-context alternative q4_0',
        r.alternatives.length >= 2 && r.alternatives[1].llamacpp_args.includes('-ctk q4_0'));
    }

    // ── lever 3: tensor split ─────────────────────────────────────────
    {
      const hw = dgpuHw();
      hw.gpus.push({ vendor: 'nvidia', name: 'RTX 4090', family: 'sm_89', vram_gb: 24.0 });
      let r = synthesize(hw, denseModel(), answers(), [fitting('cuda', 65536)], [], undefined);
      check('L3: tensor split on dual identical CUDA', hasArg(r.primary, '--split-mode tensor'));

      const vk = igpuHw();
      vk.gpus.push({ vendor: 'amd', name: 'RX 7900', family: 'gfx1100', vram_gb: 24.0 });
      r = synthesize(vk, denseModel(), answers(), [fitting('vulkan', 65536)], [], undefined);
      check('L3: no tensor split off CUDA', !hasArg(r.primary, '--split-mode'));
    }

    // ── lever 4: parallel slots ───────────────────────────────────────
    {
      let r = synthesize(igpuHw(), denseModel(),
        answers({ parallel: true, slots: 4, dedicated_slots: true }),
        [fitting('vulkan', 65536)], [], undefined);
      check('L4: dedicated slots', hasArg(r.primary, '-np 4 -no-kvu'));
      check('L4: per-slot math in rationale',
        r.primary.rationale.some(s => s.includes(String(Math.floor(r.primary.ctx_size / 4)))));

      r = synthesize(igpuHw(), denseModel(),
        answers({ parallel: true, slots: 4, dedicated_slots: false }),
        [fitting('vulkan', 65536)], [], undefined);
      check('L4: shared pool', hasArg(r.primary, '-np 4 -kvu'));
    }

    // ── lever 5: speculative decoding ─────────────────────────────────
    {
      let r = synthesize(igpuHw(), denseModel(), answers(), [fitting('vulkan', 65536)], [], undefined);
      check('L5: spec-default always', hasArg(r.primary, '--spec-default'));
      check('L5: no mtp flags on non-mtp', !hasArg(r.primary, 'draft-mtp'));

      const mtp = { ...denseModel(), has_mtp: true };
      const sweep = [];
      for (let n = 1; n <= 4; n++) {
        sweep.push({
          backend: 'vulkan', params: { spec_n: n }, pp_avg_ts: 0,
          tg_avg_ts: n === 2 ? 55.0 : 40.0 + n, n_depth: 0, ok: true,
        });
      }
      r = synthesize(igpuHw(), mtp, answers(), [fitting('vulkan', 65536)], sweep, undefined);
      check('L5: measured argmax n=2', hasArg(r.primary, '--spec-draft-n-max 2'));

      r = synthesize(igpuHw(), mtp, answers(), [fitting('vulkan', 65536)], [], undefined);
      check('L5: default n=3 unmeasured', hasArg(r.primary, '--spec-draft-n-max 3'));
    }

    // ── lever 6: backend duel ─────────────────────────────────────────
    {
      const duel = ['vulkan', 'rocm-stable'].map(b => ({
        backend: b,
        n_depth: 0,
        params: { d: 0 },
        pp_avg_ts: b === 'vulkan' ? 600 : 580,
        tg_avg_ts: b === 'vulkan' ? 42 : 37,
        ok: true,
      }));
      const r = synthesize(igpuHw(), denseModel(), answers(),
        [fitting('vulkan', 65536), fitting('rocm-stable', 65536)], duel, undefined);
      check('L6: measured winner', r.primary.llamacpp_backend === 'vulkan');

      const rh = synthesize(igpuHw(), denseModel(), answers(), [fitting('vulkan', 65536)], [], undefined);
      check('L6: heuristic fallback picks installed', rh.primary.llamacpp_backend === 'vulkan');
    }

    // ── lever 7: sampling passthrough ─────────────────────────────────
    {
      const sd = { temperature: 0.7, top_p: 0.8, top_k: 20, source: 'hf:Qwen/Qwen3-32B/generation_config.json' };
      const r = synthesize(igpuHw(), denseModel(), answers(), [fitting('vulkan', 65536)], [], sd);
      check('L7: sampling passthrough',
        r.sampling_defaults && r.sampling_defaults.temperature === 0.7);
      check('L7: sampling not in args', !hasArg(r.primary, 'temp'));
    }

    // ── lever 8: vision projector ─────────────────────────────────────
    {
      const mf = { ...denseModel(), has_vision: true };
      let r = synthesize(igpuHw(), mf, answers({ use_vision: false }),
        [fitting('vulkan', 65536)], [], undefined);
      check('L8: mmproj disabled', r.primary.mmproj_enabled === false);

      r = synthesize(igpuHw(), mf, answers({ use_vision: true }),
        [fitting('vulkan', 65536)], [], undefined);
      check('L8: mmproj kept', r.primary.mmproj_enabled === true);
    }

    // ── lever 9: batch ladder ─────────────────────────────────────────
    {
      const ladder = [512, 2048, 8192].map(b => ({
        backend: 'vulkan',
        n_depth: 0,
        params: { d: 0, ladder: true, b, ub: b },
        pp_avg_ts: b === 2048 ? 950 : (b === 8192 ? 900 : 600),
        tg_avg_ts: 40,
        ok: true,
      }));
      const r = synthesize(igpuHw(), denseModel(), answers(), [fitting('vulkan', 65536)], ladder, undefined);
      check('L9: measured best rung', hasArg(r.primary, '-b 2048 -ub 2048'));

      const rq = synthesize(igpuHw(), denseModel(), answers(), [fitting('vulkan', 65536)], [], undefined);
      check('L9: iGPU heuristic without bench', hasArg(rq.primary, '-b 2048 -ub 2048'));

      const rd = synthesize(dgpuHw(), denseModel(), answers(), [fitting('cuda', 65536)], [], undefined);
      check('L9: no heuristic on dGPU', !hasArg(rd.primary, '-b '));
    }

    // ── lever 10: ROCm direct-io ──────────────────────────────────────
    {
      let r = synthesize(igpuHw(), denseModel(), answers({ backends_to_consider: ['rocm-stable'] }),
        [fitting('rocm-stable', 65536)], [], undefined);
      check('L10: rocm gets --direct-io', hasArg(r.primary, '--direct-io'));

      r = synthesize(igpuHw(), denseModel(), answers({ backends_to_consider: ['vulkan'] }),
        [fitting('vulkan', 65536)], [], undefined);
      check('L10: vulkan does not', !hasArg(r.primary, '--direct-io'));
    }

    // ── lever 11: cpu-moe fit strategy ────────────────────────────────
    {
      const moe = { ...denseModel(), is_moe: true, expert_count: 128 };
      const f = fitting('vulkan', 32768);
      f.fitted_ngl = 20;
      f.fitted_ncmoe = 12;
      f.fits_fully = false;
      const r = synthesize(igpuHw(), moe, answers(), [f], [], undefined);
      check('L11: n-cpu-moe from fit', hasArg(r.primary, '--n-cpu-moe 12'));
      check('L11: conservative alternative',
        r.alternatives.some(a => a.llamacpp_args.includes('--cpu-moe')));
    }

    // ── lever 12: ctx rounded to fit ──────────────────────────────────
    {
      const r = synthesize(igpuHw(), denseModel(), answers(), [fitting('vulkan', 40000)], [], undefined);
      check('L12: ctx rounded down to fit', r.primary.ctx_size === 32768);

      const rmax = synthesize(igpuHw(), denseModel(), answers(), [fitting('vulkan', 200000)], [], undefined);
      check('L12: ctx capped at trained window', rmax.primary.ctx_size === 131072);
    }

    if (failures > 0) {
      throw new Error(`${failures} synthesis checks failed`);
    }
    console.log('\nAutoOpt synthesis runtime tests passed.');
  } finally {
    fs.rmSync(outputPath, { recursive: true, force: true });
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
