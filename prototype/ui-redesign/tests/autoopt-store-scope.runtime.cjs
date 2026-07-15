const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const webpack = require('webpack');

const KEY_A = 'lemonade_autoopt_runs_v2::http://server-a:1111';
const KEY_B = 'lemonade_autoopt_runs_v2::http://server-b:2222';

function installLocalStorage() {
  const store = new Map();
  global.localStorage = {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => { store.set(k, String(v)); },
    removeItem: k => { store.delete(k); },
    _dump: () => store,
  };
  return store;
}

function baseRun(id, extra) {
  return {
    id,
    model: 'org/chat-model',
    checkpoint: 'org/chat-model-GGUF:Q4_K_M',
    budget: 'standard',
    answers: { parallel: { mode: 'single' }, kv_cache_quant: 'none', ram_headroom: 'normal', allow_network: true },
    allow_unload: true,
    status: 'completed',
    created_at: '2026-07-15T10:00:00Z',
    finished_at: '2026-07-15T10:10:00Z',
    summary: 'vulkan · ctx 32768',
    stages: [],
    measurements: { fit: [], bench: [] },
    ...extra,
  };
}

const SYNTH_INPUTS = {
  hardware: { gpus: [], has_igpu: true, ram_is_vram: true, host_ram_gb: 64, installed_backends: ['vulkan'], os: 'linux' },
  facts: { architecture: 'qwen3', block_count: 36, expert_count: 0, full_attention_interval: 0, swa_layer_count: 0, n_ctx_train: 32768, kv_bytes_per_token: 90112, weights_mib: 4096, is_moe: false, is_hybrid_or_recurrent: false, has_mtp: false, base_model_repo: '', checkpoint: '', metadata_present: true },
  fits: [],
  plan: [],
  step_labels: {},
};

async function bundleStore() {
  const srcDir = path.resolve(__dirname, '../src');
  const outputPath = fs.mkdtempSync(path.join(os.tmpdir(), 'lemonade-autoopt-store-'));
  const apiStub = path.resolve(__dirname, 'stubs/autoOptApiStub.js');
  const ctrlStub = path.resolve(__dirname, 'stubs/autoOptControllerStub.js');

  const config = {
    mode: 'development',
    target: 'node',
    entry: path.resolve(srcDir, 'features/autoOpt/autoOptStore.ts'),
    output: { path: outputPath, filename: 'autoOptStore.cjs', library: { type: 'commonjs2' } },
    resolve: { extensions: ['.ts', '.tsx', '.js'] },
    module: { rules: [{ test: /\.tsx?$/, use: 'ts-loader', exclude: /node_modules/ }] },
    optimization: { minimize: false },
    plugins: [
      new webpack.NormalModuleReplacementPlugin(/^\.\.\/\.\.\/api$/, apiStub),
      new webpack.NormalModuleReplacementPlugin(/^\.\/autoOptController$/, ctrlStub),
    ],
  };

  await new Promise((resolve, reject) => {
    webpack(config, (error, stats) => {
      if (error) return reject(error);
      if (stats?.hasErrors()) return reject(new Error(stats.toString({ all: false, errors: true })));
      resolve();
    });
  });

  return { outputPath, modulePath: path.join(outputPath, 'autoOptStore.cjs') };
}

let failures = 0;
function check(name, ok) {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}`);
  if (!ok) failures += 1;
}

const tick = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  const lsStore = installLocalStorage();
  globalThis.__AO_INITIAL_BASE__ = 'http://server-a:1111';

  // Seed server A (one in-flight run to reattach + one completed) and server B (one completed).
  lsStore.set(KEY_A, JSON.stringify({ version: 2, runs: [
    baseRun('ao-a-live', { status: 'running', finished_at: undefined, summary: undefined, job_id: 'job-a-live', synth_inputs: SYNTH_INPUTS }),
    baseRun('ao-a-done'),
  ] }));
  lsStore.set(KEY_B, JSON.stringify({ version: 2, runs: [baseRun('ao-b-done')] }));

  const { outputPath, modulePath } = await bundleStore();

  try {
    const { autoOptStore } = require(modulePath);
    const ctrl = globalThis.__AO_CTRL__;
    const apiControl = globalThis.__AO_API_CONTROL__;

    // Initial scope = A: A's runs are visible and the in-flight run is reattached.
    let ids = autoOptStore.snapshot().runs.map(r => r.id).sort();
    check('scope: server A runs loaded', JSON.stringify(ids) === JSON.stringify(['ao-a-done', 'ao-a-live']));
    const aLive = ctrl.attached.find(a => a.id === 'ao-a-live');
    check('scope: A in-flight job reattached', !!aLive);
    check('scope: A poller not yet aborted', aLive && aLive.signal.aborted === false);

    // Switch A -> B.
    apiControl.setBase('http://server-b:2222');
    await tick();

    ids = autoOptStore.snapshot().runs.map(r => r.id).sort();
    check('switch: server B runs now shown', JSON.stringify(ids) === JSON.stringify(['ao-b-done']));
    check('switch: A runs no longer in memory', !ids.includes('ao-a-live') && !ids.includes('ao-a-done'));
    check('switch: A poller aborted', aLive && aLive.signal.aborted === true);
    check('switch: activeRunId reset', autoOptStore.snapshot().activeRunId === null);

    // A's aborted controller must not cross-write into B's key, nor corrupt A's key.
    const bStored = JSON.parse(lsStore.get(KEY_B)).runs.map(r => r.id).sort();
    check('no-cross-write: B key holds only B runs', JSON.stringify(bStored) === JSON.stringify(['ao-b-done']));
    const aStored = JSON.parse(lsStore.get(KEY_A)).runs.map(r => r.id).sort();
    check('no-cross-write: A key intact', JSON.stringify(aStored) === JSON.stringify(['ao-a-done', 'ao-a-live']));

    // Switching back to A restores A's runs from its key.
    apiControl.setBase('http://server-a:1111');
    await tick();
    ids = autoOptStore.snapshot().runs.map(r => r.id).sort();
    check('switch-back: A runs restored', JSON.stringify(ids) === JSON.stringify(['ao-a-done', 'ao-a-live']));

    if (failures > 0) throw new Error(`${failures} store-scope checks failed`);
    console.log('\nAutoOpt store server-scope runtime tests passed.');
  } finally {
    fs.rmSync(outputPath, { recursive: true, force: true });
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
