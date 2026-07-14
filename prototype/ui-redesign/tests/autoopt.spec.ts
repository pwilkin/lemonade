import { test, expect, Page } from '@playwright/test';

const CHAT_MODEL = 'org/chat-model';
const NPU_MODEL = 'org/npu-model';
const REGISTRY_MODEL = 'org/registry-only-model';
// Runs are scoped to the server they were measured on. The app defaults to
// http://127.0.0.1:13305 when served from the test dev server.
const RUNS_KEY = 'lemonade_autoopt_runs_v2::http://127.0.0.1:13305';
const CHAT_CHECKPOINT = 'org/chat-model-GGUF:Q4_K_M';

const MODELS = [
  {
    id: CHAT_MODEL, name: CHAT_MODEL, labels: ['llm'], recipe: 'llamacpp', downloaded: true,
    checkpoint: CHAT_CHECKPOINT, size: 4, max_context_window: 32768,
    metadata: {
      architecture: 'qwen3', block_count: 36, context_length: 32768, expert_count: 0,
      full_attention_interval: 0, swa_layer_count: 0, kv_bytes_per_token: 90112,
      base_model_repo: 'https://huggingface.co/org/base-model',
    },
  },
  { id: NPU_MODEL, name: NPU_MODEL, labels: ['llm'], recipe: 'flm', downloaded: true, max_context_window: 32768 },
  { id: REGISTRY_MODEL, name: REGISTRY_MODEL, labels: ['llm'], recipe: 'llamacpp', downloaded: false, max_context_window: 32768 },
];

const SYSTEM_INFO = {
  'OS Version': 'Linux test',
  'Physical Memory': '64.0 GB',
  devices: {
    amd_gpu: [{ available: true, name: 'Radeon 8060S', family: 'gfx1151', vram_gb: 2.0, virtual_mem_gb: 62.0 }],
    nvidia_gpu: [{ available: false, name: '' }],
  },
  recipes: {
    llamacpp: {
      default_backend: 'vulkan',
      backends: {
        vulkan: { state: 'installed' },
        rocm: { state: 'installed' },
        system: { state: 'installed' },
        cpu: { state: 'installable' },
      },
    },
  },
};

/** TTFT/TPS returned by the mock chat endpoint for the loaded config. */
function metricsForLoad(load: Record<string, unknown>): { ttftMs: number; tps: number } {
  const backend = String(load.llamacpp_backend || 'vulkan');
  const args = String(load.llamacpp_args || '');
  const ctx = Number(load.ctx_size || 0);
  const bMatch = /-b (\d+)/.exec(args);
  if (bMatch) {
    const b = Number(bMatch[1]);
    // b=2048 is the fastest prefill (lowest TTFT) and clears the 5% gate.
    return { ttftMs: b === 2048 ? 70 : (b === 8192 ? 80 : 100), tps: 40 };
  }
  const deep = ctx >= 32768;
  if (backend === 'vulkan') return { ttftMs: deep ? 300 : 90, tps: deep ? 40 : 42 };
  return { ttftMs: deep ? 380 : 120, tps: deep ? 35 : 37 };
}

interface MockCounts {
  load: Array<Record<string, unknown>>;
  unload: Array<Record<string, unknown>>;
  chat: Array<Record<string, unknown>>;
}

interface MockOptions {
  loadedModels?: Array<Record<string, unknown>>;
  loadHandler?: (body: Record<string, unknown>, counts: MockCounts) => { status?: number; json?: unknown } | null;
  hangLoad?: boolean;
}

async function mockServer(page: Page, options: MockOptions = {}): Promise<MockCounts> {
  const counts: MockCounts = { load: [], unload: [], chat: [] };
  const loaded = options.loadedModels || [];
  let lastLoad: Record<string, unknown> = {};

  await page.route('**/api/v1/health**', route => route.fulfill({
    json: { status: 'ok', version: 'test', all_models_loaded: loaded },
  }));
  await page.route('**/api/v1/models**', route => route.fulfill({ json: { data: MODELS } }));
  await page.route('**/api/v1/models/*', route => {
    const id = decodeURIComponent(route.request().url().split('/').pop() || '').split('?')[0];
    const model = MODELS.find(m => m.id === id);
    if (!model) return route.fulfill({ status: 404, json: { error: 'unknown model' } });
    return route.fulfill({ json: model });
  });
  await page.route('**/api/v1/system-info**', route => route.fulfill({ json: SYSTEM_INFO }));
  await page.route('**/api/v1/system-stats', route => route.fulfill({
    json: { cpu_percent: 10, memory_gb: 40, gpu_percent: 80, vram_gb: 20, npu_percent: 0 },
  }));
  await page.route('**/api/v1/unload', route => {
    counts.unload.push(route.request().postDataJSON() || {});
    return route.fulfill({ json: { status: 'ok' } });
  });
  await page.route('**/api/v1/load', async route => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    counts.load.push(body);
    lastLoad = body;
    if (options.hangLoad) return; // never fulfil — cancel aborts it
    const override = options.loadHandler ? options.loadHandler(body, counts) : null;
    if (override) return route.fulfill({ status: override.status ?? 200, json: override.json ?? { status: 'ok' } });
    return route.fulfill({ json: { status: 'ok' } });
  });
  await page.route('**/api/v1/chat/completions', route => {
    counts.chat.push(route.request().postDataJSON() || {});
    const m = metricsForLoad(lastLoad);
    return route.fulfill({
      json: {
        choices: [{ message: { role: 'assistant', content: 'ok' } }],
        usage: {
          prompt_tokens: 400, completion_tokens: 64,
          prefill_duration_ttft: m.ttftMs / 1000, decoding_speed_tps: m.tps,
        },
      },
    });
  });

  await page.route('https://huggingface.co/org/base-model/resolve/main/generation_config.json',
    route => route.fulfill({ json: { temperature: 0.7, top_p: 0.8, top_k: 20 } }));
  await page.route('https://huggingface.co/org/base-model/resolve/main/config.json',
    route => route.fulfill({ status: 404, body: 'Not Found' }));
  await page.route('https://huggingface.co/api/models/**', route => route.fulfill({
    json: { cardData: { base_model: 'org/base-model' } },
  }));

  return counts;
}

// A completed run in the current TTFT/TPS measurement shape.
const COMPLETED_RUN = {
  id: 'ao-20260712-101500-cafe',
  model: CHAT_MODEL,
  checkpoint: CHAT_CHECKPOINT,
  budget: 'standard',
  answers: { parallel: { mode: 'single' }, kv_cache_quant: 'q8_0', ram_headroom: 'normal', allow_network: true },
  allow_unload: true,
  status: 'completed',
  created_at: '2026-07-12T10:00:00Z',
  finished_at: '2026-07-12T10:15:00Z',
  summary: 'vulkan · ctx 32768 · -ctk q8_0 -ctv q8_0 --spec-default -b 2048 -ub 2048',
  lemonade_version: 'test',
  stages: [
    { name: 'snapshot', status: 'completed', duration_ms: 12 },
    { name: 'model_facts', status: 'completed', duration_ms: 8 },
    { name: 'hf_metadata', status: 'completed', duration_ms: 420 },
    { name: 'fit_estimate', status: 'completed', duration_ms: 3 },
    { name: 'bench_matrix', status: 'completed', duration_ms: 431000 },
    { name: 'synthesize', status: 'completed', duration_ms: 2 },
    { name: 'load_test', status: 'completed', duration_ms: 2100 },
  ],
  measurements: {
    fit: [
      { backend: 'vulkan', fits_fully: true, fitted_ctx: 0, fitted_ngl: -1, fitted_ncmoe: 0, weights_mib: 4096, kv_mib: 2816, compute_mib: 512, total_mib: 7424, available_mib: 58982, degraded: false, ok: true },
    ],
    bench: [
      { backend: 'vulkan', label: 'vulkan · d0', ctx_size: 2048, llamacpp_args: '-ctk q8_0 -ctv q8_0', params: { d: 0 }, ttft_ms: 90, tps: 42, vram_gb: 20, ok: true },
      { backend: 'rocm', label: 'rocm · d0', ctx_size: 2048, llamacpp_args: '-ctk q8_0 -ctv q8_0', params: { d: 0 }, ttft_ms: 120, tps: 37, vram_gb: 21, ok: true },
      { backend: 'vulkan', label: 'vulkan · b512', ctx_size: 2048, llamacpp_args: '-b 512 -ub 512', params: { ladder: true, b: 512, ub: 512, d: 0 }, ttft_ms: 100, tps: 40, vram_gb: 20, ok: true },
      { backend: 'vulkan', label: 'vulkan · b2048', ctx_size: 2048, llamacpp_args: '-b 2048 -ub 2048', params: { ladder: true, b: 2048, ub: 2048, d: 0 }, ttft_ms: 70, tps: 40, vram_gb: 20, ok: true },
    ],
  },
  result: {
    primary: {
      label: 'Recommended',
      llamacpp_backend: 'vulkan',
      ctx_size: 32768,
      llamacpp_args: '-ctk q8_0 -ctv q8_0 --spec-default -b 2048 -ub 2048',
      rationale: ['vulkan chosen over rocm: best measured throughput/latency balance', 'Load test passed'],
      expected: { ttft_ms: 90, tps: 42, vram_gb: 20 },
    },
    alternatives: [
      {
        label: 'Maximum quality',
        llamacpp_backend: 'vulkan',
        ctx_size: 16384,
        llamacpp_args: '--spec-default -b 2048 -ub 2048',
        rationale: ['Unquantized f16 KV cache'],
        tradeoff: 'smaller context window',
      },
    ],
    sampling_defaults: { temperature: 0.7, top_p: 0.9, top_k: 40, min_p: 0.05, source: 'hf:org/base-model/generation_config.json' },
  },
};

async function seedRuns(page: Page, runs: Array<Record<string, unknown>>, key = RUNS_KEY) {
  await page.addInitScript(([k, payload]) => {
    localStorage.setItem(k as string, JSON.stringify({ version: 2, runs: payload }));
  }, [key, runs] as const);
}

async function openPresets(page: Page) {
  await page.goto('/');
  await page.waitForSelector('.titlebar__nav');
  await page.locator('.titlebar__nav').getByText('Presets').click();
  await page.waitForSelector('.recipes');
}

async function openWizard(page: Page) {
  await openPresets(page);
  await page.locator('[data-autoopt-run-optimizer]').click();
  await page.waitForSelector('[data-autoopt-wizard]');
}

async function walkToBudget(page: Page, model = CHAT_MODEL) {
  await page.locator('[data-autoopt-model-select]').selectOption(model);
  await page.locator('[data-autoopt-next]').click(); // parallel
  await page.locator('[data-autoopt-next]').click(); // kv
  await page.locator('[data-autoopt-next]').click(); // ram
  await page.locator('[data-autoopt-next]').click(); // budget
  await expect(page.locator('[data-autoopt-step="budget"]')).toBeVisible();
}

test.describe('AutoOpt wizard + controller (generic-endpoint bench)', () => {

  test('full Benchmark flow — coordinated loads/chats, synthesized result, load test', async ({ page }) => {
    const counts = await mockServer(page, {
      loadedModels: [{ model_name: CHAT_MODEL, type: 'llm', recipe: 'llamacpp', device: 'gpu', checkpoint: '', backend_url: '', pid: 1, last_use: Date.now() }],
    });

    await openWizard(page);
    await expect(page.locator('[data-autoopt-model-select]')).toHaveValue(CHAT_MODEL, { timeout: 10000 });
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-step="parallel"]')).toBeVisible();
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-option="kv:q8_0"]')).toHaveAttribute('aria-pressed', 'true');
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-ram-suggestion]')).toContainText('Suggested for this machine (64 GB RAM)');
    await page.locator('[data-autoopt-option="ram:minimal"]').click();
    await page.locator('[data-autoopt-next]').click();

    // No vision step (dropped in this PR).
    await expect(page.locator('[data-autoopt-step="budget"]')).toBeVisible();
    await expect(page.locator('[data-autoopt-step="vision"]')).toHaveCount(0);
    await page.locator('[data-autoopt-option="budget:standard"]').click();

    // Consent gate for the Benchmark tier.
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-start]')).toBeDisabled();
    await page.locator('[data-autoopt-back]').click();
    await page.locator('[data-autoopt-consent]').check();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-start]').click();

    await expect(page.locator('[data-autoopt-step="running"]')).toBeVisible();
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--completed')).toHaveCount(7, { timeout: 20000 });
    await page.locator('[data-autoopt-wizard] .slideover__close').click();

    const run = page.locator('[data-autoopt-run]');
    await expect(run.locator('.autoopt-status-chip--completed')).toBeVisible();
    await expect(page.locator('[data-autoopt-announcement]')).toContainText(`AutoOpt run for ${CHAT_MODEL} completed.`);

    // The controller drove generic endpoints: it unloaded, loaded each config
    // with an exact body, and timed a completion.
    expect(counts.chat.length).toBeGreaterThan(0);
    const benchLoads = counts.load.filter(b => b.save_options === false && b.merge_args === false);
    expect(benchLoads.length).toBeGreaterThan(0);
    // Backend duel loaded both backends...
    const backends = new Set(benchLoads.map(b => b.llamacpp_backend));
    expect(backends.has('vulkan')).toBe(true);
    expect(backends.has('rocm')).toBe(true);
    // ...at both depths (ctx 2048 for depth 0, 32768 for the deep point)...
    const vulkanCtx = new Set(benchLoads.filter(b => b.llamacpp_backend === 'vulkan').map(b => b.ctx_size));
    expect(vulkanCtx.has(32768)).toBe(true);
    // ...and the batch ladder loaded the {512,2048,8192} rungs.
    const ladderRungs = new Set(benchLoads
      .map(b => /-b (\d+)/.exec(String(b.llamacpp_args || ''))?.[1])
      .filter(Boolean));
    expect(ladderRungs.has('512')).toBe(true);
    expect(ladderRungs.has('2048')).toBe(true);
    expect(ladderRungs.has('8192')).toBe(true);

    // Synthesized recommendation reflects the measurements: vulkan won the duel,
    // b2048 won the ladder (lowest TTFT).
    await page.locator('[data-autoopt-inspect]').click();
    await page.waitForSelector('[data-autoopt-detail]');
    const args = page.locator('[data-autoopt-rec-args]');
    await expect(args).toContainText('-ctk q8_0 -ctv q8_0');
    await expect(args).toContainText('--cache-ram 2048 -ctxcp 8');
    await expect(args).toContainText('-b 2048 -ub 2048');
    await expect(page.locator('.autoopt-rec-card__chips')).toContainText('vulkan');
    // Tables render the TTFT/TPS shape.
    await expect(page.locator('[data-autoopt-bench-duel] thead')).toContainText('TTFT ms');
    await expect(page.locator('[data-autoopt-bench-duel] thead')).toContainText('tok/s');
    await expect(page.locator('[data-autoopt-bench-ladder] thead')).toContainText('TTFT ms');
    await expect(page.locator('[data-autoopt-bench-ladder] thead')).not.toContainText('tok/s');
    // Load test recorded a pass.
    await expect(page.locator('.autoopt-rec-card__rationale')).toContainText('Load test passed');
  });

  test('Fast Scan runs without loads, unloads, or benchmarks while models are loaded', async ({ page }) => {
    const counts = await mockServer(page, {
      loadedModels: [{ model_name: CHAT_MODEL, type: 'llm', recipe: 'llamacpp', device: 'gpu', checkpoint: '', backend_url: '', pid: 1, last_use: Date.now() }],
    });

    await openWizard(page);
    await expect(page.locator('[data-autoopt-model-select]')).toHaveValue(CHAT_MODEL, { timeout: 10000 });
    await walkToBudget(page);
    await page.locator('[data-autoopt-option="budget:quick"]').click();
    await expect(page.locator('[data-autoopt-consent]')).toHaveCount(0);
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-step="review"]')).toContainText('Not needed for Fast Scan');
    await page.locator('[data-autoopt-start]').click();

    await expect(page.locator('[data-autoopt-step="running"]')).toBeVisible();
    // bench_matrix + load_test are skipped for Fast Scan.
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--completed')).toHaveCount(5, { timeout: 15000 });
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--skipped')).toHaveCount(2);

    expect(counts.load).toHaveLength(0);
    expect(counts.unload).toHaveLength(0);
    expect(counts.chat).toHaveLength(0);
  });

  test('a late system-info reply must not clobber an explicit RAM-headroom choice', async ({ page }) => {
    await mockServer(page);
    await page.unroute('**/api/v1/system-info**');
    await page.route('**/api/v1/system-info**', async route => {
      await new Promise(resolve => setTimeout(resolve, 2500));
      await route.fulfill({ json: SYSTEM_INFO });
    });

    await openWizard(page);
    await page.locator('[data-autoopt-model-select]').selectOption(CHAT_MODEL);
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-step="ram"]')).toBeVisible();
    await page.locator('[data-autoopt-option="ram:minimal"]').click();
    await page.waitForTimeout(3000);
    await expect(page.locator('[data-autoopt-option="ram:minimal"]')).toHaveAttribute('aria-pressed', 'true');

    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-option="budget:quick"]').click();
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-step="review"]')).toContainText('Minimal');
    await page.locator('[data-autoopt-start]').click();
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--completed')).toHaveCount(5, { timeout: 15000 });

    await page.locator('[data-autoopt-wizard] .slideover__close').click();
    await page.locator('[data-autoopt-inspect]').click();
    await page.waitForSelector('[data-autoopt-detail]');
    await expect(page.locator('[data-autoopt-rec-args]')).toContainText('--cache-ram 2048 -ctxcp 8');
  });

  async function runFastScan(page: Page) {
    await openWizard(page);
    await walkToBudget(page);
    await page.locator('[data-autoopt-option="budget:quick"]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-start]').click();
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--skipped')).toHaveCount(2, { timeout: 15000 });
  }

  test('a model without published sampling defaults completes hf_metadata with a neutral note', async ({ page }) => {
    await mockServer(page);
    await page.unroute('https://huggingface.co/org/base-model/resolve/main/generation_config.json');
    await page.route('https://huggingface.co/org/base-model/resolve/main/generation_config.json',
      route => route.fulfill({ status: 404, body: 'Not Found' }));

    await runFastScan(page);
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--failed')).toHaveCount(0);
    await page.locator('[data-autoopt-wizard] .slideover__close').click();
    await page.locator('[data-autoopt-inspect]').click();
    await page.waitForSelector('[data-autoopt-detail]');
    await expect(page.locator('.autoopt-stage--failed')).toHaveCount(0);
    await expect(page.locator('.autoopt-stage__note')).toContainText('no sampling defaults published');
  });

  test('hf_metadata falls back to generation defaults inside config.json', async ({ page }) => {
    await mockServer(page);
    await page.unroute('https://huggingface.co/org/base-model/resolve/main/generation_config.json');
    await page.route('https://huggingface.co/org/base-model/resolve/main/generation_config.json',
      route => route.fulfill({ status: 404, body: 'Not Found' }));
    await page.unroute('https://huggingface.co/org/base-model/resolve/main/config.json');
    await page.route('https://huggingface.co/org/base-model/resolve/main/config.json',
      route => route.fulfill({ json: { architectures: ['Qwen3ForCausalLM'], generation_config: { temperature: 0.6, top_k: 20 } } }));

    await runFastScan(page);
    await page.locator('[data-autoopt-wizard] .slideover__close').click();
    await page.locator('[data-autoopt-inspect]').click();
    await page.waitForSelector('[data-autoopt-detail]');
    await expect(page.locator('.autoopt-rec-card__chips')).toContainText('temp 0.6');
  });

  test('model picker only offers downloaded llama.cpp models and explains the scope', async ({ page }) => {
    await mockServer(page);
    await openWizard(page);

    await expect(page.locator('[data-autoopt-model-note]')).toContainText('AutoOpt currently supports downloaded llama.cpp models');
    const select = page.locator('[data-autoopt-model-select]');
    await expect(select.locator(`option[value="${CHAT_MODEL}"]`)).toHaveCount(1);
    await expect(select.locator(`option[value="${NPU_MODEL}"]`)).toHaveCount(0);
    await expect(select.locator(`option[value="${REGISTRY_MODEL}"]`)).toHaveCount(0);
  });

  test('cancel mid-bench aborts the in-flight load and marks the run cancelled', async ({ page }) => {
    await mockServer(page, { hangLoad: true });

    await openWizard(page);
    await walkToBudget(page);
    await page.locator('[data-autoopt-option="budget:standard"]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-start]').click();

    const loadRequest = page.waitForRequest('**/api/v1/load', { timeout: 15000 });
    const loadAborted = page.waitForEvent('requestfailed', {
      predicate: request => request.url().endsWith('/api/v1/load'),
      timeout: 15000,
    });
    await loadRequest;
    await page.locator('[data-autoopt-cancel-run]').click();
    await loadAborted;

    await page.locator('[data-autoopt-wizard] .slideover__close').click();
    await expect(page.locator('[data-autoopt-run] .autoopt-status-chip--cancelled')).toBeVisible({ timeout: 10000 });
  });

  test('test-by-failure: a load that fails then succeeds after backoff still completes', async ({ page }) => {
    let firstLoadTestSeen = false;
    await mockServer(page, {
      loadHandler: (body) => {
        // The load-test config is the only load carrying the full recommended
        // args (--spec-default). Fail its first attempt to force a ctx backoff;
        // the retried, smaller-context load then succeeds.
        const args = String(body.llamacpp_args || '');
        if (args.includes('--spec-default') && body.ctx_size === 32768 && !firstLoadTestSeen) {
          firstLoadTestSeen = true;
          return { status: 500, json: { error: { message: 'CUDA out of memory' } } };
        }
        return null;
      },
    });

    await openWizard(page);
    await walkToBudget(page);
    await page.locator('[data-autoopt-option="budget:standard"]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-start]').click();

    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--completed')).toHaveCount(7, { timeout: 20000 });
    await page.locator('[data-autoopt-wizard] .slideover__close').click();
    await expect(page.locator('[data-autoopt-run] .autoopt-status-chip--completed')).toBeVisible();
    await page.locator('[data-autoopt-inspect]').click();
    await page.waitForSelector('[data-autoopt-detail]');
    await expect(page.locator('.autoopt-rec-card__rationale')).toContainText('backed off');
  });

  test('the rail fires no server calls before a run is started', async ({ page }) => {
    const counts = await mockServer(page);
    await openPresets(page);
    await expect(page.locator('[data-autoopt-run-optimizer]')).toBeEnabled();
    await page.waitForTimeout(500);
    expect(counts.load).toHaveLength(0);
    expect(counts.unload).toHaveLength(0);
    expect(counts.chat).toHaveLength(0);
  });
});

test.describe('AutoOpt rail persistence + server scoping', () => {

  test('a run interrupted by page reload is failed as interrupted', async ({ page }) => {
    await mockServer(page);
    await seedRuns(page, [{ ...COMPLETED_RUN, id: 'ao-20260713-090000-dead', status: 'running', finished_at: undefined, result: undefined, summary: undefined }]);

    await openPresets(page);
    const run = page.locator('[data-autoopt-run="ao-20260713-090000-dead"]');
    await expect(run).toBeVisible();
    await expect(run.locator('.autoopt-status-chip--failed')).toContainText('interrupted');
  });

  test('completed runs survive a reload with their result intact', async ({ page }) => {
    await mockServer(page);
    await seedRuns(page, [COMPLETED_RUN]);

    await openPresets(page);
    await expect(page.locator(`[data-autoopt-run="${COMPLETED_RUN.id}"] .autoopt-status-chip--completed`)).toBeVisible();
    await page.locator(`[data-autoopt-inspect="${COMPLETED_RUN.id}"]`).click();
    await page.waitForSelector('[data-autoopt-detail]');
    await expect(page.locator('[data-autoopt-rec-args]')).toContainText('-b 2048 -ub 2048');
    await expect(page.locator('[data-autoopt-bench-duel] tbody tr')).toHaveCount(2);
    await expect(page.locator('[data-autoopt-bench-ladder] tbody tr')).toHaveCount(2);
  });

  test('runs measured on a different server do not appear here', async ({ page }) => {
    await mockServer(page);
    await seedRuns(page, [{ ...COMPLETED_RUN, id: 'ao-other-server' }], 'lemonade_autoopt_runs_v2::http://other-server:9999');

    await openPresets(page);
    await expect(page.locator('[data-autoopt-empty]')).toBeVisible();
    await expect(page.locator('[data-autoopt-run="ao-other-server"]')).toHaveCount(0);
  });

  test('delete removes a run permanently; a failed persist rolls the row and selection back', async ({ page }) => {
    const otherRun = { ...COMPLETED_RUN, id: 'ao-20260713-110000-beef', created_at: '2026-07-13T11:00:00Z' };
    await mockServer(page);
    await seedRuns(page, [otherRun, COMPLETED_RUN]);

    await openPresets(page);
    const run = page.locator(`[data-autoopt-run="${COMPLETED_RUN.id}"]`);
    await expect(run).toBeVisible();
    await run.locator('.auto-run-card__main').click();
    await expect(run.locator('.auto-run-card__main')).toHaveAttribute('aria-pressed', 'true');
    await page.evaluate((key) => {
      const original = localStorage.setItem.bind(localStorage);
      let armed = true;
      localStorage.setItem = (k: string, v: string) => {
        if (k === key && armed) { armed = false; throw new Error('storage quota exceeded'); }
        return original(k, v);
      };
    }, RUNS_KEY);

    await page.locator(`[data-autoopt-delete="${COMPLETED_RUN.id}"]`).click();
    await expect(run).toBeVisible();
    await expect(page.locator('[data-autoopt-rail-error]')).toContainText('storage quota exceeded');
    await expect(run.locator('.auto-run-card__main')).toHaveAttribute('aria-pressed', 'true');

    await page.locator(`[data-autoopt-delete="${COMPLETED_RUN.id}"]`).click();
    await expect(run).toHaveCount(0);
    const persisted = await page.evaluate((key) => {
      const parsed = JSON.parse(localStorage.getItem(key) || '{}');
      return Array.isArray(parsed.runs) ? parsed.runs.map((r: any) => r.id) : [];
    }, RUNS_KEY);
    expect(persisted).toEqual([otherRun.id]);
  });
});

test.describe('AutoOpt run detail actions', () => {

  async function openCompletedRunDetail(page: Page, options: MockOptions = {}): Promise<MockCounts> {
    const counts = await mockServer(page, options);
    await seedRuns(page, [COMPLETED_RUN]);
    await openPresets(page);
    await expect(page.locator(`[data-autoopt-run="${COMPLETED_RUN.id}"]`)).toBeVisible();
    await page.locator(`[data-autoopt-inspect="${COMPLETED_RUN.id}"]`).click();
    await page.waitForSelector('[data-autoopt-detail]');
    await expect(page.locator('[data-autoopt-recommendation]')).toBeVisible();
    return counts;
  }

  test('"Create preset" writes an optimized preset and model tuning without loading', async ({ page }) => {
    const counts = await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-create-preset]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('selected it for future loads');
    expect(counts.load).toHaveLength(0);

    const stored = await page.evaluate((model) => {
      let preset: any = null;
      let tuning: any = null;
      for (const key of Object.keys(localStorage)) {
        if (key.includes('user_presets')) {
          const presets = JSON.parse(localStorage.getItem(key) || '[]');
          preset = presets.find((p: any) => p.auto_opt_run_id) || preset;
        }
        if (key.includes('model_tunings')) {
          const tunings = JSON.parse(localStorage.getItem(key) || '{}');
          for (const [tuningKey, value] of Object.entries(tunings)) {
            if (tuningKey.startsWith(`${model}@@`) && (value as any).source === 'optimized') tuning = value;
          }
        }
      }
      return { preset, tuning };
    }, CHAT_MODEL);

    expect(stored.preset?.auto_opt_run_id).toBe(COMPLETED_RUN.id);
    expect(stored.preset.name).toContain('AutoOpt');
    expect(stored.tuning?.source).toBe('optimized');
    expect(stored.tuning.recipe_options.llamacpp_args).toBe('-ctk q8_0 -ctv q8_0 --spec-default -b 2048 -ub 2048');
    expect(stored.tuning.sampling.min_p).toBe(0.05);
  });

  test('"Create preset & apply now" also loads the model', async ({ page }) => {
    const counts = await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-create-apply]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('applied it to');
    expect(counts.load).toHaveLength(1);
    expect(counts.load[0].model_name).toBe(CHAT_MODEL);
  });

  test('"Create preset & apply now" keeps the created preset when the load fails', async ({ page }) => {
    await openCompletedRunDetail(page);
    await page.unroute('**/api/v1/load');
    await page.route('**/api/v1/load', route => route.fulfill({ status: 500, json: { error: { message: 'llama-server failed to start' } } }));

    await page.locator('[data-autoopt-create-apply]').click();
    const error = page.locator('[data-autoopt-detail] .preset-error');
    await expect(error).toContainText('was created and selected, but loading');
    await expect(error).toContainText('llama-server failed to start');

    const stored = await page.evaluate(() => {
      for (const key of Object.keys(localStorage)) {
        if (!key.includes('user_presets')) continue;
        const presets = JSON.parse(localStorage.getItem(key) || '[]');
        const preset = presets.find((p: any) => p.auto_opt_run_id);
        if (preset) return preset;
      }
      return null;
    });
    expect(stored?.name).toContain('AutoOpt');
  });

  test('"Try now without saving" loads a not-loaded model with the exact recommended options', async ({ page }) => {
    const counts = await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-try-now]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('nothing saved');

    expect(counts.load).toHaveLength(1);
    const body = counts.load[0];
    expect(body.model_name).toBe(CHAT_MODEL);
    expect(body.ctx_size).toBe(32768);
    expect(body.llamacpp_args).toBe('-ctk q8_0 -ctv q8_0 --spec-default -b 2048 -ub 2048');
    expect(body.llamacpp_backend).toBe('vulkan');
    expect(body.save_options).toBe(false);
    // No preset was saved.
    const saved = await page.evaluate(() => {
      for (const key of Object.keys(localStorage)) {
        if (key.includes('user_presets')) {
          const presets = JSON.parse(localStorage.getItem(key) || '[]');
          if (presets.some((p: any) => p.auto_opt_run_id)) return 'preset-created';
        }
      }
      return 'nothing-saved';
    });
    expect(saved).toBe('nothing-saved');
  });

  test('"Try now without saving" RELOADS a model that is already loaded (review #2)', async ({ page }) => {
    const counts = await openCompletedRunDetail(page, {
      loadedModels: [{ model_name: CHAT_MODEL, type: 'llm', recipe: 'llamacpp', device: 'gpu', checkpoint: '', backend_url: '', pid: 1, last_use: Date.now() }],
    });
    await page.locator('[data-autoopt-try-now]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('nothing saved');

    // reloadModel = unload(model) then load(model, temp options).
    expect(counts.unload.some(b => b.model_name === CHAT_MODEL)).toBe(true);
    expect(counts.load).toHaveLength(1);
    expect(counts.load[0].save_options).toBe(false);
    expect(counts.load[0].llamacpp_args).toBe('-ctk q8_0 -ctv q8_0 --spec-default -b 2048 -ub 2048');
  });

  test('"Use this instead" swaps the CTA target to the alternative', async ({ page }) => {
    const counts = await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-use-alternative="Maximum quality"]').click();
    await expect(page.locator('[data-autoopt-rec-args]')).toContainText('--spec-default -b 2048 -ub 2048');
    await page.locator('[data-autoopt-try-now]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('nothing saved');
    expect(counts.load[0].ctx_size).toBe(16384);
  });

  test('a run whose checkpoint no longer matches the model is refused on apply (review #5)', async ({ page }) => {
    await mockServer(page);
    await seedRuns(page, [{ ...COMPLETED_RUN, checkpoint: 'org/chat-model-GGUF:OLD_QUANT' }]);
    await openPresets(page);
    await page.locator(`[data-autoopt-inspect="${COMPLETED_RUN.id}"]`).click();
    await page.waitForSelector('[data-autoopt-detail]');

    await page.locator('[data-autoopt-create-preset]').click();
    await expect(page.locator('[data-autoopt-detail] .preset-error')).toContainText('different build');
    const saved = await page.evaluate(() => {
      for (const key of Object.keys(localStorage)) {
        if (key.includes('user_presets')) {
          const presets = JSON.parse(localStorage.getItem(key) || '[]');
          if (presets.some((p: any) => p.auto_opt_run_id)) return true;
        }
      }
      return false;
    });
    expect(saved).toBe(false);
  });

  test('a created preset is discoverable: sorted first, flashed, and named after its model', async ({ page }) => {
    await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-create-preset]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('Created preset');
    await page.locator('[data-autoopt-detail] .slideover__close').click();

    const zoneTitles = await page.locator('.recipes__body .zone__title').allTextContents();
    expect(zoneTitles.indexOf('Your presets')).toBeLessThan(zoneTitles.indexOf('Bundled starters'));

    const card = page.locator('[data-recipe-grid="yours"] .recipe-card').first();
    await expect(card).toHaveClass(/recipe-card--flash/);
    await expect(card.locator('[data-preset-linked-models]')).toContainText(`Optimized for ${CHAT_MODEL}`);
  });

  test('preset editor links back to the producing run via the AutoOpt chip', async ({ page }) => {
    await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-create-preset]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('Created preset');
    await page.locator('[data-autoopt-detail] .slideover__close').click();

    const yourCards = page.locator('[data-recipe-grid="yours"] .recipe-card');
    await yourCards.first().locator('.recipe-card__overlay-btn').click();
    await page.waitForSelector('.slideover.is-open');
    await expect(page.locator('[data-preset-editor-linked]')).toContainText(`Optimized for ${CHAT_MODEL}`);
    await page.locator('[data-preset-autoopt-chip]').click();
    await expect(page.locator('[data-autoopt-detail]')).toBeVisible();
  });
});
