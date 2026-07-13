import { test, expect, Page, Route } from '@playwright/test';

const CHAT_MODEL = 'org/chat-model';
const VISION_MODEL = 'org/vision-model';
const NPU_MODEL = 'org/npu-model';
const REGISTRY_MODEL = 'org/registry-only-model';
const RUNS_KEY = 'lemonade_autoopt_runs_v1';

const MODELS = [
  {
    id: CHAT_MODEL, name: CHAT_MODEL, labels: ['llm'], recipe: 'llamacpp', downloaded: true,
    checkpoint: 'org/chat-model-GGUF:Q4_K_M', max_context_window: 32768,
    metadata: {
      architecture: 'qwen3', block_count: 36, context_length: 32768, expert_count: 0,
      full_attention_interval: 0, swa_layer_count: 0, kv_bytes_per_token: 90112,
      base_model_repo: 'https://huggingface.co/org/base-model',
    },
  },
  {
    id: VISION_MODEL, name: VISION_MODEL, labels: ['llm', 'vision'], recipe: 'llamacpp', downloaded: true,
    checkpoint: 'org/vision-model-GGUF:Q4_K_M', max_context_window: 32768,
    metadata: {
      architecture: 'qwen3', block_count: 36, context_length: 32768, expert_count: 0,
      full_attention_interval: 0, swa_layer_count: 0, kv_bytes_per_token: 90112,
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

function fitEstimate(backend: string, extraArgs: string) {
  return {
    backend,
    fit_target_mib: 1024,
    extra_args: extraArgs,
    fitted_args: '-c 32768 -ngl -1',
    fitted_ctx: 32768,
    fitted_ngl: -1,
    fitted_ncmoe: 0,
    devices: [{ device: 'Vulkan0', model_mib: 4200, ctx_mib: 800, compute_mib: 400 }],
    fits_fully: true,
    ok: true,
    error: '',
  };
}

function benchSse(points: Array<Record<string, unknown>>): string {
  return 'event: progress\ndata: {"detail":"llama-bench warming up"}\n\n'
    + `event: complete\ndata: ${JSON.stringify({ points })}\n\n`;
}

function benchPointsFor(body: Record<string, unknown>): Array<Record<string, unknown>> {
  const backend = String(body.backend);
  if (body.b !== undefined) {
    const b = Number(body.b);
    return [{
      backend,
      params: { d: 0, b, ub: Number(body.ub) },
      pp_avg_ts: b === 2048 ? 950 : (b === 8192 ? 900 : 600),
      tg_avg_ts: 40,
      n_depth: 0,
      ok: true,
      error: '',
    }];
  }
  return [{
    backend,
    params: { d: 0 },
    pp_avg_ts: backend === 'vulkan' ? 1020.5 : 850.1,
    tg_avg_ts: backend === 'vulkan' ? 31.9 : 28.4,
    n_depth: 0,
    ok: true,
    error: '',
  }];
}

interface MockCounts {
  fit: Array<Record<string, unknown>>;
  bench: Array<Record<string, unknown>>;
  unload: number;
  load: Array<Record<string, unknown>>;
}

interface MockOptions {
  loadedModels?: Array<Record<string, unknown>>;
  benchHandler?: (route: Route, body: Record<string, unknown>, counts: MockCounts) => Promise<void>;
  fitHandler?: (route: Route, body: Record<string, unknown>, counts: MockCounts) => Promise<void>;
}

async function mockServer(page: Page, options: MockOptions = {}): Promise<MockCounts> {
  const counts: MockCounts = { fit: [], bench: [], unload: 0, load: [] };
  const loaded = options.loadedModels || [];

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
  await page.route('**/api/v1/unload', route => {
    counts.unload += 1;
    return route.fulfill({ json: { status: 'ok' } });
  });
  await page.route('**/api/v1/load', route => {
    counts.load.push(route.request().postDataJSON());
    return route.fulfill({ json: { status: 'ok' } });
  });

  await page.route('**/api/v1/backends/llamacpp/fit-params', async route => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    if (!body?.model || !body?.backend) {
      return route.fulfill({ status: 400, json: { error: "'model' and 'backend' are required" } });
    }
    counts.fit.push(body);
    if (options.fitHandler) return options.fitHandler(route, body, counts);
    const extraArgs = Array.isArray(body.args) ? (body.args as string[]).join(' ') : String(body.args || '');
    return route.fulfill({ json: fitEstimate(String(body.backend), extraArgs) });
  });
  await page.route('**/api/v1/backends/llamacpp/bench', async route => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    counts.bench.push(body);
    if (options.benchHandler) return options.benchHandler(route, body, counts);
    return route.fulfill({ contentType: 'text/event-stream', body: benchSse(benchPointsFor(body)) });
  });

  await page.route('https://huggingface.co/org/base-model/resolve/main/generation_config.json',
    route => route.fulfill({ json: { temperature: 0.7, top_p: 0.8, top_k: 20 } }));
  await page.route('https://huggingface.co/api/models/**', route => route.fulfill({
    json: { cardData: { base_model: 'org/base-model' } },
  }));

  return counts;
}

const COMPLETED_RUN = {
  id: 'ao-20260712-101500-cafe',
  model: CHAT_MODEL,
  checkpoint: 'org/chat-model-GGUF:Q4_K_M',
  budget: 'standard',
  answers: { parallel: { mode: 'single' }, kv_cache_quant: 'q8_0', ram_headroom: 'normal', allow_network: true },
  allow_unload: true,
  status: 'completed',
  created_at: '2026-07-12T10:00:00Z',
  finished_at: '2026-07-12T10:15:00Z',
  summary: 'vulkan · ctx 8192 · -b 512 -ub 256 -ctk q8_0 -ctv q8_0',
  lemonade_version: 'test',
  stages: [
    { name: 'snapshot', status: 'completed', duration_ms: 12 },
    { name: 'model_facts', status: 'completed', duration_ms: 8 },
    { name: 'hf_metadata', status: 'completed', duration_ms: 420 },
    { name: 'fit_probes', status: 'completed', duration_ms: 3200 },
    { name: 'bench_matrix', status: 'completed', duration_ms: 431000 },
    { name: 'load_validation', status: 'completed', duration_ms: 2100 },
    { name: 'synthesize', status: 'completed', duration_ms: 2 },
  ],
  measurements: {
    fit: [fitEstimate('vulkan', ''), fitEstimate('rocm', '')],
    bench: [
      { backend: 'vulkan', params: { d: 0 }, pp_avg_ts: 1020.5, tg_avg_ts: 31.9, n_depth: 0, ok: true, error: '' },
      { backend: 'rocm', params: { d: 0 }, pp_avg_ts: 0, tg_avg_ts: 0, n_depth: 0, ok: false, error: 'backend crashed' },
      { backend: 'vulkan', params: { d: 0, b: 512, ub: 256, ladder: true }, pp_avg_ts: 950.2, tg_avg_ts: 32.1, n_depth: 0, ok: true, error: '' },
      { backend: 'vulkan', params: { d: 0, b: 1024, ub: 512, ladder: true }, pp_avg_ts: 1020.5, tg_avg_ts: 31.9, n_depth: 0, ok: true, error: '' },
    ],
  },
  result: {
    primary: {
      label: 'Recommended',
      llamacpp_backend: 'vulkan',
      ctx_size: 8192,
      mmproj_enabled: false,
      llamacpp_args: '-b 512 -ub 256 -ctk q8_0 -ctv q8_0',
      rationale: ['Fastest prompt processing on this GPU', 'Fits with normal RAM headroom'],
      expected: { pp_ts: 1020.5, tg_ts: 31.9, vram_mib: 5400 },
    },
    alternatives: [
      {
        label: 'CPU fallback',
        llamacpp_backend: 'cpu',
        ctx_size: 16384,
        mmproj_enabled: true,
        llamacpp_args: '-b 256 -ub 128',
        rationale: ['Largest usable context'],
        tradeoff: 'Much slower generation',
        expected: { pp_ts: 120.0, tg_ts: 9.5, vram_mib: 0 },
      },
    ],
    sampling_defaults: { temperature: 0.7, top_p: 0.9, top_k: 40, min_p: 0.05, source: 'hf:org/base-model/generation_config.json' },
  },
};

async function seedRuns(page: Page, runs: Array<Record<string, unknown>>) {
  await page.addInitScript(([key, payload]) => {
    localStorage.setItem(key as string, JSON.stringify({ version: 1, runs: payload }));
  }, [RUNS_KEY, runs] as const);
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

test.describe('AutoOpt wizard + controller', () => {

  test('full Benchmark flow — steps, consent gate, controller queries, synthesized result', async ({ page }) => {
    const counts = await mockServer(page, {
      loadedModels: [{ model_name: CHAT_MODEL, type: 'llm', recipe: 'llamacpp', device: 'gpu', checkpoint: '', backend_url: '', pid: 1, last_use: Date.now() }],
    });

    await openWizard(page);

    // Model step: pre-filled with the loaded chat model.
    await expect(page.locator('[data-autoopt-model-select]')).toHaveValue(CHAT_MODEL, { timeout: 10000 });
    await page.locator('[data-autoopt-next]').click();

    await expect(page.locator('[data-autoopt-step="parallel"]')).toBeVisible();
    await page.locator('[data-autoopt-next]').click();

    await expect(page.locator('[data-autoopt-step="kv"]')).toBeVisible();
    await expect(page.locator('[data-autoopt-option="kv:q8_0"]')).toHaveAttribute('aria-pressed', 'true');
    await page.locator('[data-autoopt-next]').click();

    await expect(page.locator('[data-autoopt-step="ram"]')).toBeVisible();
    await expect(page.locator('[data-autoopt-ram-suggestion]')).toContainText('Suggested for this machine (64 GB RAM)');
    await page.locator('[data-autoopt-next]').click();

    // Vision step skipped for a text-only model.
    await expect(page.locator('[data-autoopt-step="budget"]')).toBeVisible();
    await expect(page.locator('[data-autoopt-step="vision"]')).toHaveCount(0);
    await page.locator('[data-autoopt-option="budget:standard"]').click();

    // Consent gate for the Benchmark tier.
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-step="review"]')).toBeVisible();
    await expect(page.locator('[data-autoopt-start]')).toBeDisabled();
    await page.locator('[data-autoopt-back]').click();
    await page.locator('[data-autoopt-consent]').check();
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-start]')).toBeEnabled();

    await page.locator('[data-autoopt-start]').click();
    await expect(page.locator('[data-autoopt-step="running"]')).toBeVisible();
    await expect(page.locator('[data-autoopt-close-note]')).toContainText('the run continues');

    // The controller pipeline finishes against the mocks.
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--completed')).toHaveCount(7, { timeout: 15000 });
    await page.locator('[data-autoopt-wizard] .slideover__close').click();

    const run = page.locator('[data-autoopt-run]');
    await expect(run.locator('.autoopt-status-chip--completed')).toBeVisible();
    await expect(page.locator('[data-autoopt-announcement]')).toContainText(`AutoOpt run for ${CHAT_MODEL} completed.`);

    // Controller composed the standalone queries itself.
    expect(counts.unload).toBeGreaterThanOrEqual(1);
    const realFits = counts.fit.filter(body => body.model === CHAT_MODEL);
    expect(realFits.length).toBeGreaterThanOrEqual(2);
    expect(realFits[0]).toMatchObject({ model: CHAT_MODEL, backend: 'vulkan', fit_target_mib: 1024 });
    const duelBackends = counts.bench.filter(body => body.b === undefined).map(body => body.backend);
    expect(duelBackends).toEqual(['vulkan', 'rocm']);
    const ladderRungs = counts.bench.filter(body => body.b !== undefined).map(body => body.b);
    expect(ladderRungs).toEqual([512, 2048, 8192]);

    // Synthesized recommendation is visible in the run detail.
    await page.locator('[data-autoopt-inspect]').click();
    await page.waitForSelector('[data-autoopt-detail]');
    await expect(page.locator('[data-autoopt-rec-args]')).toContainText('-ctk q8_0 -ctv q8_0');
    await expect(page.locator('[data-autoopt-rec-args]')).toContainText('-b 2048 -ub 2048');
    await expect(page.locator('[data-autoopt-bench-duel] tbody tr')).toHaveCount(2);
    await expect(page.locator('[data-autoopt-bench-ladder] tbody tr')).toHaveCount(3);
    await expect(page.locator('[data-autoopt-alternatives] tbody tr')).toHaveCount(3);
  });

  test('Fast Scan runs without consent and without benchmarks while models are loaded', async ({ page }) => {
    const counts = await mockServer(page, {
      loadedModels: [{ model_name: CHAT_MODEL, type: 'llm', recipe: 'llamacpp', device: 'gpu', checkpoint: '', backend_url: '', pid: 1, last_use: Date.now() }],
    });

    await openWizard(page);
    await expect(page.locator('[data-autoopt-model-select]')).toHaveValue(CHAT_MODEL, { timeout: 10000 });
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-step="budget"]')).toBeVisible();

    await page.locator('[data-autoopt-option="budget:quick"]').click();
    await expect(page.locator('[data-autoopt-consent]')).toHaveCount(0);
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-step="review"]')).toContainText('Not needed for Fast Scan');
    await expect(page.locator('[data-autoopt-start]')).toBeEnabled();
    await page.locator('[data-autoopt-start]').click();

    await expect(page.locator('[data-autoopt-step="running"]')).toBeVisible();
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--completed')).toHaveCount(5, { timeout: 15000 });
    await expect(page.locator('[data-autoopt-stage-list] .autoopt-stage--skipped')).toHaveCount(2);

    expect(counts.bench).toHaveLength(0);
    expect(counts.unload).toBe(0);
  });

  test('model picker only offers downloaded llama.cpp models and explains the scope', async ({ page }) => {
    await mockServer(page);
    await openWizard(page);

    await expect(page.locator('[data-autoopt-model-note]')).toContainText('AutoOpt currently supports downloaded llama.cpp models');
    await expect(page.locator('[data-autoopt-model-note]')).toContainText('not other engines');

    const select = page.locator('[data-autoopt-model-select]');
    await expect(select.locator(`option[value="${CHAT_MODEL}"]`)).toHaveCount(1);
    await expect(select.locator(`option[value="${VISION_MODEL}"]`)).toHaveCount(1);
    await expect(select.locator(`option[value="${NPU_MODEL}"]`)).toHaveCount(0);
    await expect(select.locator(`option[value="${REGISTRY_MODEL}"]`)).toHaveCount(0);
  });

  test('wizard shows the vision step for vision-capable models', async ({ page }) => {
    await mockServer(page);
    await openWizard(page);

    await page.locator('[data-autoopt-model-select]').selectOption(VISION_MODEL);
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await expect(page.locator('[data-autoopt-step="vision"]')).toBeVisible();
    await page.locator('[data-autoopt-option="vision:false"]').click();
    await expect(page.locator('[data-autoopt-option="vision:false"]')).toHaveAttribute('aria-pressed', 'true');
  });

  test('cancel mid-bench aborts the in-flight bench fetch and marks the run cancelled', async ({ page }) => {
    await mockServer(page, {
      benchHandler: () => new Promise(() => {}),
    });

    await openWizard(page);
    await page.locator('[data-autoopt-model-select]').selectOption(CHAT_MODEL);
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-option="budget:standard"]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-start]').click();

    const benchRequest = page.waitForRequest('**/api/v1/backends/llamacpp/bench', { timeout: 15000 });
    const benchAborted = page.waitForEvent('requestfailed', {
      predicate: request => request.url().includes('/backends/llamacpp/bench'),
      timeout: 15000,
    });
    await benchRequest;

    await page.locator('[data-autoopt-cancel-run]').click();
    await benchAborted;

    await page.locator('[data-autoopt-wizard] .slideover__close').click();
    await expect(page.locator('[data-autoopt-run] .autoopt-status-chip--cancelled')).toBeVisible({ timeout: 10000 });
  });

  test('a hard stage failure surfaces the error in rail and detail', async ({ page }) => {
    await mockServer(page);
    // Break model_facts: detail responses fail the run hard.
    await page.unroute('**/api/v1/models/*');
    await page.route('**/api/v1/models/*', route => route.fulfill({ status: 500, json: { error: 'registry corrupt' } }));

    await openWizard(page);
    await page.locator('[data-autoopt-model-select]').selectOption(CHAT_MODEL);
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-option="budget:quick"]').click();
    await page.locator('[data-autoopt-next]').click();
    await page.locator('[data-autoopt-start]').click();

    await expect(page.locator('[data-autoopt-step="running"] .preset-error')).toContainText('could not read model metadata', { timeout: 15000 });
    await page.locator('[data-autoopt-wizard] .slideover__close').click();

    const run = page.locator('[data-autoopt-run]');
    await expect(run.locator('.autoopt-status-chip--failed')).toContainText('could not read model metadata');
    await page.locator('[data-autoopt-inspect]').click();
    await page.waitForSelector('[data-autoopt-detail]');
    await expect(page.locator('[data-autoopt-detail-error]')).toContainText('could not read model metadata');
    await expect(page.locator('.autoopt-stage--failed .autoopt-stage__name')).toContainText('model_facts');
  });

  test('a server without the tool endpoints shows the unsupported notice', async ({ page }) => {
    await page.route('**/api/v1/health**', route => route.fulfill({
      json: { status: 'ok', version: 'test', all_models_loaded: [] },
    }));
    await page.route('**/api/v1/models**', route => route.fulfill({ json: { data: MODELS } }));
    await page.route('**/api/v1/system-info**', route => route.fulfill({ json: SYSTEM_INFO }));
    await page.route('**/api/v1/backends/llamacpp/**', route => route.fulfill({ status: 404, body: 'Not Found' }));

    await openPresets(page);
    const notice = page.locator('[data-autoopt-unsupported]');
    await expect(notice).toBeVisible({ timeout: 10000 });
    await expect(notice).toContainText('does not support the llama.cpp tool endpoints');
    await expect(page.locator('[data-autoopt-run-optimizer]')).toBeDisabled();
  });
});

test.describe('AutoOpt rail persistence', () => {

  test('a run interrupted by page reload is failed as interrupted', async ({ page }) => {
    await mockServer(page);
    await seedRuns(page, [{
      ...COMPLETED_RUN,
      id: 'ao-20260713-090000-dead',
      status: 'running',
      finished_at: undefined,
      result: undefined,
      summary: undefined,
    }]);

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
    await expect(page.locator('[data-autoopt-rec-args]')).toContainText('-b 512 -ub 256 -ctk q8_0 -ctv q8_0');
    await expect(page.locator('[data-autoopt-bench-duel] tbody tr')).toHaveCount(2);
    await expect(page.locator('[data-autoopt-bench-ladder] tbody tr')).toHaveCount(2);
  });

  test('delete removes a run permanently; a failed persist rolls the row and selection back', async ({ page }) => {
    const otherRun = { ...COMPLETED_RUN, id: 'ao-20260713-110000-beef', created_at: '2026-07-13T11:00:00Z' };
    await mockServer(page);
    await seedRuns(page, [otherRun, COMPLETED_RUN]);

    await openPresets(page);
    const run = page.locator(`[data-autoopt-run="${COMPLETED_RUN.id}"]`);
    await expect(run).toBeVisible();

    // Select the run, then make the next persisted write fail (quota).
    await run.locator('.auto-run-card__main').click();
    await expect(run.locator('.auto-run-card__main')).toHaveAttribute('aria-pressed', 'true');
    await page.evaluate((key) => {
      const original = localStorage.setItem.bind(localStorage);
      let armed = true;
      localStorage.setItem = (k: string, v: string) => {
        if (k === key && armed) {
          armed = false;
          throw new Error('storage quota exceeded');
        }
        return original(k, v);
      };
    }, RUNS_KEY);

    await page.locator(`[data-autoopt-delete="${COMPLETED_RUN.id}"]`).click();
    await expect(run).toBeVisible();
    await expect(page.locator('[data-autoopt-rail-error]')).toContainText('storage quota exceeded');
    await expect(run.locator('.auto-run-card__main')).toHaveAttribute('aria-pressed', 'true');

    // Second attempt (storage healthy again) removes it for good.
    await page.locator(`[data-autoopt-delete="${COMPLETED_RUN.id}"]`).click();
    await expect(run).toHaveCount(0);
    await expect(page.locator(`[data-autoopt-run="${otherRun.id}"]`)).toBeVisible();
    const persisted = await page.evaluate((key) => {
      const parsed = JSON.parse(localStorage.getItem(key) || '{}');
      return Array.isArray(parsed.runs) ? parsed.runs.map((r: any) => r.id) : [];
    }, RUNS_KEY);
    expect(persisted).toEqual([otherRun.id]);
  });
});

test.describe('AutoOpt run detail actions', () => {

  async function openCompletedRunDetail(page: Page): Promise<MockCounts> {
    const counts = await mockServer(page);
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
    await expect(page.locator('[data-autoopt-cta-help]')).toContainText('future loads');
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

    expect(stored.preset).not.toBeNull();
    expect(stored.preset.auto_opt_run_id).toBe(COMPLETED_RUN.id);
    expect(stored.preset.name).toContain('AutoOpt');
    expect(stored.tuning).not.toBeNull();
    expect(stored.tuning.source).toBe('optimized');
    expect(stored.tuning.auto_opt_run_id).toBe(COMPLETED_RUN.id);
    expect(stored.tuning.recipe_options.llamacpp_args).toBe('-b 512 -ub 256 -ctk q8_0 -ctv q8_0');
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
    await page.route('**/api/v1/load', route => route.fulfill({
      status: 500,
      json: { error: { message: 'llama-server failed to start' } },
    }));

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
    expect(stored).not.toBeNull();
    expect(stored.name).toContain('AutoOpt');
  });

  test('"Try now without saving" loads once with the recommended runtime options', async ({ page }) => {
    const counts = await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-try-now]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('nothing saved');

    expect(counts.load).toHaveLength(1);
    const body = counts.load[0];
    expect(body.model_name).toBe(CHAT_MODEL);
    expect(body.ctx_size).toBe(8192);
    expect(body.llamacpp_args).toBe('-b 512 -ub 256 -ctk q8_0 -ctv q8_0');
    expect(body.llamacpp_backend).toBe('vulkan');
    expect(body.mmproj_enabled).toBe(false);
    expect(body.save_options).toBe(false);

    const stored = await page.evaluate(() => {
      for (const key of Object.keys(localStorage)) {
        if (key.includes('user_presets')) {
          const presets = JSON.parse(localStorage.getItem(key) || '[]');
          if (presets.some((p: any) => p.auto_opt_run_id)) return 'preset-created';
        }
      }
      return 'nothing-saved';
    });
    expect(stored).toBe('nothing-saved');
  });

  test('"Use this instead" swaps the CTA target to the alternative', async ({ page }) => {
    const counts = await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-use-alternative="CPU fallback"]').click();
    await expect(page.locator('[data-autoopt-rec-args]')).toContainText('-b 256 -ub 128');
    await page.locator('[data-autoopt-try-now]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('nothing saved');
    expect(counts.load[0].ctx_size).toBe(16384);
    expect(counts.load[0].llamacpp_backend).toBe('cpu');
  });

  test('preset editor links back to the producing run via the AutoOpt chip', async ({ page }) => {
    await openCompletedRunDetail(page);
    await page.locator('[data-autoopt-create-preset]').click();
    await expect(page.locator('[data-autoopt-detail-notice]')).toContainText('Created preset');
    await page.locator('[data-autoopt-detail] .slideover__close').click();

    const yourCards = page.locator('[data-recipe-grid="yours"] .recipe-card');
    await expect(yourCards.first()).toBeVisible();
    await yourCards.first().locator('.recipe-card__overlay-btn').click();
    await page.waitForSelector('.slideover.is-open');

    await expect(page.locator('[data-preset-autoopt-chip]')).toBeVisible();
    await page.locator('[data-preset-autoopt-chip]').click();
    await expect(page.locator('[data-autoopt-detail]')).toBeVisible();
  });
});
