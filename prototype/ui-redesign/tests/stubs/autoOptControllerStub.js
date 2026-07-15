const AUTOOPT_STAGES = ['snapshot', 'model_facts', 'hf_metadata', 'fit_estimate', 'bench_job', 'synthesize'];

const attached = [];

function track(id, signal) {
  attached.push({ id, signal });
  return new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => {
      reject(Object.assign(new Error('cancelled by user'), { name: 'AbortError' }));
    }, { once: true });
  });
}

function resumeAutoOptRun(run, _cb, signal) {
  return track(run.id, signal);
}

function executeAutoOptRun(_request, signal, _cb) {
  return track('(new)', signal);
}

globalThis.__AO_CTRL__ = { attached };

module.exports = { __esModule: true, AUTOOPT_STAGES, resumeAutoOptRun, executeAutoOptRun };
