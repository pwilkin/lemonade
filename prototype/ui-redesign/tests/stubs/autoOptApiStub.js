const statusListeners = [];
const state = { base: globalThis.__AO_INITIAL_BASE__ || 'http://server-a:1111' };

const api = {
  get baseUrl() { return state.base; },
  get healthData() { return { version: 'test' }; },
  get allModels() { return []; },
  onStatusChange(fn) {
    statusListeners.push(fn);
    return () => {
      const i = statusListeners.indexOf(fn);
      if (i >= 0) statusListeners.splice(i, 1);
    };
  },
  interruptJob() { return Promise.resolve(); },
  deleteJob() { return Promise.resolve(); },
};

globalThis.__AO_API_CONTROL__ = {
  setBase(b) {
    state.base = b;
    statusListeners.slice().forEach(fn => { try { fn('connected'); } catch {} });
  },
};

module.exports = { __esModule: true, default: api };
