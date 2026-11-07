import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import api, {
  type HFModelResult,
  type HuggingFaceRepoFile,
  type PullVariant,
  type ServerDirectoryListing,
  fetchHuggingFaceRepoFiles,
  friendlyErrorMessage,
  searchHuggingFaceRepositories,
} from '../api';
import { useI18n } from '../i18n';
import { Icon } from './Icon';
import { WorkspaceActionButton } from './WorkspacePanels';

export type CheckpointSourceMode = 'checkpoint' | 'repository' | 'local';

const SEARCH_DEBOUNCE_MS = 300;

export function isLocalCheckpointPath(value: string): boolean {
  return /^(\/|~|[A-Za-z]:[\\/]|\\\\|\.{1,2}[\\/])/.test(value.trim());
}

export function inferCheckpointSourceMode(value: string): CheckpointSourceMode | '' {
  const trimmed = value.trim();
  if (!trimmed) return '';
  if (isLocalCheckpointPath(trimmed)) return 'local';
  return trimmed.includes(':') ? 'checkpoint' : 'repository';
}

function splitCheckpoint(value: string): { repo: string; selector: string } {
  const trimmed = value.trim();
  const colon = trimmed.indexOf(':');
  if (colon < 0) return { repo: trimmed, selector: '' };
  return { repo: trimmed.slice(0, colon), selector: trimmed.slice(colon + 1) };
}

function hasExtension(path: string, extensions: string[]): boolean {
  if (extensions.length === 0) return true;
  const lower = path.toLowerCase();
  return extensions.some(ext => lower.endsWith(ext.toLowerCase()));
}

function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}

function parentDirectory(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, '');
  const index = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
  if (index < 0) return '';
  if (index === 0) return trimmed.slice(0, 1);
  return trimmed.slice(0, index);
}

/* ── Repository search ─────────────────────────────────────── */

interface RepositorySearchProps {
  inputId: string;
  searchFilter?: string;
  disabled?: boolean;
  onSelect: (repo: string) => void;
}

const RepositorySearch: React.FC<RepositorySearchProps> = ({ inputId, searchFilter, disabled, onSelect }) => {
  const { t } = useI18n();
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [results, setResults] = useState<HFModelResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeIndex, setActiveIndex] = useState(-1);
  const listboxId = `${inputId}-listbox`;
  const trimmed = query.trim();

  useEffect(() => {
    if (trimmed.length < 2) {
      setResults([]);
      setLoading(false);
      setError('');
      return undefined;
    }
    const controller = new AbortController();
    setLoading(true);
    const timer = window.setTimeout(() => {
      searchHuggingFaceRepositories(trimmed, { filter: searchFilter }, controller.signal)
        .then(found => { setResults(found); setError(''); })
        .catch(err => {
          if (controller.signal.aborted) return;
          setResults([]);
          setError(friendlyErrorMessage(err));
        })
        .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [trimmed, searchFilter]);

  const options = useMemo(() => {
    const ids = results.map(result => result.id);
    const typedRepo = trimmed.includes('/') && !ids.includes(trimmed) ? [trimmed] : [];
    return [...typedRepo, ...ids];
  }, [results, trimmed]);

  useEffect(() => { setActiveIndex(-1); }, [options.length]);

  const choose = (repo: string) => {
    onSelect(repo);
    setQuery('');
    setOpen(false);
    setActiveIndex(-1);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setOpen(true);
      setActiveIndex(prev => options.length > 0 ? (prev + 1) % options.length : -1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setOpen(true);
      setActiveIndex(prev => options.length > 0 ? (prev <= 0 ? options.length - 1 : prev - 1) : -1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && options[activeIndex]) choose(options[activeIndex]);
      else if (trimmed.includes('/')) choose(trimmed);
    } else if (e.key === 'Escape') {
      setOpen(false);
      setActiveIndex(-1);
    }
  };

  const resultById = new Map(results.map(result => [result.id, result]));

  return (
    <div className="checkpoint-source-field__search">
      <Icon name="search" size={14} className="checkpoint-source-field__search-icon" />
      <input
        id={inputId}
        className="input"
        role="combobox"
        aria-expanded={open && trimmed.length >= 2}
        aria-controls={listboxId}
        aria-autocomplete="list"
        aria-activedescendant={open && activeIndex >= 0 ? `${listboxId}-${activeIndex}` : undefined}
        value={query}
        disabled={disabled}
        placeholder={t('Search Hugging Face repositories')}
        autoComplete="off"
        spellCheck={false}
        onChange={e => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 120)}
        onKeyDown={handleKeyDown}
      />
      {open && trimmed.length >= 2 && (
        <div className="omni-component-picker__menu checkpoint-source-field__menu">
          <div role="listbox" id={listboxId} aria-label={t('Hugging Face repositories')}>
            {options.map((repo, index) => {
              const result = resultById.get(repo);
              return (
                <div
                  key={repo}
                  id={`${listboxId}-${index}`}
                  role="option"
                  aria-selected={index === activeIndex}
                  className={`omni-component-picker__option${index === activeIndex ? ' omni-component-picker__option--focused' : ''}`}
                  onMouseDown={e => e.preventDefault()}
                  onClick={() => choose(repo)}
                >
                  <span className="omni-component-picker__option-name">{repo}</span>
                  <span className="omni-component-picker__option-detail">
                    {result
                      ? t('{downloads} downloads', { downloads: (result.downloads || 0).toLocaleString() })
                      : t('Use this repository id as typed')}
                  </span>
                </div>
              );
            })}
          </div>
          {loading && <div className="omni-component-picker__empty">{t('Searching Hugging Face...')}</div>}
          {!loading && error && <div className="omni-component-picker__empty">{error}</div>}
          {!loading && !error && options.length === 0 && (
            <div className="omni-component-picker__empty">{t('No repositories found.')}</div>
          )}
        </div>
      )}
    </div>
  );
};

/* ── Quantization / file selector ──────────────────────────── */

interface RepoSelectorProps {
  repo: string;
  value: string;
  extensions: string[];
  showQuantizations: boolean;
  quantizationValue: 'name' | 'file';
  onChange: (selector: string) => void;
}

interface SelectorOption {
  value: string;
  label: string;
}

const RepoSelector: React.FC<RepoSelectorProps> = ({ repo, value, extensions, showQuantizations, quantizationValue, onChange }) => {
  const { t } = useI18n();
  const [variants, setVariants] = useState<PullVariant[]>([]);
  const [files, setFiles] = useState<HuggingFaceRepoFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setFailed(false);
    const wantsVariants = showQuantizations && extensions.includes('.gguf');
    Promise.allSettled([
      wantsVariants ? api.pullVariants(repo, 'huggingface', controller.signal) : Promise.resolve(null),
      fetchHuggingFaceRepoFiles(repo, controller.signal),
    ]).then(([variantResult, fileResult]) => {
      if (controller.signal.aborted) return;
      setVariants(variantResult.status === 'fulfilled' && variantResult.value ? variantResult.value.variants || [] : []);
      setFiles(fileResult.status === 'fulfilled' ? fileResult.value : []);
      setFailed(variantResult.status === 'rejected' && fileResult.status === 'rejected');
      setLoading(false);
    });
    return () => controller.abort();
  }, [repo, showQuantizations, extensions]);

  const { quantizations, otherFiles } = useMemo(() => {
    const covered = new Set<string>();
    const quantOptions: SelectorOption[] = [];
    for (const variant of variants) {
      for (const file of variant.files) covered.add(file);
      if (variant.draft_file) covered.add(variant.draft_file);
      const size = formatBytes(variant.size_bytes);
      quantOptions.push({
        value: quantizationValue === 'file' && !variant.sharded ? variant.primary_file : variant.name,
        label: [variant.name, variant.sharded ? t('{count} shards', { count: variant.files.length }) : '', size].filter(Boolean).join(' · '),
      });
    }
    const fileOptions = files
      .filter(file => hasExtension(file.path, extensions) && !covered.has(file.path))
      .map(file => ({ value: file.path, label: [file.path, formatBytes(file.size_bytes)].filter(Boolean).join(' · ') }));
    return { quantizations: quantOptions, otherFiles: fileOptions };
  }, [variants, files, extensions, quantizationValue, t]);

  const known = value && [...quantizations, ...otherFiles].some(option => option.value === value);

  if (!loading && (failed || (quantizations.length === 0 && otherFiles.length === 0))) {
    return (
      <input
        className="input checkpoint-source-field__selector"
        value={value}
        onChange={e => onChange(e.target.value.trim())}
        placeholder={failed ? t('Could not list files; type a quantization or file') : t('No matching files; type a quantization or file')}
        spellCheck={false}
      />
    );
  }

  return (
    <select
      className="select checkpoint-source-field__selector"
      value={value}
      disabled={loading}
      onChange={e => onChange(e.target.value)}
      aria-label={t('Quantization or file')}
    >
      <option value="">{loading ? t('Loading files...') : t('Choose a quantization or file')}</option>
      {value && !known && !loading && <option value={value}>{value}</option>}
      {quantizations.length > 0 && (
        <optgroup label={t('Quantizations')}>
          {quantizations.map(option => <option key={`q:${option.value}`} value={option.value}>{option.label}</option>)}
        </optgroup>
      )}
      {otherFiles.length > 0 && (
        <optgroup label={quantizations.length > 0 ? t('Other files') : t('Files')}>
          {otherFiles.map(option => <option key={`f:${option.value}`} value={option.value}>{option.label}</option>)}
        </optgroup>
      )}
    </select>
  );
};

/* ── Server file browser ───────────────────────────────────── */

interface ServerFileBrowserProps {
  initialPath: string;
  extensions: string[];
  onSelect: (path: string) => void;
  onClose: () => void;
}

const ServerFileBrowser: React.FC<ServerFileBrowserProps> = ({ initialPath, extensions, onSelect, onClose }) => {
  const { t } = useI18n();
  const [listing, setListing] = useState<ServerDirectoryListing | null>(null);
  const [pathInput, setPathInput] = useState(initialPath);
  const [selected, setSelected] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const requestRef = useRef<AbortController | null>(null);

  const load = (path: string) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setLoading(true);
    setError('');
    api.listServerFiles(path, extensions, controller.signal)
      .then(result => {
        if (controller.signal.aborted) return;
        setListing(result);
        setPathInput(result.path);
        setSelected('');
      })
      .catch(err => {
        if (controller.signal.aborted) return;
        setError(friendlyErrorMessage(err));
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
  };

  const initialPathRef = useRef(initialPath);
  useEffect(() => {
    load(initialPathRef.current);
    return () => requestRef.current?.abort();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const body = (
    <div className="inspect-modal-overlay" onClick={onClose}>
      <div
        className="inspect-modal-content server-file-browser"
        role="dialog"
        aria-modal="true"
        aria-label={t('Choose a file on the server')}
        onClick={e => e.stopPropagation()}
      >
        <div className="inspect-modal-header">
          <h4>{t('Choose a file on the server')}</h4>
          <WorkspaceActionButton appearance="quiet" size="toolbar" icon="x" iconOnly className="close-modal-btn" onClick={onClose} aria-label={t('Cancel')} title={t('Cancel')} />
        </div>
        <div className="inspect-modal-body server-file-browser__body">
          <form
            className="server-file-browser__location"
            onSubmit={e => { e.preventDefault(); load(pathInput.trim()); }}
          >
            <WorkspaceActionButton
              size="small"
              icon="chevron-up"
              iconOnly
              disabled={!listing?.parent || loading}
              onClick={() => listing?.parent && load(listing.parent)}
              aria-label={t('Parent directory')}
              title={t('Parent directory')}
            />
            <input
              className="input"
              value={pathInput}
              onChange={e => setPathInput(e.target.value)}
              spellCheck={false}
              aria-label={t('Directory')}
            />
            <WorkspaceActionButton size="small" type="submit" disabled={loading}>{t('Open')}</WorkspaceActionButton>
          </form>
          {listing && listing.roots.length > 1 && (
            <div className="server-file-browser__roots">
              {listing.roots.map(root => (
                <button key={root} type="button" className="server-file-browser__root" onClick={() => load(root)}>{root}</button>
              ))}
            </div>
          )}
          <div className="server-file-browser__hint">
            {t('Showing folders and {extensions} files on the machine running Lemonade.', { extensions: extensions.join(', ') })}
          </div>
          {error && <div className="custom-model-form__error">{error}</div>}
          <div className="server-file-browser__list" role="listbox" aria-label={t('Directory contents')} aria-busy={loading}>
            {listing && listing.entries.length === 0 && !loading && (
              <div className="server-file-browser__empty">{t('No folders or matching files here.')}</div>
            )}
            {listing?.entries.map(entry => (
              <div
                key={entry.path}
                role="option"
                aria-selected={entry.path === selected}
                tabIndex={0}
                className={`server-file-browser__entry server-file-browser__entry--${entry.kind}${entry.path === selected ? ' server-file-browser__entry--selected' : ''}`}
                onClick={() => (entry.kind === 'directory' ? load(entry.path) : setSelected(entry.path))}
                onDoubleClick={() => { if (entry.kind === 'file') onSelect(entry.path); }}
                onKeyDown={e => {
                  if (e.key !== 'Enter') return;
                  if (entry.kind === 'directory') load(entry.path);
                  else onSelect(entry.path);
                }}
              >
                <Icon name={entry.kind === 'directory' ? 'chevron-right' : 'file'} size={14} />
                <span className="server-file-browser__name">{entry.name}</span>
                <span className="server-file-browser__size">{entry.kind === 'file' ? formatBytes(entry.size_bytes) : ''}</span>
              </div>
            ))}
            {listing?.truncated && <div className="server-file-browser__empty">{t('Only the first 5000 entries are shown.')}</div>}
          </div>
        </div>
        <div className="inspect-modal-footer server-file-browser__footer">
          <WorkspaceActionButton appearance="secondary" onClick={onClose}>{t('Cancel')}</WorkspaceActionButton>
          <WorkspaceActionButton appearance="primary" disabled={!selected} onClick={() => onSelect(selected)}>{t('Select')}</WorkspaceActionButton>
        </div>
      </div>
    </div>
  );
  return createPortal(body, document.body);
};

/* ── Hybrid field ──────────────────────────────────────────── */

export interface CheckpointSourceFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  extensions: string[];
  searchFilter?: string;
  showQuantizations?: boolean;
  quantizationValue?: 'name' | 'file';
  example?: React.ReactNode;
  wide?: boolean;
}

export const CheckpointSourceField: React.FC<CheckpointSourceFieldProps> = ({
  label,
  value,
  onChange,
  extensions,
  searchFilter,
  showQuantizations = true,
  quantizationValue = 'name',
  example,
  wide,
}) => {
  const { t } = useI18n();
  const baseId = useId();
  const labelId = `${baseId}-label`;
  const [mode, setMode] = useState<CheckpointSourceMode | ''>(() => inferCheckpointSourceMode(value));
  const [browsing, setBrowsing] = useState(false);
  const lastEmitted = useRef(value);

  useEffect(() => {
    if (value === lastEmitted.current) return;
    lastEmitted.current = value;
    setMode(inferCheckpointSourceMode(value));
  }, [value]);

  const emit = (next: string) => {
    lastEmitted.current = next;
    onChange(next);
  };

  const { repo, selector } = mode === 'local' ? { repo: '', selector: '' } : splitCheckpoint(value);

  const changeMode = (next: CheckpointSourceMode | '') => {
    setMode(next);
    if (next === 'repository' && repo) emit(repo);
    else if (next === 'checkpoint' && repo && mode === 'repository') emit(repo);
    else emit('');
  };

  const sourceControl = (() => {
    if (!mode) {
      return <input className="input" disabled placeholder={t('Choose a source type first')} aria-labelledby={labelId} />;
    }
    if (mode === 'local') {
      return (
        <div className="checkpoint-source-field__local">
          <input
            className="input"
            value={value}
            onChange={e => emit(e.target.value)}
            placeholder={t('/path/to/model{extension}', { extension: extensions[0] || '' })}
            spellCheck={false}
            aria-labelledby={labelId}
          />
          <WorkspaceActionButton size="small" icon="hard-drive" onClick={() => setBrowsing(true)}>{t('Browse...')}</WorkspaceActionButton>
        </div>
      );
    }
    if (!repo) {
      return (
        <RepositorySearch
          inputId={`${baseId}-search`}
          searchFilter={searchFilter}
          onSelect={selectedRepo => emit(selectedRepo)}
        />
      );
    }
    return (
      <div className="checkpoint-source-field__chosen">
        <span className="checkpoint-source-field__chip" title={repo}>
          <Icon name="hugging-face" size={12} />
          <span className="checkpoint-source-field__chip-text">{repo}</span>
          <button
            type="button"
            className="checkpoint-source-field__chip-clear"
            onClick={() => emit('')}
            aria-label={t('Clear repository')}
            title={t('Clear repository')}
          >×</button>
        </span>
        {mode === 'checkpoint' && (
          <RepoSelector
            repo={repo}
            value={selector}
            extensions={extensions}
            showQuantizations={showQuantizations}
            quantizationValue={quantizationValue}
            onChange={nextSelector => emit(nextSelector ? `${repo}:${nextSelector}` : repo)}
          />
        )}
      </div>
    );
  })();

  return (
    <div className={`custom-model-form__field checkpoint-source-field${wide ? ' custom-model-form__wide' : ''}`} role="group" aria-labelledby={labelId}>
      <span id={labelId} className="checkpoint-source-field__label">{label}</span>
      {example}
      <div className="checkpoint-source-field__row">
        <select
          className="select checkpoint-source-field__mode"
          value={mode}
          onChange={e => changeMode(e.target.value as CheckpointSourceMode | '')}
          aria-label={t('Source type')}
        >
          <option value="">{t('Source...')}</option>
          <option value="checkpoint">{t('Checkpoint')}</option>
          <option value="repository">{t('Hugging Face repository')}</option>
          <option value="local">{t('Local path')}</option>
        </select>
        <div className="checkpoint-source-field__source">{sourceControl}</div>
      </div>
      {browsing && (
        <ServerFileBrowser
          initialPath={value ? parentDirectory(value) : ''}
          extensions={extensions}
          onSelect={path => { emit(path); setBrowsing(false); }}
          onClose={() => setBrowsing(false)}
        />
      )}
    </div>
  );
};
