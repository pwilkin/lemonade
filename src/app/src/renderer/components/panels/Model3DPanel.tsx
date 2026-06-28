import React, { useState, useEffect, useRef } from 'react';
import { useModels } from '../../hooks/useModels';
import { Modality } from '../../hooks/useInferenceState';
import { ModelsData } from '../../utils/modelData';
import { AppSettings } from '../../utils/appSettings';
import { serverFetch } from '../../utils/serverConfig';
import ModelSelector from '../ModelSelector';
import EmptyState from '../EmptyState';
import TypingIndicator from '../TypingIndicator';
import ModelViewer3D from '../ModelViewer3D';

interface Model3DPanelProps {
  isBusy: boolean;
  isPreFlight: boolean;
  isInferring: boolean;
  activeModality: Modality | null;
  runPreFlight: (modality: Modality, options: { modelName: string; modelsData: ModelsData; onError: (msg: string) => void }) => Promise<boolean>;
  reset: () => void;
  showError: (msg: string) => void;
  appSettings: AppSettings | null;
}

const Model3DPanel: React.FC<Model3DPanelProps> = ({
  isBusy, isPreFlight, isInferring, activeModality, runPreFlight, reset, showError,
}) => {
  const { selectedModel, modelsData } = useModels();

  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const glbUrlRef = useRef<string | null>(null);
  glbUrlRef.current = glbUrl;

  useEffect(() => () => { if (glbUrlRef.current) URL.revokeObjectURL(glbUrlRef.current); }, []);

  const handlePickImage = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setImageDataUrl(ev.target?.result as string);
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const handleGenerate = async () => {
    if (!imageDataUrl || isBusy || !selectedModel) return;

    // Strip the "data:image/...;base64," prefix — the server wants raw base64.
    const comma = imageDataUrl.indexOf(',');
    const b64 = comma >= 0 ? imageDataUrl.slice(comma + 1) : imageDataUrl;

    const ready = await runPreFlight('model3d', { modelName: selectedModel, modelsData, onError: showError });
    if (!ready) return;

    try {
      const response = await serverFetch('/3d/generations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, image: b64 }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const blob = await response.blob();
      if (glbUrlRef.current) URL.revokeObjectURL(glbUrlRef.current);
      setGlbUrl(URL.createObjectURL(blob));
    } catch (error: any) {
      console.error('3D generation failed:', error);
      showError(`Failed to generate 3D model: ${error.message || 'Unknown error'}`);
    } finally {
      reset();
    }
  };

  return (
    <>
      <div className="chat-messages">
        {!glbUrl && !isBusy && <EmptyState title="Lemonade 3D Generator" />}
        {glbUrl && (
          <div className="chat-message" style={{ height: '420px' }}>
            <ModelViewer3D src={glbUrl} />
            <a href={glbUrl} download="model.glb" className="download-link" style={{ display: 'inline-block', marginTop: '6px' }}>
              Download .glb
            </a>
          </div>
        )}
        {isPreFlight && activeModality === 'model3d' && (
          <div className="model-loading-indicator"><span className="model-loading-text">Loading 3D model...</span></div>
        )}
        {isInferring && activeModality === 'model3d' && (
          <div className="model-loading-indicator"><span className="model-loading-text">Reconstructing 3D mesh (this can take a few minutes)...</span></div>
        )}
      </div>

      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <div className="model3d-input-row" style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handlePickImage} />
            <button className="secondary-button" onClick={() => fileInputRef.current?.click()} disabled={isBusy}>
              {imageDataUrl ? 'Change image' : 'Choose image'}
            </button>
            {imageDataUrl && (
              <img src={imageDataUrl} alt="input" style={{ height: '48px', borderRadius: '4px' }} />
            )}
          </div>
          <div className="chat-controls">
            <div className="chat-controls-left">
              <ModelSelector disabled={isBusy} />
            </div>
            {isBusy ? (
              <button className="chat-send-button" disabled title="Processing...">
                <TypingIndicator size="small" />
              </button>
            ) : (
              <button className="chat-send-button" onClick={handleGenerate} disabled={!imageDataUrl} title="Generate 3D">
                Generate
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default Model3DPanel;
