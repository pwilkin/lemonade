import React, { useState, useEffect, useRef } from 'react';
import { useModels } from '../../hooks/useModels';
import { Modality } from '../../hooks/useInferenceState';
import { ModelsData } from '../../utils/modelData';
import { AppSettings } from '../../utils/appSettings';
import { serverFetch } from '../../utils/serverConfig';
import ModelSelector from '../ModelSelector';
import EmptyState from '../EmptyState';
import InferenceControls from '../InferenceControls';
import ImagePreviewList from '../ImagePreviewList';
import { ImageUploadIcon } from '../Icons';
import ModelViewer3D from '../ModelViewer3D';

// Cascade options with rough on-device cost (measured on a 16 GB GPU / 31 GB host).
const RESOLUTIONS = [
  { value: '512',  label: '512 — ~3 GB VRAM, ~3 GB RAM, fast' },
  { value: '1024', label: '1024 — ~15 GB VRAM, ~15 GB RAM, sharp' },
  { value: '1536', label: '1536 — ~16+ GB VRAM, heavy, slow' },
];

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
  const [resolution, setResolution] = useState('512');
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
    const comma = imageDataUrl.indexOf(',');
    const b64 = comma >= 0 ? imageDataUrl.slice(comma + 1) : imageDataUrl;

    const ready = await runPreFlight('model3d', { modelName: selectedModel, modelsData, onError: showError });
    if (!ready) return;

    try {
      const response = await serverFetch('/3d/generations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, image: b64, resolution }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
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
          <div className="chat-message" style={{ height: '440px', maxWidth: '100%' }}>
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
          <div className="model-loading-indicator"><span className="model-loading-text">Reconstructing 3D mesh (this can take a couple of minutes)...</span></div>
        )}
      </div>

      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <ImagePreviewList
            images={imageDataUrl ? [imageDataUrl] : []}
            onRemove={() => setImageDataUrl(null)}
            altPrefix="Input"
          />
          <div className="chat-input model3d-hint" style={{ opacity: 0.7, pointerEvents: 'none' }}>
            {imageDataUrl ? 'Ready — press generate to reconstruct a 3D mesh.' : 'Attach an image to reconstruct into a 3D model.'}
          </div>
          <InferenceControls
            isBusy={isBusy}
            isInferring={isInferring}
            stoppable={false}
            onSend={handleGenerate}
            sendDisabled={!imageDataUrl}
            sendTitle="Generate 3D"
            modelSelector={<ModelSelector disabled={isBusy} />}
            leftControls={
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handlePickImage}
                  style={{ display: 'none' }}
                />
                <button
                  className="image-upload-button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isBusy}
                  title={imageDataUrl ? 'Change image' : 'Upload image'}
                >
                  <ImageUploadIcon />
                </button>
                <select
                  className="model3d-res-select"
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value)}
                  disabled={isBusy}
                  title="Cascade resolution (geometry detail vs VRAM/RAM cost)"
                  style={{ fontSize: '0.85em', maxWidth: '320px' }}
                >
                  {RESOLUTIONS.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
                </select>
              </>
            }
          />
        </div>
      </div>
    </>
  );
};

export default Model3DPanel;
