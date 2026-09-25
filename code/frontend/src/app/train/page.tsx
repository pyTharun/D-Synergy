'use client'

import { useState, useEffect, useRef } from 'react'
import { api } from '@/lib/api'

const DATASET_OPTIONS = [
  { key: 'oncology',   name: 'OncologyScreen' },
  { key: 'oneil',      name: 'Oneil' },
  { key: 'drugcomb',   name: 'DrugComb' },
  { key: 'drugcombdb', name: 'DrugCombDB' },
  { key: 'almanac',    name: 'Almanac' },
]

const MODEL_OPTIONS = [
  { key: 'linear',            name: 'Linear Regression' },
  { key: 'random_forest',     name: 'Random Forest' },
  { key: 'adaboost',          name: 'AdaBoost' },
  { key: 'gradient_boosting', name: 'Gradient Boosting' },
]

type JobStatus = 'idle' | 'queued' | 'running' | 'completed' | 'failed'

export default function TrainPage() {
  const [dataset, setDataset]         = useState('oncology')
  const [modelType, setModelType]     = useState('linear')
  const [testSize, setTestSize]       = useState(0.20)
  const [radius, setRadius]           = useState(2)
  const [fpSize, setFpSize]           = useState(2048)
  const [randomState, setRandomState] = useState(42)

  const [status, setStatus] = useState<JobStatus>('idle')
  const [jobId, setJobId]   = useState<string | null>(null)
  const [message, setMessage] = useState('')
  const [result, setResult]   = useState<null | {
    metrics?: { MAE: number; RMSE: number; R2: number; MSE: number }
    training_samples?: number
    testing_samples?: number
    model_name?: string
    trained_at?: string
  }>(null)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Poll for training status
  useEffect(() => {
    if (!jobId || status === 'completed' || status === 'failed') {
      if (pollRef.current) clearInterval(pollRef.current)
      return
    }

    pollRef.current = setInterval(async () => {
      try {
        const s = await api.trainingStatus(jobId)
        setStatus(s.status)
        setMessage(s.message)
        if (s.status === 'completed' || s.status === 'failed') {
          setResult(s.result)
          if (pollRef.current) clearInterval(pollRef.current)
        }
      } catch {
        // ignore poll errors
      }
    }, 2000)

    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [jobId, status])

  async function handleTrain(e: React.FormEvent) {
    e.preventDefault()
    setStatus('queued')
    setMessage('Starting training job...')
    setResult(null)
    setJobId(null)

    try {
      const res = await api.startTraining({
        dataset,
        model_type: modelType,
        test_size: testSize,
        morgan_radius: radius,
        fp_size: fpSize,
        random_state: randomState,
      })
      setJobId(res.job_id)
      setStatus(res.status as JobStatus)
      setMessage(res.message)
    } catch (e) {
      setStatus('failed')
      setMessage(e instanceof Error ? e.message : 'Failed to start training.')
    }
  }

  const isRunning = status === 'queued' || status === 'running'

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Train Model</h1>
        <p className="page-subtitle">
          Configure and train a regression model on a selected dataset. Results are saved automatically.
        </p>
      </div>

      <div className="grid-2">
        {/* Training Form */}
        <div className="card">
          <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 20, color: 'var(--text-primary)' }}>
            Training Configuration
          </h2>

          <form onSubmit={handleTrain}>
            <div className="form-group">
              <label className="form-label">Dataset</label>
              <select id="train-dataset" className="form-select" value={dataset}
                onChange={(e) => setDataset(e.target.value)} disabled={isRunning}>
                {DATASET_OPTIONS.map((d) => (
                  <option key={d.key} value={d.key}>{d.name}</option>
                ))}
              </select>
              <span className="form-hint">Make sure the CSV is in the data/ directory</span>
            </div>

            <div className="form-group">
              <label className="form-label">Model</label>
              <select id="train-model" className="form-select" value={modelType}
                onChange={(e) => setModelType(e.target.value)} disabled={isRunning}>
                {MODEL_OPTIONS.map((m) => (
                  <option key={m.key} value={m.key}>{m.name}</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div className="form-group">
                <label className="form-label">Test Size</label>
                <input type="number" className="form-input" value={testSize} step={0.05} min={0.05} max={0.5}
                  onChange={(e) => setTestSize(parseFloat(e.target.value))} disabled={isRunning} />
                <span className="form-hint">{Math.round(testSize * 100)}% test</span>
              </div>
              <div className="form-group">
                <label className="form-label">Random State</label>
                <input type="number" className="form-input" value={randomState}
                  onChange={(e) => setRandomState(parseInt(e.target.value))} disabled={isRunning} />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div className="form-group">
                <label className="form-label">Morgan Radius</label>
                <select className="form-select" value={radius}
                  onChange={(e) => setRadius(parseInt(e.target.value))} disabled={isRunning}>
                  {[1, 2, 3, 4].map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Fingerprint Size</label>
                <select className="form-select" value={fpSize}
                  onChange={(e) => setFpSize(parseInt(e.target.value))} disabled={isRunning}>
                  {[512, 1024, 2048, 4096].map((s) => <option key={s} value={s}>{s} bits</option>)}
                </select>
              </div>
            </div>

            <button id="train-submit" type="submit" className="btn btn-primary btn-lg"
              disabled={isRunning} style={{ width: '100%' }}>
              {isRunning ? (
                <><span className="spinner" style={{ width: 16, height: 16 }} /> Training...</>
              ) : (
                '⚡ Train Model'
              )}
            </button>
          </form>

          {/* Warning */}
          <div className="alert alert-info" style={{ marginTop: 16 }}>
            <span>ℹ</span>
            <span style={{ fontSize: 12 }}>
              Training runs in the background. Large datasets (Almanac ~150k rows) may take several minutes.
              The page will update automatically when done.
            </span>
          </div>
        </div>

        {/* Status Panel */}
        <div>
          {status !== 'idle' && (
            <div className="card" style={{ marginBottom: 16 }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>
                Training Status
              </h3>

              {/* Status indicator */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                {isRunning && <span className="spinner" />}
                {status === 'completed' && <span style={{ color: 'var(--accent-emerald)', fontSize: 20 }}>✓</span>}
                {status === 'failed'    && <span style={{ color: 'var(--accent-rose)', fontSize: 20 }}>✕</span>}
                <span style={{
                  fontSize: 13,
                  color: status === 'completed' ? 'var(--accent-emerald)' :
                         status === 'failed'    ? 'var(--accent-rose)'   : 'var(--text-secondary)',
                }}>
                  {message}
                </span>
              </div>

              {/* Progress bar animation while running */}
              {isRunning && (
                <div className="progress-bar">
                  <div className="progress-fill" style={{
                    width: '100%',
                    animation: 'progressIndeterminate 1.5s ease-in-out infinite',
                  }} />
                </div>
              )}

              {/* Job ID */}
              {jobId && (
                <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)' }}>
                  Job ID: <span className="mono">{jobId}</span>
                </div>
              )}
            </div>
          )}

          {/* Result */}
          {status === 'completed' && result && (
            <div className="card card-glow" style={{
              background: 'linear-gradient(135deg, rgba(16,185,129,0.08), rgba(20,184,166,0.05))',
              animation: 'fadeInUp 0.4s ease',
            }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--accent-emerald)', marginBottom: 20 }}>
                ✓ Training Complete
              </h3>

              {result.metrics && (
                <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', marginBottom: 16 }}>
                  {[
                    ['MAE', result.metrics.MAE],
                    ['RMSE', result.metrics.RMSE],
                    ['MSE', result.metrics.MSE],
                    ['R²', result.metrics.R2],
                  ].map(([k, v]) => (
                    <div key={String(k)} className="metric-card">
                      <div className="metric-name">{k}</div>
                      <div className="metric-value" style={{ fontSize: 20 }}>
                        {typeof v === 'number' ? v.toFixed(4) : '—'}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div style={{ display: 'flex', gap: 24, fontSize: 12, color: 'var(--text-muted)', flexWrap: 'wrap' }}>
                {result.training_samples != null && (
                  <span>Training: <strong style={{ color: 'var(--text-secondary)' }}>{result.training_samples.toLocaleString()}</strong></span>
                )}
                {result.testing_samples != null && (
                  <span>Testing: <strong style={{ color: 'var(--text-secondary)' }}>{result.testing_samples.toLocaleString()}</strong></span>
                )}
                {result.model_name && (
                  <span>Model: <strong style={{ color: 'var(--text-secondary)' }}>{result.model_name}</strong></span>
                )}
              </div>

              <p style={{ marginTop: 16, fontSize: 12, color: 'var(--text-muted)' }}>
                Model saved to <span className="mono">models/{dataset}/</span>.
                You can now use it on the <a href="/predict" style={{ color: 'var(--accent-teal)' }}>Predict</a> page.
              </p>
            </div>
          )}

          {status === 'idle' && (
            <div className="card" style={{
              minHeight: 300,
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              border: '2px dashed var(--border-primary)', background: 'transparent',
            }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>⚙️</div>
              <h3 style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
                Configure and train
              </h3>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', maxWidth: 280 }}>
                Select a dataset and model type, then click &quot;Train Model&quot; to start. Results appear here.
              </p>
            </div>
          )}

          {/* Data Flow Reminder */}
          <div className="card" style={{ marginTop: 16, padding: 16 }}>
            <h3 style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: 12 }}>
              Pipeline
            </h3>
            {[
              'Load & clean CSV',
              'Validate SMILES',
              'Generate Morgan fingerprints',
              'One-hot encode cell lines',
              'Train/test split (80/20)',
              `Train ${MODEL_OPTIONS.find((m) => m.key === modelType)?.name ?? modelType}`,
              'Calculate MAE / RMSE / R²',
              'Save model.pkl + encoder.pkl',
            ].map((step, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                <span style={{
                  width: 18, height: 18, borderRadius: '50%',
                  background: 'rgba(20,184,166,0.1)', color: 'var(--accent-teal)',
                  fontSize: 10, fontWeight: 700,
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                }}>{i + 1}</span>
                <span>{step}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes progressIndeterminate {
          0%   { transform: translateX(-100%); width: 50%; }
          100% { transform: translateX(300%); width: 50%; }
        }
      `}</style>
    </>
  )
}
