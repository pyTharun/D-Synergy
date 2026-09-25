'use client'

import { useState, useEffect } from 'react'
import { api, Dataset, PredictResponse } from '@/lib/api'

export default function PredictPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [cellLines, setCellLines] = useState<string[]>([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [drug1, setDrug1] = useState('')
  const [drug2, setDrug2] = useState('')
  const [cellline, setCellline] = useState('')
  const [cellSearch, setCellSearch] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingCells, setLoadingCells] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listDatasets().then((ds) => {
      const trained = ds.filter((d) => d.model_trained)
      setDatasets(trained)
      if (trained.length > 0) setSelectedDataset(trained[0].key)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!selectedDataset) return
    setLoadingCells(true)
    setCellLines([])
    setCellline('')
    api.cellLines(selectedDataset)
      .then((r) => setCellLines(r.cell_lines))
      .catch(() => setCellLines([]))
      .finally(() => setLoadingCells(false))
  }, [selectedDataset])

  const filteredCells = cellLines.filter((c) =>
    c.toLowerCase().includes(cellSearch.toLowerCase())
  )

  async function handlePredict(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setResult(null)
    setLoading(true)

    try {
      const res = await api.predict({ dataset: selectedDataset, drug1, drug2, cellline })
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  const trainedDatasets = datasets.filter((d) => d.model_trained)

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Predict Drug Synergy</h1>
        <p className="page-subtitle">
          Enter two drug SMILES strings and a cancer cell line to predict the synergy score.
        </p>
      </div>

      {trainedDatasets.length === 0 && !loading && (
        <div className="alert alert-warning">
          <span>⚠</span>
          <span>No trained models found. Please go to <a href="/train" style={{ color: 'var(--accent-teal)' }}>Train Model</a> first.</span>
        </div>
      )}

      <div className="grid-2">
        {/* Prediction Form */}
        <div className="card">
          <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 20, color: 'var(--text-primary)' }}>
            Prediction Inputs
          </h2>

          <form onSubmit={handlePredict}>
            {/* Dataset */}
            <div className="form-group">
              <label className="form-label">Dataset</label>
              <select
                id="predict-dataset"
                className="form-select"
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                required
              >
                {trainedDatasets.length === 0 && <option value="">No trained models</option>}
                {trainedDatasets.map((d) => (
                  <option key={d.key} value={d.key}>{d.display_name}</option>
                ))}
              </select>
              <span className="form-hint">Select a dataset with a trained model</span>
            </div>

            {/* Drug 1 */}
            <div className="form-group">
              <label className="form-label">Drug 1 — SMILES</label>
              <textarea
                id="predict-drug1"
                className="form-textarea mono"
                rows={3}
                value={drug1}
                onChange={(e) => setDrug1(e.target.value)}
                placeholder="e.g. CC(C)CC(NC(=O)...)B(O)O"
                required
                style={{ resize: 'vertical', fontSize: 12 }}
              />
              <span className="form-hint">Full SMILES molecular structure string</span>
            </div>

            {/* Drug 2 */}
            <div className="form-group">
              <label className="form-label">Drug 2 — SMILES</label>
              <textarea
                id="predict-drug2"
                className="form-textarea mono"
                rows={3}
                value={drug2}
                onChange={(e) => setDrug2(e.target.value)}
                placeholder="e.g. Cc1nc(Nc2ncc(...)s2)cc(N2CCN...)n1"
                required
                style={{ resize: 'vertical', fontSize: 12 }}
              />
              <span className="form-hint">Full SMILES molecular structure string</span>
            </div>

            {/* Cell Line */}
            <div className="form-group">
              <label className="form-label">Cancer Cell Line</label>
              <div className="search-container" style={{ marginBottom: 8 }}>
                <svg className="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
                </svg>
                <input
                  type="text"
                  className="form-input search-input"
                  placeholder="Search cell lines..."
                  value={cellSearch}
                  onChange={(e) => setCellSearch(e.target.value)}
                  style={{ marginBottom: 0 }}
                />
              </div>
              <select
                id="predict-cellline"
                className="form-select"
                value={cellline}
                onChange={(e) => setCellline(e.target.value)}
                required
                size={5}
                style={{ height: 120 }}
              >
                {loadingCells && <option disabled>Loading cell lines...</option>}
                {filteredCells.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
              <span className="form-hint">
                {loadingCells ? 'Loading...' : `${filteredCells.length} cell lines from training data`}
              </span>
            </div>

            {error && (
              <div className="alert alert-error">
                <span>✕</span>
                <span>{error}</span>
              </div>
            )}

            <button
              id="predict-submit"
              type="submit"
              className="btn btn-primary btn-lg"
              disabled={loading || !selectedDataset || !drug1 || !drug2 || !cellline}
              style={{ width: '100%' }}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 16, height: 16 }} />
                  Predicting...
                </>
              ) : (
                '🔬 Predict Synergy'
              )}
            </button>
          </form>
        </div>

        {/* Result Panel */}
        <div>
          {result ? (
            <div className="result-card">
              <div className="result-label">Predicted Synergy Score</div>
              <div className="result-score">
                {result.predicted_score.toFixed(2)}
              </div>
              <span className={`badge ${result.predicted_score > 0 ? 'badge-teal' : 'badge-rose'}`}>
                {result.predicted_score > 10 ? '↑ Strong Synergy' :
                 result.predicted_score > 0  ? '↑ Mild Synergy'  : '↓ Antagonism'}
              </span>
              <p className="result-interpretation">{result.interpretation}</p>

              <div className="divider" style={{ margin: '20px 0' }} />

              <div className="result-meta">
                <div className="result-meta-item">
                  <span className="result-meta-label">Dataset</span>
                  <span className="result-meta-value">{result.display_name}</span>
                </div>
                <div className="result-meta-item">
                  <span className="result-meta-label">Model</span>
                  <span className="result-meta-value">{result.model_name}</span>
                </div>
                <div className="result-meta-item">
                  <span className="result-meta-label">Morgan Radius</span>
                  <span className="result-meta-value">{result.morgan_radius}</span>
                </div>
                <div className="result-meta-item">
                  <span className="result-meta-label">Fingerprint</span>
                  <span className="result-meta-value">{result.fingerprint_size} bits</span>
                </div>
                <div className="result-meta-item">
                  <span className="result-meta-label">Cell Line</span>
                  <span className="result-meta-value">{result.cellline}</span>
                </div>
              </div>

              <div style={{
                marginTop: 24,
                padding: '12px 16px',
                background: 'rgba(245,158,11,0.08)',
                border: '1px solid rgba(245,158,11,0.2)',
                borderRadius: 8,
                fontSize: 11,
                color: 'var(--accent-amber)',
                textAlign: 'left',
              }}>
                ⚠ This is a predicted synergy score based on ML modeling of existing data.
                It is not a clinical recommendation. Do not use for medical decisions.
              </div>
            </div>
          ) : (
            <div className="card" style={{
              minHeight: 400,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px dashed var(--border-primary)',
              background: 'transparent',
            }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>🧪</div>
              <h3 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
                No prediction yet
              </h3>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', textAlign: 'center', maxWidth: 280 }}>
                Fill in the form on the left and click &quot;Predict Synergy&quot; to see the result here.
              </p>
            </div>
          )}

          {/* Quick Examples */}
          <div className="card" style={{ marginTop: 16 }}>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Example SMILES
            </h3>
            {[
              { name: 'Bortezomib', smiles: 'CC(C)CC(NC(=O)C(Cc1ccccc1)NC(=O)c1cnccn1)B(O)O' },
              { name: 'Selumetinib', smiles: 'Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1' },
            ].map((ex) => (
              <div key={ex.name} style={{
                padding: '10px 12px',
                background: 'var(--bg-secondary)',
                borderRadius: 8,
                marginBottom: 8,
                cursor: 'pointer',
              }}
                onClick={() => {
                  if (!drug1) setDrug1(ex.smiles)
                  else setDrug2(ex.smiles)
                }}
              >
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
                  {ex.name} <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>— click to use</span>
                </div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', wordBreak: 'break-all' }}>
                  {ex.smiles.substring(0, 50)}...
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
