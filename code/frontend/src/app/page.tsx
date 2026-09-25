'use client'

import { useState, useEffect } from 'react'
import { api, Dataset, ModelMetrics } from '@/lib/api'

export default function Dashboard() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [models, setModels] = useState<ModelMetrics[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadData() {
      try {
        const [ds, ms] = await Promise.all([api.listDatasets(), api.listModels()])
        setDatasets(ds)
        setModels(ms)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to connect to backend.')
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  const trainedCount = datasets.filter((d) => d.model_trained).length
  const availableCount = datasets.filter((d) => d.csv_available).length
  const bestR2 = models.length > 0
    ? Math.max(...models.map((m) => m.metrics?.R2 ?? -Infinity))
    : null

  return (
    <>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Drug Combination Synergy Predictor</h1>
        <p className="page-subtitle">
          Machine-learning based prediction of drug-combination synergy across cancer cell lines
          using Morgan molecular fingerprints and regression models.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="alert alert-error" style={{ marginBottom: 24 }}>
          <span>⚠</span>
          <span>
            <strong>Backend connection failed:</strong> {error}
            <br />
            <small>Make sure FastAPI is running on port 8000.</small>
          </span>
        </div>
      )}

      {/* Stat Cards */}
      <div className="stat-grid">
        <StatCard
          icon="🧬"
          iconClass="teal"
          label="Datasets"
          value={loading ? '—' : String(availableCount)}
          desc="CSV datasets available"
        />
        <StatCard
          icon="🤖"
          iconClass="purple"
          label="Trained Models"
          value={loading ? '—' : String(trainedCount)}
          desc={`of ${availableCount} datasets trained`}
        />
        <StatCard
          icon="⚗️"
          iconClass="cyan"
          label="Best R²"
          value={loading ? '—' : bestR2 !== null ? bestR2.toFixed(3) : 'N/A'}
          desc="Highest R² across trained models"
        />
        <StatCard
          icon="🔬"
          iconClass="green"
          label="Algorithm"
          value="ML"
          desc="Morgan fingerprints + regression"
        />
      </div>

      {/* Dataset Status Grid */}
      <div style={{ marginBottom: 32 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>
          Dataset Status
        </h2>
        {loading ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
            {[...Array(5)].map((_, i) => (
              <div key={i} className="skeleton" style={{ height: 100, borderRadius: 12 }} />
            ))}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
            {datasets.map((ds) => {
              const model = models.find((m) => m.dataset === ds.key)
              return (
                <div key={ds.key} className="card" style={{ padding: 20 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                    <div>
                      <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 15 }}>
                        {ds.display_name}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2, fontFamily: 'monospace' }}>
                        {ds.key}
                      </div>
                    </div>
                    <span className={`badge ${
                      ds.status === 'trained' ? 'badge-teal' :
                      ds.status === 'available' ? 'badge-cyan' : 'badge-rose'
                    }`}>
                      {ds.status}
                    </span>
                  </div>
                  {model && (
                    <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--text-muted)' }}>
                      <span>MAE <strong style={{ color: 'var(--accent-teal)' }}>{model.metrics?.MAE}</strong></span>
                      <span>R² <strong style={{ color: 'var(--accent-teal)' }}>{model.metrics?.R2}</strong></span>
                      <span>RMSE <strong style={{ color: 'var(--accent-teal)' }}>{model.metrics?.RMSE}</strong></span>
                    </div>
                  )}
                  {!model && ds.csv_available && (
                    <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      No model trained yet — go to <strong>Train Model</strong> to start.
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Quick Prediction CTA */}
      <div className="card card-glow" style={{ background: 'linear-gradient(135deg, rgba(20,184,166,0.08), rgba(56,189,248,0.05))' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <h3 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
              Ready to predict?
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', maxWidth: 480 }}>
              Enter two drug SMILES strings and a cancer cell line to get a predicted synergy score from the trained regression model.
            </p>
          </div>
          <a href="/predict" className="btn btn-primary btn-lg">
            Predict Synergy →
          </a>
        </div>
      </div>

      {/* Pipeline Overview */}
      <div style={{ marginTop: 32 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>
          ML Pipeline
        </h2>
        <div style={{ display: 'flex', gap: 0, overflowX: 'auto', paddingBottom: 8 }}>
          {[
            { step: '1', label: 'SMILES', sub: 'Drug 1 + Drug 2' },
            { step: '2', label: 'RDKit', sub: 'Parse molecules' },
            { step: '3', label: 'Morgan FP', sub: '2048 bits each' },
            { step: '4', label: 'Cell Line', sub: 'One-hot encode' },
            { step: '5', label: 'Feature Matrix', sub: '4096 + N features' },
            { step: '6', label: 'Regression', sub: 'Linear / RF / XGB' },
            { step: '7', label: 'Score', sub: 'MAE / RMSE / R²' },
          ].map((item, idx, arr) => (
            <div key={item.step} style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-primary)',
                borderRadius: 10,
                padding: '14px 18px',
                textAlign: 'center',
                minWidth: 110,
                flexShrink: 0,
              }}>
                <div style={{
                  width: 24, height: 24,
                  borderRadius: '50%',
                  background: 'rgba(20,184,166,0.15)',
                  color: 'var(--accent-teal)',
                  fontSize: 11, fontWeight: 700,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  margin: '0 auto 8px',
                }}>{item.step}</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 3 }}>{item.label}</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{item.sub}</div>
              </div>
              {idx < arr.length - 1 && (
                <div style={{ color: 'var(--text-muted)', fontSize: 18, padding: '0 4px', flexShrink: 0 }}>→</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

function StatCard({ icon, iconClass, label, value, desc }: {
  icon: string; iconClass: string; label: string; value: string; desc: string
}) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${iconClass}`}>
        <span style={{ fontSize: 18 }}>{icon}</span>
      </div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-desc">{desc}</div>
    </div>
  )
}
