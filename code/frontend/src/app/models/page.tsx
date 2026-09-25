'use client'

import { useState, useEffect } from 'react'
import { api, Dataset, ModelMetrics } from '@/lib/api'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar,
} from 'recharts'

export default function ModelsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selected, setSelected] = useState('')
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [actuals, setActuals] = useState<{ actual: number[]; predicted: number[] } | null>(null)
  const [errors, setErrors] = useState<{ bin_edges: number[]; counts: number[] } | null>(null)
  const [allModels, setAllModels] = useState<ModelMetrics[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.listDatasets().then((ds) => {
      const trained = ds.filter((d) => d.model_trained)
      setDatasets(trained)
      if (trained.length > 0) setSelected(trained[0].key)
    })
    api.listModels().then(setAllModels).catch(() => {})
  }, [])

  useEffect(() => {
    if (!selected) return
    setLoading(true)
    setMetrics(null); setActuals(null); setErrors(null)

    Promise.all([
      api.modelMetrics(selected),
      api.modelActuals(selected).catch(() => null),
      api.modelErrors(selected).catch(() => null),
    ]).then(([m, a, e]) => {
      setMetrics(m)
      if (a) setActuals({ actual: a.actual, predicted: a.predicted })
      if (e) setErrors({ bin_edges: e.bin_edges, counts: e.counts })
    }).catch(() => {}).finally(() => setLoading(false))
  }, [selected])

  // Build scatter data
  const scatterData = actuals
    ? actuals.actual.map((a, i) => ({ x: a, y: actuals.predicted[i] }))
    : []

  // Build error histogram
  const errorHistData = errors
    ? errors.counts.map((count, i) => ({
        bin: errors.bin_edges[i]?.toFixed(1) ?? '',
        count,
      }))
    : []

  const allActualVals = actuals ? actuals.actual : []
  const domainMin = allActualVals.length > 0 ? Math.floor(Math.min(...allActualVals)) : -50
  const domainMax = allActualVals.length > 0 ? Math.ceil(Math.max(...allActualVals)) : 100

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Model Performance</h1>
        <p className="page-subtitle">
          Actual vs predicted synergy scores and regression metrics from real training runs.
        </p>
      </div>

      {datasets.length === 0 && (
        <div className="alert alert-warning">
          <span>⚠</span>
          <span>No trained models found. Go to <a href="/train" style={{ color: 'var(--accent-teal)' }}>Train Model</a> first.</span>
        </div>
      )}

      {/* Dataset Tabs */}
      {datasets.length > 0 && (
        <div className="tabs">
          {datasets.map((d) => (
            <button key={d.key} className={`tab ${selected === d.key ? 'active' : ''}`}
              onClick={() => setSelected(d.key)}>
              {d.display_name}
            </button>
          ))}
        </div>
      )}

      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--text-muted)', marginBottom: 24 }}>
          <span className="spinner" />
          <span>Loading model data...</span>
        </div>
      )}

      {metrics && (
        <>
          {/* Metric Cards */}
          <div className="metrics-grid">
            <MetricBox name="MAE" value={metrics.metrics?.MAE} desc="Mean Absolute Error" />
            <MetricBox name="RMSE" value={metrics.metrics?.RMSE} desc="Root Mean Square Error" />
            <MetricBox name="MSE" value={metrics.metrics?.MSE} desc="Mean Square Error" />
            <MetricBox name="R²" value={metrics.metrics?.R2} desc="Coefficient of Determination" color={
              metrics.metrics?.R2 > 0.7 ? 'var(--accent-emerald)' :
              metrics.metrics?.R2 > 0.4 ? 'var(--accent-amber)' : 'var(--accent-rose)'
            } />
          </div>

          {/* Model Info */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 32, fontSize: 13, color: 'var(--text-secondary)' }}>
              {[
                ['Model', metrics.model_name],
                ['Dataset', metrics.display_name],
                ['Training Samples', metrics.training_samples?.toLocaleString()],
                ['Testing Samples', metrics.testing_samples?.toLocaleString()],
                ['Morgan Radius', metrics.morgan_radius],
                ['Fingerprint Size', `${metrics.fingerprint_size} bits`],
                ['Trained At', metrics.trained_at ? new Date(metrics.trained_at).toLocaleString() : '—'],
              ].map(([k, v]) => (
                <div key={String(k)}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 3 }}>{k}</div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Charts */}
          <div className="grid-2">
            {/* Scatter: Actual vs Predicted */}
            <div className="chart-container">
              <div className="chart-title">Actual vs. Predicted Synergy Score</div>
              <div className="chart-subtitle">
                Each point is a test sample. Perfect predictions lie on the diagonal line.
              </div>
              {scatterData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,45,66,0.7)" />
                    <XAxis
                      type="number" dataKey="x" name="Actual"
                      label={{ value: 'Actual Score', position: 'insideBottom', offset: -10, fill: '#8ba3c7', fontSize: 11 }}
                      tick={{ fill: '#8ba3c7', fontSize: 10 }}
                      domain={[domainMin, domainMax]}
                    />
                    <YAxis
                      type="number" dataKey="y" name="Predicted"
                      label={{ value: 'Predicted Score', angle: -90, position: 'insideLeft', fill: '#8ba3c7', fontSize: 11 }}
                      tick={{ fill: '#8ba3c7', fontSize: 10 }}
                      domain={[domainMin, domainMax]}
                    />
                    <Tooltip
                      contentStyle={{ background: '#0d1422', border: '1px solid #1e2d42', borderRadius: 8, color: '#f0f6ff', fontSize: 12 }}
                      formatter={(v: unknown, n: string) => [Number(v).toFixed(2), n]}
                    />
                    {/* Perfect prediction line */}
                    <ReferenceLine
                      segment={[{ x: domainMin, y: domainMin }, { x: domainMax, y: domainMax }]}
                      stroke="#14b8a6" strokeDasharray="4 4" strokeOpacity={0.5}
                    />
                    <Scatter data={scatterData} fill="#38bdf8" opacity={0.5} r={2} />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                  No actuals data available. Re-train to generate.
                </div>
              )}
            </div>

            {/* Error Distribution Histogram */}
            <div className="chart-container">
              <div className="chart-title">Prediction Error Distribution</div>
              <div className="chart-subtitle">Histogram of (actual − predicted) errors</div>
              {errorHistData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={errorHistData} margin={{ top: 10, right: 10, bottom: 20, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,45,66,0.7)" />
                    <XAxis dataKey="bin" tick={{ fill: '#8ba3c7', fontSize: 9 }} interval={4}
                      label={{ value: 'Error', position: 'insideBottom', offset: -10, fill: '#8ba3c7', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#8ba3c7', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: '#0d1422', border: '1px solid #1e2d42', borderRadius: 8, color: '#f0f6ff' }}
                      formatter={(v: unknown) => [v, 'Count']}
                    />
                    <Bar dataKey="count" fill="#8b5cf6" radius={[3, 3, 0, 0]} opacity={0.8} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                  No error data available.
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* All Models Comparison Table */}
      {allModels.length > 1 && (
        <div style={{ marginTop: 32 }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>
            Model Comparison
          </h2>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Model</th>
                  <th>Train Samples</th>
                  <th>Test Samples</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>R²</th>
                  <th>Trained At</th>
                </tr>
              </thead>
              <tbody>
                {allModels.map((m) => (
                  <tr key={m.dataset}>
                    <td><span className="badge badge-teal" style={{ fontSize: 10 }}>{m.display_name}</span></td>
                    <td style={{ color: 'var(--text-primary)', fontFamily: 'inherit', fontSize: 13 }}>{m.model_name}</td>
                    <td>{m.training_samples?.toLocaleString()}</td>
                    <td>{m.testing_samples?.toLocaleString()}</td>
                    <td style={{ color: 'var(--accent-amber)' }}>{m.metrics?.MAE}</td>
                    <td style={{ color: 'var(--accent-cyan)' }}>{m.metrics?.RMSE}</td>
                    <td style={{ color: m.metrics?.R2 > 0.5 ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontWeight: 700 }}>
                      {m.metrics?.R2}
                    </td>
                    <td style={{ fontSize: 11 }}>{m.trained_at ? new Date(m.trained_at).toLocaleDateString() : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}

function MetricBox({ name, value, desc, color = 'var(--accent-teal)' }: {
  name: string; value: number; desc: string; color?: string
}) {
  return (
    <div className="metric-card">
      <div className="metric-name">{name}</div>
      <div className="metric-value" style={{ color }}>
        {value !== undefined && value !== null ? value.toFixed(4) : '—'}
      </div>
      <div className="metric-desc">{desc}</div>
    </div>
  )
}
