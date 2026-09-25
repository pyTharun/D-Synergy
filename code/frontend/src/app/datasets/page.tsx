'use client'

import { useState, useEffect } from 'react'
import { api, Dataset, DatasetStats, DatasetSample } from '@/lib/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selected, setSelected] = useState('')
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [rows, setRows] = useState<DatasetSample[]>([])
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState('')
  const [loadingStats, setLoadingStats] = useState(false)
  const [loadingRows, setLoadingRows] = useState(false)

  useEffect(() => {
    api.listDatasets().then((ds) => {
      const avail = ds.filter((d) => d.csv_available)
      setDatasets(avail)
      if (avail.length > 0) setSelected(avail[0].key)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!selected) return
    setLoadingStats(true)
    setStats(null)
    api.datasetStats(selected)
      .then(setStats)
      .catch(() => {})
      .finally(() => setLoadingStats(false))
  }, [selected])

  useEffect(() => {
    if (!selected) return
    setLoadingRows(true)
    api.datasetSamples(selected, page, 50, search)
      .then((r) => {
        setRows(r.rows)
        setTotalPages(r.total_pages)
        setTotal(r.total)
      })
      .catch(() => {})
      .finally(() => setLoadingRows(false))
  }, [selected, page, search])

  // Build histogram data for recharts
  const histData = stats?.score_distribution
    ? stats.score_distribution.counts.map((count, i) => ({
        bin: stats.score_distribution.bin_edges[i]?.toFixed(1) ?? '',
        count,
      }))
    : []

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Dataset Explorer</h1>
        <p className="page-subtitle">Browse and inspect drug-combination synergy datasets.</p>
      </div>

      {/* Dataset Tabs */}
      <div className="tabs">
        {datasets.map((d) => (
          <button
            key={d.key}
            className={`tab ${selected === d.key ? 'active' : ''}`}
            onClick={() => { setSelected(d.key); setPage(1); setSearch('') }}
          >
            {d.display_name}
            <span className={`badge ${d.model_trained ? 'badge-teal' : 'badge-muted'}`}
              style={{ marginLeft: 6, fontSize: 9, padding: '1px 6px' }}>
              {d.model_trained ? 'trained' : 'untrained'}
            </span>
          </button>
        ))}
      </div>

      {/* Stats Cards */}
      {loadingStats ? (
        <div className="stat-grid">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="skeleton" style={{ height: 96, borderRadius: 12 }} />
          ))}
        </div>
      ) : stats ? (
        <div className="stat-grid" style={{ marginBottom: 24 }}>
          <MiniStat label="Total Rows" value={stats.total_rows.toLocaleString()} />
          <MiniStat label="Valid Rows" value={stats.valid_rows.toLocaleString()} />
          <MiniStat label="Drug Pairs" value={stats.unique_drug_pairs.toLocaleString()} />
          <MiniStat label="Cell Lines" value={stats.unique_cell_lines.toLocaleString()} />
          {stats.score_stats.mean !== null && (
            <>
              <MiniStat label="Mean Score" value={stats.score_stats.mean.toFixed(2)} />
              <MiniStat label="Median Score" value={stats.score_stats.median?.toFixed(2) ?? '—'} />
              <MiniStat label="Min Score" value={stats.score_stats.min?.toFixed(2) ?? '—'} />
              <MiniStat label="Max Score" value={stats.score_stats.max?.toFixed(2) ?? '—'} />
            </>
          )}
        </div>
      ) : null}

      {/* Score Distribution Chart */}
      {stats && histData.length > 0 && (
        <div className="chart-container" style={{ marginBottom: 32 }}>
          <div className="chart-title">Synergy Score Distribution</div>
          <div className="chart-subtitle">Histogram of synergy scores in {stats.display_name}</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={histData} margin={{ top: 0, right: 0, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,45,66,0.7)" />
              <XAxis dataKey="bin" tick={{ fill: '#8ba3c7', fontSize: 10 }} interval={4} />
              <YAxis tick={{ fill: '#8ba3c7', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#0d1422', border: '1px solid #1e2d42', borderRadius: 8, color: '#f0f6ff' }}
                labelFormatter={(l) => `Score ≈ ${l}`}
                formatter={(v: unknown) => [v, 'Count']}
              />
              <Bar dataKey="count" fill="#14b8a6" radius={[3, 3, 0, 0]} opacity={0.85} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Search + Table */}
      <div className="card" style={{ padding: 0 }}>
        <div style={{ padding: '20px 20px 0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
            <h2 style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
              Sample Records
              {!loadingRows && <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--text-muted)', marginLeft: 8 }}>({total.toLocaleString()} total)</span>}
            </h2>
            <div className="search-container" style={{ minWidth: 280 }}>
              <svg className="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
              </svg>
              <input
                type="text"
                className="form-input search-input"
                placeholder="Search cell line or drug..."
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(1) }}
              />
            </div>
          </div>
        </div>

        <div className="table-container" style={{ borderRadius: 0, borderLeft: 0, borderRight: 0, borderBottom: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Drug 1 (SMILES)</th>
                <th>Drug 2 (SMILES)</th>
                <th>Cell Line</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {loadingRows ? (
                [...Array(8)].map((_, i) => (
                  <tr key={i}>
                    {[...Array(4)].map((_, j) => (
                      <td key={j}><div className="skeleton" style={{ height: 14, borderRadius: 4 }} /></td>
                    ))}
                  </tr>
                ))
              ) : rows.map((row, i) => (
                <tr key={i}>
                  <td title={row.drugname1_full}>
                    <span className="mono">{row.drugname1}</span>
                  </td>
                  <td title={row.drugname2_full}>
                    <span className="mono">{row.drugname2}</span>
                  </td>
                  <td>
                    <span className="badge badge-cyan" style={{ fontSize: 10 }}>{row.cellline}</span>
                  </td>
                  <td style={{ color: row.score > 0 ? 'var(--accent-teal)' : 'var(--accent-rose)', fontWeight: 600 }}>
                    {row.score.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="pagination" style={{ padding: '16px 0' }}>
            <button className="pagination-btn" onClick={() => setPage(1)} disabled={page === 1}>«</button>
            <button className="pagination-btn" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1}>‹</button>
            {[...Array(Math.min(7, totalPages))].map((_, i) => {
              const p = Math.max(1, Math.min(totalPages - 6, page - 3)) + i
              return (
                <button key={p} className={`pagination-btn ${p === page ? 'active' : ''}`} onClick={() => setPage(p)}>
                  {p}
                </button>
              )
            })}
            <button className="pagination-btn" onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page === totalPages}>›</button>
            <button className="pagination-btn" onClick={() => setPage(totalPages)} disabled={page === totalPages}>»</button>
          </div>
        )}
      </div>
    </>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-card" style={{ padding: 16 }}>
      <div className="stat-label" style={{ marginBottom: 4 }}>{label}</div>
      <div className="stat-value" style={{ fontSize: 20 }}>{value}</div>
    </div>
  )
}
