export default function AboutPage() {
  return (
    <>
      <div className="page-header">
        <h1 className="page-title">About This Project</h1>
        <p className="page-subtitle">
          A research tool for machine-learning based drug-combination synergy prediction.
        </p>
      </div>

      {/* Disclaimer */}
      <div className="alert alert-warning" style={{ marginBottom: 32 }}>
        <span>⚠</span>
        <strong>Research Tool Only.</strong>&nbsp;
        This application predicts synergy scores based on existing experimental datasets.
        Results are NOT clinical recommendations and should NOT be used for medical decisions.
      </div>

      <div className="grid-2">
        {/* Project Overview */}
        <div className="card">
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>Project Overview</h2>
          <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 12 }}>
            Cancer treatment often involves combining multiple drugs rather than using a single drug.
            Two drugs can interact in different ways — they can work better together (synergy), have
            approximately additive effects, or interfere with each other (antagonism).
          </p>
          <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 12 }}>
            Experimentally testing every possible drug combination across every cancer cell line is
            extremely expensive and time-consuming. This project uses machine learning to predict
            drug-combination synergy from existing experimental data.
          </p>
          <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            The core question: <em style={{ color: 'var(--text-primary)' }}>
              &ldquo;Given two drugs and a cancer cell line, what is the experimentally observed synergy score
              for that drug pair?&rdquo;
            </em>
          </p>
        </div>

        {/* Tech Stack */}
        <div className="card">
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>Technology Stack</h2>
          {[
            { category: 'ML Pipeline', items: ['Python 3.11', 'pandas', 'NumPy', 'scikit-learn', 'RDKit', 'joblib', 'XGBoost'] },
            { category: 'Backend', items: ['FastAPI', 'Uvicorn', 'Pydantic'] },
            { category: 'Frontend', items: ['Next.js 14', 'React 18', 'TypeScript', 'Recharts'] },
            { category: 'Environment', items: ['conda (conda-forge for RDKit)'] },
          ].map((group) => (
            <div key={group.category} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 8 }}>
                {group.category}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {group.items.map((item) => (
                  <span key={item} className="badge badge-cyan" style={{ fontSize: 11 }}>{item}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ML Approach */}
      <div className="card" style={{ marginTop: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 20, color: 'var(--text-primary)' }}>
          Machine Learning Approach
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
          {[
            {
              icon: '🧬',
              title: 'Morgan Fingerprints',
              desc: 'Each drug SMILES is converted to a 2048-bit Morgan fingerprint using RDKit (radius=2). This captures the chemical neighborhood of each atom up to 2 bonds away.',
            },
            {
              icon: '🏥',
              title: 'Cell-Line Encoding',
              desc: 'Cancer cell lines (e.g. ACH-000788) are one-hot encoded using scikit-learn\'s OneHotEncoder, fitted only on training data to prevent data leakage.',
            },
            {
              icon: '📊',
              title: 'Regression Problem',
              desc: 'The synergy score is a continuous value — so this is a regression problem, not classification. Metrics: MAE, RMSE, R².',
            },
            {
              icon: '⚗️',
              title: 'Feature Vector',
              desc: 'Drug 1 fingerprint (2048) + Drug 2 fingerprint (2048) + cell-line one-hot = ~4096+ features per sample.',
            },
          ].map((item) => (
            <div key={item.title} style={{ background: 'var(--bg-secondary)', borderRadius: 10, padding: 18 }}>
              <div style={{ fontSize: 28, marginBottom: 10 }}>{item.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Datasets */}
      <div className="card" style={{ marginTop: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>Datasets</h2>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
          Five drug-combination synergy datasets, all with SMILES-encoded drug names, cancer cell line identifiers, and continuous synergy scores.
        </p>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Dataset</th>
                <th>File</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['OncologyScreen', 'OncologyScreenLINCS_PRISM.csv', 'NCI Oncology Screen synergy measurements'],
                ['Oneil', 'OneilLINCS_PRISM.csv', "O'Neil et al. drug combination screen"],
                ['DrugComb', 'DrugComb_LINCS_PRISM.csv', 'DrugComb database entries'],
                ['DrugCombDB', 'DrugCombDBLINCS_PRISM.csv', 'DrugCombDB extended dataset'],
                ['Almanac', 'AlmanacLINCS_PRISM.csv', 'NCI-ALMANAC combination screen'],
              ].map(([name, file, desc]) => (
                <tr key={name}>
                  <td><span className="badge badge-teal" style={{ fontSize: 10 }}>{name}</span></td>
                  <td className="mono" style={{ fontSize: 11 }}>{file}</td>
                  <td style={{ fontSize: 12, fontFamily: 'inherit' }}>{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Limitations */}
      <div className="card" style={{ marginTop: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, color: 'var(--text-primary)' }}>
          ⚠ Important Limitations
        </h2>
        <ul style={{ paddingLeft: 20, color: 'var(--text-secondary)', fontSize: 13.5, lineHeight: 2 }}>
          <li>Predictions are based on patterns in existing experimental data, not mechanistic biology.</li>
          <li>A positive predicted score does not mean a drug combination will be clinically effective.</li>
          <li>The model can only predict for drug pairs and cell lines present in or similar to the training data.</li>
          <li>Random train/test splitting may lead to optimistic performance estimates (drug-pair split is more rigorous).</li>
          <li>Do not use this tool for medical decisions, prescriptions, or clinical recommendations.</li>
        </ul>
      </div>
    </>
  )
}
