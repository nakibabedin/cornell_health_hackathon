const scoreColor = (score) => {
  if (score < 30) return { bg: 'bg-green-50', text: 'text-green-700', ring: 'ring-green-200' }
  if (score < 60) return { bg: 'bg-yellow-50', text: 'text-yellow-700', ring: 'ring-yellow-200' }
  return { bg: 'bg-red-50', text: 'text-red-700', ring: 'ring-red-200' }
}

const trendArrow = {
  improving: { symbol: '\u2193', label: 'Improving', cls: 'text-green-600' },
  stable: { symbol: '\u2192', label: 'Stable', cls: 'text-gray-500' },
  worsening: { symbol: '\u2191', label: 'Worsening', cls: 'text-red-600' },
  insufficient_data: { symbol: '\u2014', label: 'First sessions', cls: 'text-blue-500' },
}

const featureLabels = {
  'HNR': 'Harmonics-to-Noise Ratio',
  'NHR': 'Noise-to-Harmonics Ratio',
  'MDVP:Jitter(%)': 'Pitch Variation (Jitter)',
  'MDVP:Shimmer': 'Amplitude Variation (Shimmer)',
  'MDVP:Shimmer(dB)': 'Shimmer (dB)',
  'MDVP:Fo(Hz)': 'Fundamental Frequency',
}

export default function ScoreCard({ result }) {
  const colors = scoreColor(result.score)
  const trend = trendArrow[result.trend] || trendArrow.insufficient_data

  return (
    <div className={`rounded-lg border p-6 space-y-4 ${colors.bg} ring-1 ${colors.ring}`}>
      <div className="text-center">
        <p className="text-sm text-gray-500 uppercase tracking-wide font-medium">PD Voice Index</p>
        <p className={`text-5xl font-bold mt-1 ${colors.text}`}>{result.score}</p>
        <p className={`text-sm font-medium mt-2 ${trend.cls}`}>
          {trend.symbol} {trend.label}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-3 text-center text-sm">
        <div className="bg-white/60 rounded p-2">
          <p className="text-gray-400">PD Probability</p>
          <p className="font-semibold text-gray-700">{(result.pd_probability * 100).toFixed(0)}%</p>
        </div>
        <div className="bg-white/60 rounded p-2">
          <p className="text-gray-400">UPDRS Estimate</p>
          <p className="font-semibold text-gray-700">{result.updrs_estimate}</p>
        </div>
        <div className="bg-white/60 rounded p-2">
          <p className="text-gray-400">Confidence</p>
          <p className="font-semibold text-gray-700">{(result.confidence * 100).toFixed(0)}%</p>
        </div>
      </div>

      {result.top_changed_features && result.top_changed_features.length > 0 && (
        <div>
          <p className="text-sm font-medium text-gray-600 mb-2">Top Changed Features</p>
          <div className="space-y-1">
            {result.top_changed_features.slice(0, 3).map((f, i) => (
              <div key={i} className="flex justify-between text-sm bg-white/60 rounded px-3 py-1.5">
                <span className="text-gray-600">
                  {featureLabels[f.feature] || f.feature}
                </span>
                <span className={f.direction === 'worse' ? 'text-red-600' : 'text-green-600'}>
                  {f.direction === 'worse' ? '+' : ''}{f.deviation_pct}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {!result.baseline_established && (
        <p className="text-xs text-blue-500 text-center">
          Baseline not yet established. {3 - (result.session_count || 0)} more sessions needed.
        </p>
      )}
    </div>
  )
}
