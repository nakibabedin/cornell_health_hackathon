import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceArea,
} from 'recharts'

function formatDate(dateStr) {
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-medium text-gray-900">{formatDate(d.date)}</p>
      <p className="text-gray-600">Score: <span className="font-semibold">{d.score}</span></p>
      {d.trend && <p className="text-gray-500 capitalize">Trend: {d.trend.replace('_', ' ')}</p>}
    </div>
  )
}

export default function TrendChart({ timeline }) {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-400">
        No sessions recorded yet. Start a recording to see your timeline.
      </div>
    )
  }

  const data = timeline.map(t => ({
    ...t,
    dateLabel: formatDate(t.date),
  }))

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-sm font-medium text-gray-600 mb-3">PD Voice Index Over Time</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          {/* Color zones */}
          <ReferenceArea y1={0} y2={30} fill="#dcfce7" fillOpacity={0.4} />
          <ReferenceArea y1={30} y2={60} fill="#fef9c3" fillOpacity={0.4} />
          <ReferenceArea y1={60} y2={100} fill="#fee2e2" fillOpacity={0.4} />

          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="dateLabel"
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ fill: '#3b82f6', r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-4 mt-2 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="w-3 h-2 bg-green-200 rounded" /> Low concern (0-30)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-2 bg-yellow-200 rounded" /> Moderate (30-60)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-2 bg-red-200 rounded" /> High concern (60-100)
        </span>
      </div>
    </div>
  )
}
