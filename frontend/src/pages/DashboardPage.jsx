import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import TrendChart from '../components/TrendChart'
import { fetchTimeline, fetchSessions } from '../api'

const labelColors = {
  low_concern: 'bg-green-100 text-green-700',
  moderate_concern: 'bg-yellow-100 text-yellow-700',
  high_concern: 'bg-red-100 text-red-700',
}

export default function DashboardPage() {
  const { userId } = useParams()
  const [timeline, setTimeline] = useState([])
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetchTimeline(userId),
      fetchSessions(userId),
    ]).then(([tl, sess]) => {
      setTimeline(tl)
      setSessions(sess)
      setLoading(false)
    })
  }, [userId])

  if (loading) {
    return <p className="text-gray-500 text-center mt-12">Loading dashboard...</p>
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold text-gray-900">Patient Dashboard</h2>
        <Link
          to={`/record/${userId}`}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 no-underline"
        >
          New Recording
        </Link>
      </div>

      <TrendChart timeline={timeline} />

      {sessions.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-4 py-3 border-b border-gray-100">
            <h3 className="text-sm font-medium text-gray-600">Session History</h3>
          </div>
          <div className="divide-y divide-gray-100">
            {[...sessions].reverse().map((s, i) => (
              <div key={s.id || i} className="px-4 py-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    {new Date(s.created_at).toLocaleDateString('en-US', {
                      month: 'long', day: 'numeric', year: 'numeric'
                    })}
                  </p>
                  {s.trend && (
                    <p className="text-xs text-gray-400 capitalize mt-0.5">
                      {s.trend.replace('_', ' ')}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  {s.pd_score != null && (
                    <span className="text-lg font-semibold text-gray-700">{s.pd_score}</span>
                  )}
                  {s.label && (
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${labelColors[s.label] || ''}`}>
                      {s.label.replace('_', ' ')}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-800">
        <strong>Disclaimer:</strong> This tool is for research purposes only and is not a medical
        diagnostic device. Always consult with a qualified healthcare professional for medical advice.
      </div>
    </div>
  )
}
