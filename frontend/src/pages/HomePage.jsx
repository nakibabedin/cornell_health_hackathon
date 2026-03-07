import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchUsers, createUser } from '../api'

const trendBadge = {
  improving: { text: 'Improving', cls: 'bg-green-100 text-green-700' },
  stable: { text: 'Stable', cls: 'bg-gray-100 text-gray-600' },
  worsening: { text: 'Worsening', cls: 'bg-red-100 text-red-700' },
  insufficient_data: { text: 'New', cls: 'bg-blue-100 text-blue-600' },
}

export default function HomePage() {
  const [users, setUsers] = useState([])
  const [newName, setNewName] = useState('')
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    fetchUsers().then(u => { setUsers(u); setLoading(false) })
  }, [])

  const handleCreate = async (e) => {
    e.preventDefault()
    if (!newName.trim()) return
    const user = await createUser(newName.trim())
    setUsers(prev => [...prev, { ...user, session_count: 0 }])
    setNewName('')
  }

  if (loading) {
    return <p className="text-gray-500 text-center mt-12">Loading users...</p>
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-gray-900">Select Patient</h2>
        <p className="text-gray-500 mt-1">Choose a patient to start a voice recording session or view their dashboard.</p>
      </div>

      {users.length > 0 && (
        <div className="grid gap-3">
          {users.map(u => {
            const badge = trendBadge[u.last_trend] || trendBadge.insufficient_data
            return (
              <div key={u.id} className="bg-white rounded-lg border border-gray-200 p-4 flex items-center justify-between">
                <div>
                  <h3 className="font-medium text-gray-900">{u.name}</h3>
                  <div className="flex items-center gap-3 mt-1 text-sm text-gray-500">
                    <span>{u.session_count || 0} sessions</span>
                    {u.last_score != null && (
                      <span>Score: {u.last_score}</span>
                    )}
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${badge.cls}`}>
                      {badge.text}
                    </span>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => navigate(`/record/${u.id}`)}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700"
                  >
                    Record
                  </button>
                  <button
                    onClick={() => navigate(`/dashboard/${u.id}`)}
                    className="bg-white text-gray-700 px-4 py-2 rounded-lg text-sm font-medium border border-gray-300 hover:bg-gray-50"
                  >
                    Dashboard
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <form onSubmit={handleCreate} className="flex gap-2">
        <input
          type="text"
          value={newName}
          onChange={e => setNewName(e.target.value)}
          placeholder="New patient name..."
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          className="bg-gray-900 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-gray-800"
        >
          Add Patient
        </button>
      </form>
    </div>
  )
}
