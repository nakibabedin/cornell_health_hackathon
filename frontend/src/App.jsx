import { Routes, Route, Link } from 'react-router-dom'
import HomePage from './pages/HomePage'
import RecordPage from './pages/RecordPage'
import DashboardPage from './pages/DashboardPage'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <Link to="/" className="text-xl font-bold text-gray-900 no-underline">
            PD Voice Monitor
          </Link>
          <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">
            Research Use Only
          </span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/record/:userId" element={<RecordPage />} />
          <Route path="/dashboard/:userId" element={<DashboardPage />} />
        </Routes>
      </main>

      <footer className="text-center text-xs text-gray-400 py-4">
        This tool is for research purposes only. It is not a medical diagnostic device.
      </footer>
    </div>
  )
}
