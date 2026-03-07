import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import AudioRecorder from '../components/AudioRecorder'
import ScoreCard from '../components/ScoreCard'
import { analyzeAudio } from '../api'

export default function RecordPage() {
  const { userId } = useParams()
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  const handleRecordingComplete = async (blob) => {
    setAnalyzing(true)
    setError(null)
    try {
      const data = await analyzeAudio(userId, blob)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold text-gray-900">Voice Recording</h2>
        <Link
          to={`/dashboard/${userId}`}
          className="text-sm text-blue-600 hover:text-blue-700"
        >
          View Dashboard
        </Link>
      </div>

      {!result && !analyzing && (
        <AudioRecorder onRecordingComplete={handleRecordingComplete} />
      )}

      {analyzing && (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <div className="inline-block w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <p className="mt-4 text-gray-600 font-medium">Analyzing your voice...</p>
          <p className="text-sm text-gray-400 mt-1">Extracting features and running models</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <ScoreCard result={result} />
          <div className="flex gap-3">
            <button
              onClick={() => { setResult(null); setError(null) }}
              className="flex-1 bg-blue-600 text-white py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700"
            >
              Record Again
            </button>
            <Link
              to={`/dashboard/${userId}`}
              className="flex-1 bg-white text-gray-700 py-2.5 rounded-lg text-sm font-medium border border-gray-300 hover:bg-gray-50 text-center"
            >
              View Full Dashboard
            </Link>
          </div>
        </div>
      )}
    </div>
  )
}
