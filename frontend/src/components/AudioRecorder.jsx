import { useState, useRef, useCallback, useEffect } from 'react'

export default function AudioRecorder({ onRecordingComplete }) {
  const [state, setState] = useState('idle') // idle | recording | done
  const [elapsed, setElapsed] = useState(0)
  const mediaRecorder = useRef(null)
  const chunks = useRef([])
  const timerRef = useRef(null)
  const canvasRef = useRef(null)
  const analyserRef = useRef(null)
  const animFrameRef = useRef(null)
  const streamRef = useRef(null)

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current
    const analyser = analyserRef.current
    if (!canvas || !analyser) return

    const ctx = canvas.getContext('2d')
    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    analyser.getByteTimeDomainData(dataArray)

    ctx.fillStyle = '#f9fafb'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = 2
    ctx.strokeStyle = '#3b82f6'
    ctx.beginPath()

    const sliceWidth = canvas.width / bufferLength
    let x = 0
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0
      const y = (v * canvas.height) / 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
      x += sliceWidth
    }
    ctx.lineTo(canvas.width, canvas.height / 2)
    ctx.stroke()

    animFrameRef.current = requestAnimationFrame(drawWaveform)
  }, [])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Set up analyser for waveform
      const audioCtx = new AudioContext()
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 2048
      source.connect(analyser)
      analyserRef.current = analyser

      const recorder = new MediaRecorder(stream)
      chunks.current = []
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data)
      }
      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: 'audio/webm' })
        onRecordingComplete(blob)
        setState('done')
        // Stop waveform
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      }

      mediaRecorder.current = recorder
      recorder.start()
      setState('recording')
      setElapsed(0)

      timerRef.current = setInterval(() => {
        setElapsed(prev => prev + 1)
      }, 1000)

      drawWaveform()
    } catch (err) {
      alert('Microphone access denied. Please allow microphone access and try again.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
      mediaRecorder.current.stop()
    }
    if (timerRef.current) clearInterval(timerRef.current)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
    }
  }

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
    }
  }, [])

  const formatTime = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
      <div className="text-center">
        <p className="text-gray-700 font-medium">
          Please say <span className="text-blue-600">"Ahhh"</span> in a sustained voice for at least 5 seconds
        </p>
        <p className="text-sm text-gray-400 mt-1">Hold a steady tone at your normal speaking pitch</p>
      </div>

      <canvas
        ref={canvasRef}
        width={600}
        height={100}
        className="w-full h-24 rounded border border-gray-100"
      />

      <div className="flex items-center justify-center gap-4">
        {state === 'idle' && (
          <button
            onClick={startRecording}
            className="bg-red-500 text-white px-6 py-3 rounded-full text-sm font-medium hover:bg-red-600 flex items-center gap-2"
          >
            <span className="w-3 h-3 bg-white rounded-full inline-block" />
            Start Recording
          </button>
        )}

        {state === 'recording' && (
          <>
            <span className="text-red-500 font-mono text-lg animate-pulse">
              {formatTime(elapsed)}
            </span>
            <button
              onClick={stopRecording}
              disabled={elapsed < 3}
              className={`px-6 py-3 rounded-full text-sm font-medium flex items-center gap-2 ${
                elapsed < 3
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-gray-900 text-white hover:bg-gray-800'
              }`}
            >
              <span className="w-3 h-3 bg-white rounded inline-block" />
              Stop
            </button>
            {elapsed < 3 && (
              <span className="text-xs text-gray-400">Min 3 seconds</span>
            )}
          </>
        )}

        {state === 'done' && (
          <p className="text-green-600 font-medium">Recording complete. Analyzing...</p>
        )}
      </div>
    </div>
  )
}
