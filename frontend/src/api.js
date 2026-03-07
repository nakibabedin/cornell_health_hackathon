const BASE = '/api/v1';

export async function fetchUsers() {
  const res = await fetch(`${BASE}/users`);
  return res.json();
}

export async function createUser(name) {
  const res = await fetch(`${BASE}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  return res.json();
}

export async function fetchTimeline(userId) {
  const res = await fetch(`${BASE}/users/${userId}/timeline`);
  return res.json();
}

export async function fetchSessions(userId) {
  const res = await fetch(`${BASE}/users/${userId}/sessions`);
  return res.json();
}

export async function analyzeAudio(userId, audioBlob) {
  const form = new FormData();
  form.append('user_id', userId);
  form.append('file', audioBlob, 'recording.webm');
  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Analysis failed');
  }
  return res.json();
}
