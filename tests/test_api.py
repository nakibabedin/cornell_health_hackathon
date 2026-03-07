"""Tests for FastAPI endpoints."""

import os
import asyncio
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

from backend.database import DB_PATH, init_db

# Use a test database
TEST_DB = DB_PATH.parent / "test_pd_voice.db"


@pytest.fixture(autouse=True)
def use_test_db(monkeypatch):
    """Redirect database to a temporary test file."""
    import backend.database as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", TEST_DB)
    yield
    if TEST_DB.exists():
        os.remove(TEST_DB)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(use_test_db):
    from backend.main import app

    os.makedirs(TEST_DB.parent, exist_ok=True)
    await init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_create_user(client):
    resp = await client.post("/api/v1/users", json={"name": "Test User"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test User"
    assert "id" in data


@pytest.mark.anyio
async def test_list_users(client):
    await client.post("/api/v1/users", json={"name": "Alice"})
    await client.post("/api/v1/users", json={"name": "Bob"})

    resp = await client.get("/api/v1/users")
    assert resp.status_code == 200
    users = resp.json()
    assert len(users) >= 2
    names = [u["name"] for u in users]
    assert "Alice" in names
    assert "Bob" in names


@pytest.mark.anyio
async def test_create_session(client):
    user_resp = await client.post("/api/v1/users", json={"name": "Session User"})
    user_id = user_resp.json()["id"]

    resp = await client.post("/api/v1/sessions", json={"user_id": user_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == user_id
    assert "id" in data


@pytest.mark.anyio
async def test_get_user_sessions(client):
    user_resp = await client.post("/api/v1/users", json={"name": "Timeline User"})
    user_id = user_resp.json()["id"]

    await client.post("/api/v1/sessions", json={"user_id": user_id})
    await client.post("/api/v1/sessions", json={"user_id": user_id})

    resp = await client.get(f"/api/v1/users/{user_id}/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 2


@pytest.mark.anyio
async def test_timeline_empty(client):
    user_resp = await client.post("/api/v1/users", json={"name": "Empty User"})
    user_id = user_resp.json()["id"]

    resp = await client.get(f"/api/v1/users/{user_id}/timeline")
    assert resp.status_code == 200
    assert resp.json() == []
