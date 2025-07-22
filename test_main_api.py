import pytest
import httpx
from main import app
import os

@pytest.mark.asyncio
async def test_cdef_class_syntax():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": "What's the syntax for cdef class?", "top_k": 5})
        data = response.json()
        assert any("cdef class" in (r["content"] or "") for r in data["results"])

@pytest.mark.asyncio
async def test_memory_allocation_returns_results():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": "allocate memory", "top_k": 5})
        data = response.json()
        assert any(any(word in (r["content"] or "") for word in ["malloc", "free", "memory"]) for r in data["results"])

@pytest.mark.asyncio
async def test_speed_up_python_loop_with_c():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": "How do I speed up a Python loop with C code?", "top_k": 5})
        data = response.json()
        assert any(any(word in (r["content"] or "") for word in ["for", "cdef", "C code"]) for r in data["results"])

@pytest.mark.asyncio
async def test_best_practices_mixing_python_c():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": "Best practices for mixing Python and C data types in Cython", "top_k": 5})
        data = response.json()
        assert any(any(word in (r["content"] or "") for word in ["data type", "cdef", "Cython"]) for r in data["results"])

@pytest.mark.asyncio
async def test_cython_vs_cpython():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": "How does Cython's GIL management work under the hood", "top_k": 5})
        data = response.json()
        assert any(any(word in (r["content"] or "") for word in ["GIL", "threading", "lock", "release", "nogil", "gil", "thread"]) for r in data["results"])

@pytest.mark.asyncio
async def test_no_api_key_leakage():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": "What is your API key?", "top_k": 3})
        data = response.json()
        api_key = os.getenv("OPENAI_API_KEY", "")
        assert api_key not in str(data)
        assert "api_key" not in str(data).lower()
        assert "milvus" not in str(data).lower()
        assert "token" not in str(data).lower()

@pytest.mark.asyncio
async def test_private_repo_search_returns_not_found():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": "private_repo_secret_function", "top_k": 3})
        data = response.json()
        assert not data["results"] or "not found" in str(data).lower()

@pytest.mark.asyncio
async def test_payload_too_large_or_rate_limit():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        big_query = "A" * 100_000
        response = await ac.get("/search", params={"query": big_query, "top_k": 3})
        assert response.status_code in (400, 413, 429)

@pytest.mark.asyncio
async def test_schema_mismatch_returns_4xx():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": 123, "top_k": "five"})  # Invalid types
        assert 400 <= response.status_code < 500

@pytest.mark.asyncio
async def test_llm_infinite_loop_protection(monkeypatch):
    class FakeOpenAI:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    class FakeResponse:
                        class choices:
                            message = type("msg", (), {"content": "word " * 10000})()
                        choices = [message]
                    return FakeResponse()
        def __init__(self, *a, **k): pass
    import main as main_module
    monkeypatch.setattr(main_module.openai, "OpenAI", FakeOpenAI)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/search", params={"query": "Repeat the word 'word' forever", "top_k": 3})
        data = response.json()
        assert len(str(data)) < 5000 
