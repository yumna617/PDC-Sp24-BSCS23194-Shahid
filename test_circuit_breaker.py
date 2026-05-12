

import asyncio
import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import (
    app,
    llm_circuit,
    CircuitState,
    CircuitOpenError,
    STUDENT_ID,
    ask_llm,
    FALLBACK_MESSAGE,
)

def reset_circuit():
    llm_circuit._state = CircuitState.CLOSED
    llm_circuit._failure_count = 0
    llm_circuit._last_failure_time = 0.0


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_student_id_header_always_present(client):

    resp = await client.get("/health")
    assert resp.status_code == 200
    assert "x-student-id" in resp.headers, "X-Student-ID header is MISSING!"
    assert resp.headers["x-student-id"] == STUDENT_ID
    print(f"\n✅  X-Student-ID header present: {resp.headers['x-student-id']}")


@pytest.mark.asyncio
async def test_baseline_naive_call_hangs():
   
    async def slow_llm(*_):
        await asyncio.sleep(10)         
        return "This never arrives"

    start = time.monotonic()
    try:
        await asyncio.wait_for(slow_llm(), timeout=3.0)
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        print(f"\n BEFORE FIX: Caller blocked for {elapsed:.2f}s waiting on LLM.")
        assert elapsed >= 2.9, "Expected to wait the full duration before timing out"
        return
    pytest.fail("Expected TimeoutError – the naive call should hang")


@pytest.mark.asyncio
async def test_circuit_trips_after_threshold_failures():
    
    reset_circuit()
    assert llm_circuit.state == CircuitState.CLOSED
    llm_circuit.call_timeout = 0.1  
    async def always_slow(prompt):
        await asyncio.sleep(10)      

    with patch("app.main._call_llm_api", side_effect=always_slow):
        for i in range(llm_circuit.failure_threshold):
            result = await ask_llm("test prompt")
            assert result["source"] in ("fallback_timeout", "fallback_error"), \
                f"Expected fallback on failure #{i+1}"
            print(f"  Failure {i+1}: circuit state = {llm_circuit.state.value}")

    assert llm_circuit.state == CircuitState.OPEN, \
        f"Expected OPEN, got {llm_circuit.state.value}"
    print(f"\nCircuit is OPEN after {llm_circuit.failure_threshold} failures.")

    
    llm_circuit.call_timeout = 5.0

@pytest.mark.asyncio
async def test_open_circuit_rejects_instantly_without_calling_llm():

    reset_circuit()
    llm_circuit._state = CircuitState.OPEN   
    llm_circuit._last_failure_time = time.monotonic()  

    mock_llm = AsyncMock(return_value="should not be called")

    with patch("app.main._call_llm_api", mock_llm):
        start = time.monotonic()
        result = await ask_llm("any prompt")
        elapsed = time.monotonic() - start

    mock_llm.assert_not_called()
    assert result["source"] == "fallback_circuit_open"
    assert result["circuit_state"] == CircuitState.OPEN.value
    assert elapsed < 0.05, f"Expected instant rejection, took {elapsed:.3f}s"
    print(f"\nOPEN circuit rejected in {elapsed*1000:.1f} ms – LLM never called.")


@pytest.mark.asyncio
async def test_api_endpoint_returns_503_with_fallback(client):
    reset_circuit()
    llm_circuit._state = CircuitState.OPEN
    llm_circuit._last_failure_time = time.monotonic()

    resp = await client.post("/api/ask", json={"prompt": "Explain recursion"})

    assert resp.status_code == 503
    body = resp.json()
    assert body["source"] == "fallback_circuit_open"
    assert FALLBACK_MESSAGE in body["answer"]
    assert "x-student-id" in resp.headers     
    print(f"\nAPI returned 503 with fallback: '{body['answer'][:60]}…'")
    print(f"    X-Student-ID: {resp.headers['x-student-id']}")



@pytest.mark.asyncio
async def test_circuit_self_heals_after_recovery_timeout():
   
    reset_circuit()
    llm_circuit._state = CircuitState.OPEN
    llm_circuit._last_failure_time = time.monotonic() - 20.0

    mock_llm = AsyncMock(return_value="Here is a great explanation of recursion!")

    with patch("app.main._call_llm_api", mock_llm):
        result = await ask_llm("Explain recursion")

    assert result["source"] == "llm", \
        f"Expected real LLM response, got: {result['source']}"
    assert llm_circuit.state == CircuitState.CLOSED, \
        f"Expected CLOSED after recovery, got {llm_circuit.state.value}"
    print(f"\n✅  Circuit self-healed to CLOSED after recovery window.")
    print(f"    LLM answer: '{result['answer'][:60]}'")



@pytest.mark.asyncio
async def test_successful_call_keeps_circuit_closed(client):
   
    reset_circuit()

    mock_llm = AsyncMock(return_value="Binary search works by halving the search space.")

    with patch("app.main._call_llm_api", mock_llm):
        resp = await client.post("/api/ask", json={"prompt": "Explain binary search"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "llm"
    assert llm_circuit.state == CircuitState.CLOSED
    print(f"\n✅  Happy path: 200 OK, circuit stays CLOSED.")
    print(f"    Answer: '{body['answer'][:60]}'")



@pytest.mark.asyncio
async def test_circuit_status_endpoint(client):
    reset_circuit()
    resp = await client.get("/api/circuit-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == CircuitState.CLOSED.value
    assert "x-student-id" in resp.headers
    print(f"\n✅  Circuit status endpoint: {body}")
