# Yumna Shahid BSCS23194
## PDC-Sp24-12345-YourLastName

## What This Solves

The naive StudySync backend calls an external LLM API synchronously. When the LLM is slow or down, the server thread blocks for up to 60 seconds — freezing the entire application for every user.

**Fix:** A fully async **Circuit Breaker** wraps every LLM call. After 3 consecutive failures, the breaker trips to `OPEN`, rejecting all subsequent LLM calls **instantly** (< 50 ms) and returning a graceful fallback message. After a configurable recovery window, it self-heals via a `HALF_OPEN` probe.


## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the tests (prove failure + fix)
```bash
pytest tests/test_circuit_breaker.py -v -s
```
### 3. Run the dev server
```bash
cd studysync
uvicorn app.main:app --reload --port 8000
```
### 4. Manually trigger the circuit breaker
```bash
# Simulate 3 failures (LLM unreachable) then watch instant rejection:
for i in 1 2 3 4 5; do
  curl -s -X POST http://localhost:8000/api/ask \
    -H "Content-Type: application/json" \
    -d '{"prompt":"test"}' | python3 -m json.tool
done
```


