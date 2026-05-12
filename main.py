

import asyncio
import time
import httpx
import logging
from enum import Enum
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

STUDENT_ID = "BSCS23194"   

class StudentIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Student-ID"] = STUDENT_ID
        return response


class CircuitState(Enum):
    CLOSED   = "CLOSED"   
    OPEN     = "OPEN"    
    HALF_OPEN = "HALF_OPEN" 


class CircuitBreaker:
    

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 15.0, 
        call_timeout: float = 5.0,       
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.call_timeout = call_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def _trip(self):
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()
        log.warning(
            f"[CircuitBreaker:{self.name}] TRIPPED → OPEN after "
            f"{self._failure_count} failures."
        )

    def _reset(self):
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        log.info(f"[CircuitBreaker:{self.name}] RESET → CLOSED (service recovered).")

    def _maybe_transition_to_half_open(self):
       
        if (
            self._state == CircuitState.OPEN
            and (time.monotonic() - self._last_failure_time) >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            log.info(
                f"[CircuitBreaker:{self.name}] Recovery window elapsed → HALF_OPEN. "
                "Sending probe request."
            )

    async def call(self, fn, *args, **kwargs):
        
        async with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN – request rejected immediately."
                )

        try:
            result = await asyncio.wait_for(fn(*args, **kwargs), timeout=self.call_timeout)

            async with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._reset()
                else:
                   
                    self._failure_count = 0

            return result

        except (asyncio.TimeoutError, Exception) as exc:
            async with self._lock:
                self._failure_count += 1
                log.error(
                    f"[CircuitBreaker:{self.name}] Failure #{self._failure_count}: {exc}"
                )

                if self._state == CircuitState.HALF_OPEN:
                    self._trip()
                elif self._failure_count >= self.failure_threshold:
                    self._trip()

            raise


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""


llm_circuit = CircuitBreaker(
    name="LLM-API",
    failure_threshold=3,
    recovery_timeout=15.0,
    call_timeout=5.0,
)

LLM_API_URL = "http://localhost:9999/llm"  


async def _call_llm_api(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            LLM_API_URL,
            json={"prompt": prompt},
            timeout=60,   
        )
        resp.raise_for_status()
        data = resp.json()
        return data["answer"]


FALLBACK_MESSAGE = (
    "Our AI assistant is temporarily unavailable. "
    "Please try again in a few moments. "
    "In the meantime, check our Study Resources library for help."
)


async def ask_llm(prompt: str) -> dict:

    try:
        answer = await llm_circuit.call(_call_llm_api, prompt)
        return {
            "answer": answer,
            "source": "llm",
            "circuit_state": llm_circuit.state.value,
        }

    except CircuitOpenError:
        log.warning("[LLM] Circuit is OPEN – returning cached fallback immediately.")
        return {
            "answer": FALLBACK_MESSAGE,
            "source": "fallback_circuit_open",
            "circuit_state": llm_circuit.state.value,
        }

    except asyncio.TimeoutError:
        log.warning("[LLM] Call timed out – returning fallback.")
        return {
            "answer": FALLBACK_MESSAGE,
            "source": "fallback_timeout",
            "circuit_state": llm_circuit.state.value,
        }

    except Exception as exc:
        log.error(f"[LLM] Unexpected error: {exc} – returning fallback.")
        return {
            "answer": FALLBACK_MESSAGE,
            "source": "fallback_error",
            "circuit_state": llm_circuit.state.value,
        }


app = FastAPI(title="StudySync API", version="1.0.0")

app.add_middleware(StudentIDMiddleware)         
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "student_id": STUDENT_ID}


@app.post("/api/ask")
async def ask_endpoint(request: Request):
    
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    result = await ask_llm(prompt)
    status = 200 if result["source"] == "llm" else 503
    return JSONResponse(result, status_code=status)


@app.get("/api/circuit-status")
async def circuit_status():

    return {
        "circuit": llm_circuit.name,
        "state": llm_circuit.state.value,
        "failure_count": llm_circuit._failure_count,
        "recovery_timeout_seconds": llm_circuit.recovery_timeout,
    }
