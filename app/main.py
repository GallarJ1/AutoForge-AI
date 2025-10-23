from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field
from typing import List
import os
import time
import asyncio
from collections import defaultdict, deque

try:
    from .ai_provider import AzureOpenAIClient     # when imported as app.main
except ImportError:
    from ai_provider import AzureOpenAIClient      # when run as top-level (main.py)


app = FastAPI()

# -------------------------------
# Simple in-memory rate limiter
# -------------------------------
RATE_LIMIT_COUNT = 3          # max requests allowed
RATE_LIMIT_WINDOW = 60 * 60   # 60 minutes (in seconds)

# Map ip -> deque[timestamps]
_request_buckets: dict[str, deque] = defaultdict(deque)
_bucket_lock = asyncio.Lock()

def _client_ip(req: Request) -> str:
    """
    Try to get real client IP even when behind Azure’s reverse proxy.
    Azure App Service sets X-Forwarded-For: "client, proxy1, proxy2, ..."
    We take the first IP if header exists; fallback to req.client.host.
    """
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host or "unknown"

async def enforce_rate_limit(req: Request) -> None:
    now = time.time()
    ip = _client_ip(req)

    async with _bucket_lock:
        bucket = _request_buckets[ip]
        # drop timestamps older than window
        cutoff = now - RATE_LIMIT_WINDOW
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_COUNT:
            # too many requests within window
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {RATE_LIMIT_COUNT} requests every {RATE_LIMIT_WINDOW // 60} minutes."
            )

        # record this request
        bucket.append(now)

# -------------------------------
# Health + root (unchanged)
# -------------------------------
@app.get("/health")
def health():
    return {"status": "Alive!"}

@app.get("/")
def root():
    return {"message": "Hello from Azure with AI!"}

# -------------------------------
# Chat request schema
# -------------------------------
class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)

# -------------------------------
# AI endpoint with rate limiting
# -------------------------------
@app.post("/ai/chat")
async def ai_chat(req_body: ChatRequest, req: Request):
    # 1) Rate limit per IP (3 req / 30 min)
    await enforce_rate_limit(req)

    # 2) Basic prompt-size guardrail (keeps costs tiny)
    total_chars = sum(len(m.content) for m in req_body.messages)
    if total_chars > 6000:
        raise HTTPException(status_code=413, detail="Prompt too long")

    # 3) Call Azure OpenAI
    try:
        client = AzureOpenAIClient()
        result = await client.chat(
            messages=[m.model_dump() for m in req_body.messages],
            max_tokens=req_body.max_tokens,
            temperature=req_body.temperature,
        )
        return result
    except Exception as e:
        # Hide internals but surface a clear error
        raise HTTPException(status_code=502, detail=f"AI provider error: {e}")
    
