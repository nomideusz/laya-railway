"""TypeSafe-compatible HTTP API for Laya, the open-weights System One decision model.

POST /v1/systemone takes the body TypeSafe's endpoint takes and answers in the same shape,
so the TypeSafe SDKs work unchanged once TYPESAFE_BASE_URL points here.
"""
import asyncio
import ctypes
import gc
import hmac
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from laya import Router
from laya.router import normalise_name
from pydantic import BaseModel, Field

API_KEY = os.environ.get("LAYA_API_KEY", "")
if not API_KEY:
    raise SystemExit("LAYA_API_KEY is not set; refusing to serve an open endpoint")

MODELS_DIR = os.environ.get("LAYA_MODELS_DIR", "/models/laya")
RELEASE_DATE = "2026-09-20"  # the pinned Hugging Face revision (see Dockerfile)
DESCRIPTIONS = {
    "english": "ModernBERT-large, 421M. Best on English text.",
    "multilingual": "mmBERT-base, 322M. 100+ languages, faster.",
    "typed-decisions": "ModernBERT-large, 421M, fine-tuned on four triage workflows.",
}


def cpu_quota() -> int:
    """CPUs this container may use. torch sizes its thread pool from the host's cores
    (48 on Railway), far past the cgroup quota, and oversubscribed threads get throttled."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


torch.set_num_threads(int(os.environ.get("LAYA_THREADS") or cpu_quota()))
# Railway shows stderr as errors, and laya's load-time warnings are notes, not failures.
warnings.showwarning = lambda message, category, *_: print("%s: %s" % (category.__name__, message))

LOADED = [normalise_name(n) for n in os.environ.get("LAYA_MODELS", "english,multilingual").split(",") if n.strip()]
router = Router(models={"english": (MODELS_DIR, None),
                        "multilingual": (MODELS_DIR, "multilingual"),
                        "typed-decisions": (MODELS_DIR, "typed-decisions")},
                max_loaded=len(LOADED))
# Multilingual first: its load briefly needs ~2.3 GB on top of what it keeps, English ~0.8 GB,
# so this order peaks ~1.3 GB lower than the other.
for name in sorted(LOADED, key=lambda n: n != "multilingual"):
    started = time.monotonic()
    router.load(name)
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand the load's scratch buffers back to the OS
    print("loaded %s in %.0fs" % (name, time.monotonic() - started))
print("serving %s with %d threads" % (", ".join(LOADED), torch.get_num_threads()))

# One thread runs every forward pass: one at a time, since each already uses every thread torch
# has, and always the same thread, so its heap is reused. Spread over the server's worker threads,
# memory grew ~0.5 GB under load; this way it stays flat.
inference = ThreadPoolExecutor(1)

Text = str | dict[str, Any] | list[Any]


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: Text = ""
    criteria: dict[str, Text | None] = Field(min_length=1, max_length=255)


class Score(BaseModel):
    type: Literal["score"]
    instructions: Text = ""
    criteria: list[Text] = Field(min_length=2, max_length=10)


class NoulCriteria(BaseModel):
    true: Text | None = None
    false: Text | None = None


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: Text = ""
    criteria: NoulCriteria | None = None


class SystemOneRequest(BaseModel):
    state: Text
    model: str = "auto"
    questions: dict[str, Annotated[Choice | Score | Noul, Field(discriminator="type")]] = Field(min_length=1)


def authorized(creds: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False))):
    if creds is None or not hmac.compare_digest(creds.credentials.encode(), API_KEY.encode()):
        raise HTTPException(401, "Missing or invalid API key", {"WWW-Authenticate": "Bearer"})


def checkpoint(req: SystemOneRequest, questions: dict) -> dict:
    """Which loaded checkpoint answers: the one named, or the router's pick for the text.
    TypeSafe model names (jev-latest, ...) mean the router, so SDK defaults just work."""
    model = req.model.strip().lower()
    if model in ("auto", "laya-auto") or model.startswith("jev"):
        routed = router.route(req.state, questions)
        name, reason = routed["model"], routed["reason"]
        if name not in LOADED:
            fallback = "multilingual" if "multilingual" in LOADED else LOADED[0]
            name, reason = fallback, "%s; %s is not loaded, so %s answered" % (reason, name, fallback)
        return {"model": name, "reason": reason}
    try:
        name = normalise_name(model.removeprefix("laya-"))
    except ValueError:
        raise HTTPException(422, "unknown model %r; use laya-auto (or a TypeSafe name such as jev-latest) "
                                 "or one of %s" % (req.model, ", ".join("laya-" + n for n in LOADED))) from None
    if name not in LOADED:
        raise HTTPException(422, "checkpoint %r is not loaded here; LAYA_MODELS loads %s"
                                 % (name, ", ".join("laya-" + n for n in LOADED)))
    return {"model": name, "reason": "explicit model=%r" % req.model}


app = FastAPI(title="Laya", version=RELEASE_DATE, redoc_url=None,
              description="TypeSafe-compatible System One API. Authorize with your LAYA_API_KEY.")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "models": LOADED}


@app.get("/v1/models", dependencies=[Depends(authorized)])
def models():
    names = [{"name": "laya-auto", "description": "Routes each request by script and language: "
              + " or ".join(LOADED) + ". Also answers to TypeSafe names such as jev-latest.",
              "release_date": RELEASE_DATE}]
    names += [{"name": "laya-" + n, "description": DESCRIPTIONS[n], "release_date": RELEASE_DATE} for n in LOADED]
    return {"models": names}


@app.post("/v1/systemone", dependencies=[Depends(authorized)])
async def system_one(req: SystemOneRequest):
    questions = {qid: q.model_dump() for qid, q in req.questions.items()}
    decision = checkpoint(req, questions)
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            inference, router.load(decision["model"]).system_one, req.state, questions)
    except ValueError as e:  # e.g. more options than the checkpoint's option budget holds
        raise HTTPException(422, str(e)) from None
    result["model"] = "laya-" + decision["model"]
    result["routing"] = decision
    return result


if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["handlers"]["default"]["stream"] = "ext://sys.stdout"  # Railway shows stderr as errors
    # "" = every interface, IPv4 and IPv6. Railway's private network can be IPv6-only, and asyncio
    # makes a "::" listener IPv6-only, which would shut out IPv4 clients.
    # Railway's edge reuses idle connections to the app and drops them itself within ~10 s. If
    # uvicorn closes one first (its default keep-alive is 5 s), a request the edge sends at that
    # moment is lost and the caller gets a 502. A long keep-alive leaves the closing to the edge.
    uvicorn.run(app, host="", port=int(os.environ.get("PORT", "8080")), log_config=LOGGING_CONFIG,
                timeout_keep_alive=3600)
