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


def memory_limit() -> float:
    """GB of RAM this container may use."""
    try:
        limit = open("/sys/fs/cgroup/memory.max").read().strip()
        if limit != "max":
            return int(limit) / 1e9
    except (OSError, ValueError):
        pass
    return float("inf")


torch.set_num_threads(int(os.environ.get("LAYA_THREADS") or cpu_quota()))
# Railway shows stderr as errors, and laya's load-time warnings are notes, not failures.
warnings.showwarning = lambda message, category, *_: print("%s: %s" % (category.__name__, message))

# A checkpoint whose load does not fit is OOM-killed, and the service restarts until Railway gives
# up on it. Cgroup peaks under laya 0.4.17: multilingual 1.71 GB, english 2.01, both 3.39 (a load
# now peaks at about what it keeps), plus ~10% for the API under load. typed-decisions is unmeasured;
# it is ModernBERT-large like english.
NEED_GB = {"multilingual": 1.9, "english": 2.2}


def needed_gb(names) -> float:
    return sum(NEED_GB.get(n, NEED_GB["english"]) for n in names)


# Multilingual first: under laya 0.3.5 its load briefly needed ~2.3 GB on top of what it keeps, so
# this order peaked lower; it also makes multilingual, which reads every language, the one kept.
MEMORY_GB = memory_limit()
WANTED = sorted((normalise_name(n) for n in os.environ.get("LAYA_MODELS", "english,multilingual").split(",") if n.strip()),
                key=lambda n: n != "multilingual")
# Drop checkpoints from the end until the rest fits. With none left, stay up instead of crash-looping
# (Trial and Free cap a service at 1 GB and 0.5 GB), and answer every call with the reason.
LOADED = list(WANTED)
while LOADED and needed_gb(LOADED) > MEMORY_GB:
    LOADED.pop()
TOO_SMALL = ("This service has %.1f GB of RAM, and Laya's checkpoints need at least %.1f GB. "
             "Deploy it on a Hobby or Pro plan." % (MEMORY_GB, needed_gb(WANTED[:1]))) if not LOADED else ""
if LOADED != WANTED and LOADED:
    print("WARNING: %.1f GB of RAM fits %s, not %s; raise the memory limit to %.1f GB to load all of LAYA_MODELS"
          % (MEMORY_GB, ", ".join(LOADED), ", ".join(WANTED), needed_gb(WANTED)))
router = Router(models={"english": (MODELS_DIR, None),
                        "multilingual": (MODELS_DIR, "multilingual"),
                        "typed-decisions": (MODELS_DIR, "typed-decisions")},
                max_loaded=len(LOADED) or 1)
for name in LOADED:
    started = time.monotonic()
    router.load(name)
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand the load's scratch buffers back to the OS
    print("loaded %s in %.0fs" % (name, time.monotonic() - started))
print("WARNING: " + TOO_SMALL if TOO_SMALL else "serving %s with %d threads" % (", ".join(LOADED), torch.get_num_threads()))

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


def ready():
    if TOO_SMALL:
        raise HTTPException(503, TOO_SMALL)


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
              description=TOO_SMALL or "TypeSafe-compatible System One API. Authorize with your LAYA_API_KEY.")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "models": LOADED}


@app.get("/v1/models", dependencies=[Depends(authorized), Depends(ready)])
def models():
    names = [{"name": "laya-auto", "description": "Routes each request by script and language: "
              + " or ".join(LOADED) + ". Also answers to TypeSafe names such as jev-latest.",
              "release_date": RELEASE_DATE}]
    names += [{"name": "laya-" + n, "description": DESCRIPTIONS[n], "release_date": RELEASE_DATE} for n in LOADED]
    return {"models": names}


@app.post("/v1/systemone", dependencies=[Depends(authorized), Depends(ready)])
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
