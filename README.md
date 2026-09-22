# Deploy and Host Laya on Railway

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/new/template/laya?utm_medium=integration&utm_source=button&utm_campaign=laya)

[Laya](https://github.com/NandhaKishorM/laya) is an open-weights (Apache-2.0) decision model. You give it a piece of text (a ticket, an email, a JSON record) and typed questions: pick one of several options, place it on a scale, or yes/no. It answers each with calibrated probabilities in a single forward pass and never generates text. The interface follows TypeSafe's Jev, and this template serves Laya over Jev's HTTP API, so code written for the TypeSafe SDKs runs against your own server once you change one environment variable.

## About Hosting Laya

One service, no database or volume. The checkpoints ship inside the image.

- **Drop-in for the TypeSafe API.** `POST /v1/systemone` and `GET /v1/models` take and return the same shapes as `api.typesafe.ai`. The official Python and JavaScript SDKs work unchanged; both were tested against this template.
- **Route mode, as upstream recommends.** Both general-purpose checkpoints are loaded: English (ModernBERT-large, 421M) and multilingual (mmBERT-base, 322M, 100+ languages). Each request goes to the right one by script and language.
- **No GPU, no downloads at boot.** The weights are baked into the image at a pinned Hugging Face revision, so answers don't change under you, and a restart never waits on Hugging Face. It runs on Railway's CPUs at about 0.2–0.9 s per call.
- **Private by default.** Every `/v1` call needs the generated `LAYA_API_KEY` as a Bearer token. The server refuses to start without one.

## Common Use Cases

- Routing support tickets and emails to the right team, with a confidence you can threshold
- Guardrails and moderation checks in an LLM pipeline without paying per token
- Scoring leads, reviews, or survey answers in any of 100+ languages, with the text staying on your servers

## Dependencies for Laya Hosting

- None. It is a single service that runs on CPU.

### Deployment Dependencies

- [Laya on GitHub](https://github.com/NandhaKishorM/laya) and [on Hugging Face](https://huggingface.co/convaiinnovations/laya)
- [TypeSafe API reference](https://docs.typesafe.ai/api) (the request and answer format)
- [Template source on GitHub](https://github.com/nomideusz/laya-railway)

### Implementation Details

**Use it from the TypeSafe SDKs** (`pip install typesafe-sdk`, `npm install @typesafe-ai/sdk`) by setting two variables in your app:

```bash
TYPESAFE_BASE_URL=https://<your-laya-domain>.up.railway.app
TYPESAFE_API_KEY=<LAYA_API_KEY from the Laya service's Variables tab>
```

From another service in the same Railway project, use `http://${{Laya.RAILWAY_PRIVATE_DOMAIN}}:8080` as the base URL to stay on the private network.

**Or call it directly:**

```bash
curl https://<your-laya-domain>.up.railway.app/v1/systemone \
  -H "Authorization: Bearer $LAYA_API_KEY" -H "Content-Type: application/json" \
  -d '{"state": "We were billed twice for March. Please refund the duplicate today.",
       "questions": {
         "team": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Payments and refunds", "technical": "Bugs and errors"}},
         "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}}'
```

Interactive docs, where you can authorize with the key and try requests, are at `/docs` on your domain.

**Models.** `laya-auto` routes each request, and TypeSafe names such as `jev-latest` (the SDK default) are treated as `laya-auto`, so existing code needs no changes. `laya-english` and `laya-multilingual` force one checkpoint. Each answer includes a `routing` field saying which checkpoint answered and why.

**Memory and plan.** A Hobby or Pro plan is needed, because a checkpoint needs more memory while loading than Trial or Free allow. `LAYA_MODELS` picks what gets loaded:

| `LAYA_MODELS` | Memory | Throughput on 8 vCPU |
| --- | --- | --- |
| `english,multilingual` (default) | ~3.4 GB | 3.5–12 calls/s, depending on the checkpoint |
| `multilingual` | ~1.7 GB | ~12 calls/s |
| `english` | ~2 GB | ~3.5 calls/s |

`multilingual` alone halves the memory and handles English too. Upstream reports the English checkpoint is more accurate on English text. Calls are answered one at a time, because each one already uses every CPU the service has, so bursts queue up rather than competing. Memory stays flat under load. Boot takes under a minute while the checkpoints load.

**Know the limits before you switch.** Laya is not Jev. Upstream's benchmarks show it ahead on some tasks and behind on others, and third-party comparisons vary, so check it against a few hundred of your own labelled examples first. Text past 512 tokens (English) or 1024 tokens (multilingual) is cut off, not rejected. On the English checkpoint, confidence for choice questions with 11 or more options is uncalibrated; laya notes this in the boot log. `usage.output_tokens` is always 0, as with Jev. There are no rate limits beyond what your CPUs can do.

## Why Deploy Laya on Railway?

Railway is a singular platform to deploy your infrastructure stack. Railway will host your infrastructure so you don't have to deal with configuration, while allowing you to vertically and horizontally scale it.

By deploying Laya on Railway, you are one step closer to supporting a complete full-stack application with minimal burden. Host your servers, databases, AI agents, and more on Railway.
