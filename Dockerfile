FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 USE_TF=0

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# The two general-purpose checkpoints (English and multilingual) are baked in, so boots never
# depend on Hugging Face. Pinned: a re-upload upstream would change every deploy's answers.
ARG LAYA_REVISION=1c5edc17a7acd8701df6fc341c0d179f1c62c982
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('convaiinnovations/laya', revision='$LAYA_REVISION', local_dir='/models/laya', \
allow_patterns=[p + f for p in ('', 'multilingual/') \
for f in ('rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*')])" \
 && rm -rf /models/laya/.cache \
 && python -c "from laya.agent import _fix_tokenizer_config as fix; fix('/models/laya'); fix('/models/laya/multilingual')"

COPY server.py .
RUN useradd --system --no-create-home laya
USER laya
ENV HF_HUB_OFFLINE=1

CMD ["python", "server.py"]
