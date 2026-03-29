#!/usr/bin/env python3
"""
FastAPI wrapper for the vLLM Turbo S3 multilingual TTS path.

Design goals:
- Keep one shared model instance.
- Queue requests and batch them over a short admission window.
- Expose one normal WAV endpoint and one chunked streaming WAV endpoint.

Note:
This service streams WAV bytes after synthesis finishes. The current runtime
returns full audio at the end, so this is transport streaming, not token-level
incremental speech generation.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from benchmark_multilingual_concurrency import load_model


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tensor_to_wav_bytes(wav_tensor, sample_rate: int) -> bytes:
    wav_np = wav_tensor.squeeze(0).detach().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, wav_np, samplerate=sample_rate, format="WAV")
    return buf.getvalue()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language_id: str = "ar"
    audio_prompt_path: str | None = None

    exaggeration: float = 0.5
    cfg_weight: float = 0.0
    temperature: float = 0.0
    repetition_penalty: float = 2.0
    min_p: float = 0.05
    top_p: float = 1.0

    max_new_tokens: int = 256
    auto_max_new_tokens: bool = True
    auto_max_new_tokens_cap: int = 128

    stream_chunk_bytes: int = 32768


class TTSResponseMeta(BaseModel):
    request_id: str
    queue_wait_s: float
    total_s: float
    profile: dict[str, Any]


@dataclass
class _QueuedItem:
    request_id: str
    payload: TTSRequest
    enqueued_at: float
    future: asyncio.Future


class _BatchScheduler:
    def __init__(
        self,
        *,
        model,
        default_audio_prompt_path: str | None,
        batch_window_ms: float,
        max_batch_size: int,
    ):
        self.model = model
        self.default_audio_prompt_path = default_audio_prompt_path
        self.batch_window_s = max(0.0, float(batch_window_ms) / 1000.0)
        self.max_batch_size = max(1, int(max_batch_size))
        self.queue: asyncio.Queue[_QueuedItem] = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._runner_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(self._runner(), name="tts-batch-runner")

    async def close(self) -> None:
        self._stop_event.set()
        if self._runner_task is not None:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None

    async def submit(self, payload: TTSRequest) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        item = _QueuedItem(
            request_id=request_id,
            payload=payload,
            enqueued_at=time.perf_counter(),
            future=fut,
        )
        await self.queue.put(item)
        return await fut

    async def _runner(self) -> None:
        while not self._stop_event.is_set():
            first = await self.queue.get()
            batch = [first]

            if self.batch_window_s > 0:
                await asyncio.sleep(self.batch_window_s)

            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            started = time.perf_counter()
            try:
                outputs = await asyncio.to_thread(self._infer_batch, batch)
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
                continue

            for item, output in zip(batch, outputs):
                if item.future.done():
                    continue
                queue_wait_s = started - item.enqueued_at
                output["request_id"] = item.request_id
                output["queue_wait_s"] = queue_wait_s
                item.future.set_result(output)

    def _infer_batch(self, batch: list[_QueuedItem]) -> list[dict[str, Any]]:
        sessions = []
        texts = []
        for item in batch:
            req = item.payload
            audio_prompt_path = req.audio_prompt_path or self.default_audio_prompt_path
            session = self.model.create_session(
                audio_prompt_path=audio_prompt_path,
                language_id=req.language_id,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p,
                top_p=req.top_p,
                max_new_tokens=req.max_new_tokens,
                auto_max_new_tokens=req.auto_max_new_tokens,
                auto_max_new_tokens_cap=req.auto_max_new_tokens_cap,
            )
            sessions.append(session)
            texts.append(req.text)

        batch_started = time.perf_counter()
        results = self.model.generate_many_with_sessions(sessions, texts)
        batch_total_s = time.perf_counter() - batch_started

        output_rows: list[dict[str, Any]] = []
        for result in results:
            wav_bytes = _tensor_to_wav_bytes(result["wav"], self.model.sr)
            output_rows.append(
                {
                    "audio_wav": wav_bytes,
                    "profile": result.get("profile", {}),
                    "batch_total_s": batch_total_s,
                }
            )
        return output_rows


def _load_service_model():
    return load_model(
        "vllm_turbo_s3",
        device=os.getenv("API_DEVICE", "cuda"),
        checkpoint_dir=os.getenv("CHECKPOINT_DIR"),
        base_checkpoint_dir=os.getenv("BASE_CHECKPOINT_DIR"),
        turbo_s3_checkpoint_dir=os.getenv("TURBO_S3_CHECKPOINT_DIR"),
        vllm_model_dir=os.getenv("VLLM_MODEL_DIR"),
        vllm_export_dir=os.getenv("VLLM_EXPORT_DIR"),
        vllm_prompt_builder_device=os.getenv("VLLM_PROMPT_BUILDER_DEVICE", "cpu"),
        vllm_tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
        vllm_gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5")),
        vllm_enforce_eager=_env_bool("VLLM_ENFORCE_EAGER", True),
        vllm_dtype=os.getenv("VLLM_DTYPE", "auto"),
        vllm_max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
        vllm_enable_prefix_caching=_env_bool("VLLM_ENABLE_PREFIX_CACHING", False),
        vllm_export_copy=_env_bool("VLLM_EXPORT_COPY", False),
    )


app = FastAPI(title="Chatterbox vLLM Turbo S3 API", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    model = await asyncio.to_thread(_load_service_model)
    scheduler = _BatchScheduler(
        model=model,
        default_audio_prompt_path=os.getenv("DEFAULT_AUDIO_PROMPT_PATH"),
        batch_window_ms=float(os.getenv("API_BATCH_WINDOW_MS", "5.0")),
        max_batch_size=int(os.getenv("API_MAX_BATCH_SIZE", "8")),
    )
    await scheduler.start()
    app.state.model = model
    app.state.scheduler = scheduler


@app.on_event("shutdown")
async def _shutdown() -> None:
    scheduler: _BatchScheduler | None = getattr(app.state, "scheduler", None)
    if scheduler is not None:
        await scheduler.close()
    model = getattr(app.state, "model", None)
    if model is not None and hasattr(model, "close"):
        try:
            await asyncio.to_thread(model.close)
        except Exception:
            pass


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/v1/tts", response_class=Response)
async def synthesize(req: TTSRequest) -> Response:
    scheduler: _BatchScheduler = app.state.scheduler
    started = time.perf_counter()
    try:
        out = await scheduler.submit(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc!r}") from exc

    total_s = time.perf_counter() - started
    profile = out.get("profile", {})
    headers = {
        "X-Request-Id": out["request_id"],
        "X-Queue-Wait-S": f"{out.get('queue_wait_s', 0.0):.4f}",
        "X-Total-S": f"{total_s:.4f}",
        "X-T3-S": f"{float(profile.get('t3_s', 0.0)):.4f}",
        "X-S3-S": f"{float(profile.get('s3_s', 0.0)):.4f}",
    }
    return Response(content=out["audio_wav"], media_type="audio/wav", headers=headers)


@app.post("/v1/tts/stream")
async def synthesize_stream(req: TTSRequest) -> StreamingResponse:
    scheduler: _BatchScheduler = app.state.scheduler
    started = time.perf_counter()
    try:
        out = await scheduler.submit(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc!r}") from exc

    audio = out["audio_wav"]
    chunk_bytes = max(1024, int(req.stream_chunk_bytes))

    async def iterator():
        for offset in range(0, len(audio), chunk_bytes):
            yield audio[offset : offset + chunk_bytes]
            await asyncio.sleep(0)

    total_s = time.perf_counter() - started
    profile = out.get("profile", {})
    headers = {
        "X-Request-Id": out["request_id"],
        "X-Queue-Wait-S": f"{out.get('queue_wait_s', 0.0):.4f}",
        "X-Total-S": f"{total_s:.4f}",
        "X-T3-S": f"{float(profile.get('t3_s', 0.0)):.4f}",
        "X-S3-S": f"{float(profile.get('s3_s', 0.0)):.4f}",
    }
    return StreamingResponse(iterator(), media_type="audio/wav", headers=headers)


@app.post("/v1/tts/meta", response_model=TTSResponseMeta)
async def synthesize_meta(req: TTSRequest) -> TTSResponseMeta:
    scheduler: _BatchScheduler = app.state.scheduler
    started = time.perf_counter()
    try:
        out = await scheduler.submit(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc!r}") from exc
    total_s = time.perf_counter() - started
    return TTSResponseMeta(
        request_id=out["request_id"],
        queue_wait_s=float(out.get("queue_wait_s", 0.0)),
        total_s=total_s,
        profile=out.get("profile", {}),
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("fastapi_vllm_tts_service:app", host=host, port=port, reload=False)
