#!/usr/bin/env python3
"""
FastAPI wrapper for the vLLM Turbo S3 multilingual TTS path.

Design goals:
- Keep one shared model instance.
- Queue requests and batch them over a short admission window.
- Expose one normal WAV endpoint, one transport-streaming WAV endpoint, and
  one true chunked-streaming NDJSON endpoint.

Chunked streaming (/v1/tts/stream_chunks):
  Text is split on punctuation / word-count boundaries before inference.
  Each chunk is synthesised as an independent vLLM call.  Chunk jobs from
  multiple concurrent requests are batched together so the engine stays busy
  across requests.  Results are streamed as NDJSON events as soon as each
  chunk finishes — well before the full text is done.

Note:
/v1/tts and /v1/tts/stream still wait for full synthesis; they are unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import io
import json
import logging
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


LOGGER = logging.getLogger("chatterbox.fastapi_vllm_tts")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

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


def _extract_stage_timings(profile: dict[str, Any]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for key, value in profile.items():
        if isinstance(value, (int, float)) and key.endswith("_s"):
            timings[key] = float(value)
    return timings


def _extract_stage_meta(profile: dict[str, Any]) -> dict[str, float]:
    keys = (
        "t3_alignment_analyzer_supported",
        "t3_alignment_analyzer_active",
        "t3_batch_size",
        "t3_text_token_len",
        "t3_prompt_speech_token_len",
        "t3_initial_speech_len",
        "t3_cond_seq_len",
        "t3_prompt_embed_seq_len",
        "t3_prompt_embed_hidden_size",
        "t3_generated_tokens",
        "t3_max_new_tokens_requested",
        "t3_max_new_tokens_effective",
        "s3_finalize_order",
        "s3_finalize_batch_size",
    )
    meta: dict[str, float] = {}
    for key in keys:
        value = profile.get(key)
        if isinstance(value, (int, float)):
            meta[key] = float(value)
    return meta


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

_HARD_PUNCT = frozenset('.!?؟！')
_SOFT_PUNCT = frozenset(',،;؛:')


def split_text_for_streaming(
    text: str,
    *,
    target_words: int = 5,
    max_words: int = 8,
) -> list[str]:
    """Split *text* into synthesisable chunks on natural boundaries.

    Priority order:
    1. Hard punctuation (.!?؟) with ≥ 2 accumulated words.
    2. Soft punctuation (,،;؛:) with ≥ target_words accumulated words.
    3. Word-count cap (max_words).

    Trailing fragments shorter than 2 words are merged into the previous chunk.
    Punctuation is preserved in the emitted chunk text.
    """
    words = text.split()
    if not words:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        current.append(word)
        last = word.rstrip()
        if not last:
            continue
        ch = last[-1]
        n = len(current)
        if ch in _HARD_PUNCT and n >= 2:
            chunks.append(' '.join(current))
            current = []
        elif ch in _SOFT_PUNCT and n >= target_words:
            chunks.append(' '.join(current))
            current = []
        elif n >= max_words:
            chunks.append(' '.join(current))
            current = []

    if current:
        remainder = ' '.join(current)
        if chunks and len(current) < 2:
            # Merge tiny trailing fragment into previous chunk.
            chunks[-1] = chunks[-1] + ' ' + remainder
        else:
            chunks.append(remainder)

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

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

    # Chunked-streaming options (used by /v1/tts/stream_chunks).
    chunked_streaming: bool = False
    chunk_target_words: int = 5
    chunk_max_words: int = 8
    chunk_auto_max_new_tokens_cap: int = 64


class TTSResponseMeta(BaseModel):
    request_id: str
    queue_wait_s: float
    total_s: float
    profile: dict[str, Any]


# ---------------------------------------------------------------------------
# Scheduler internals
# ---------------------------------------------------------------------------

@dataclass
class _QueuedItem:
    """A whole-text synthesis request."""
    request_id: str
    payload: TTSRequest
    enqueued_at: float
    future: asyncio.Future


@dataclass
class _StreamingRequestState:
    """Mutable state for one /v1/tts/stream_chunks request."""
    request_id: str
    payload: TTSRequest
    chunks: list[str]
    chunk_cap: int          # effective auto_max_new_tokens_cap for each chunk
    session: Any            # StreamingSession; None until first chunk runs
    result_queue: asyncio.Queue
    enqueued_at: float
    first_chunk_emitted_at: float | None = None
    chunks_emitted: int = 0


@dataclass
class _ChunkJob:
    """One pending chunk synthesis job."""
    request_id: str
    chunk_text: str
    chunk_index: int
    chunk_count: int
    is_final: bool
    enqueued_at: float
    result_queue: asyncio.Queue
    state: _StreamingRequestState   # back-ref so runner can enqueue next chunk


@dataclass
class _WorkItem:
    """Union of whole-request and chunk jobs, sharing one queue."""
    kind: str                       # "whole" | "chunk"
    whole: _QueuedItem | None = None
    chunk: _ChunkJob | None = None


# ---------------------------------------------------------------------------
# Batch scheduler
# ---------------------------------------------------------------------------

class _BatchScheduler:
    """
    Single runner loop that services both whole-request and chunk jobs.

    Whole-request jobs resolve asyncio Futures.
    Chunk jobs push NDJSON event dicts to per-request asyncio Queues and
    enqueue the next chunk back into the work queue when done.

    All inference runs inside one asyncio.to_thread() call so the engine
    is never driven from two threads simultaneously.
    """

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
        self._work_queue: asyncio.Queue[_WorkItem] = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._runner_task: asyncio.Task | None = None
        self._batch_seq = 0
        self._trace_enabled = _env_bool("API_TRACE_ENABLED", True)
        self._trace_stdout = _env_bool("API_TRACE_STDOUT", False)
        self._recent_batches: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=max(1, int(os.getenv("API_TRACE_RECENT_BATCHES", "200")))
        )

    def get_recent_batch_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        return list(self._recent_batches)[-limit:]

    async def start(self) -> None:
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(
                self._runner(), name="tts-batch-runner"
            )

    async def close(self) -> None:
        self._stop_event.set()
        if self._runner_task is not None:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None

    # ------------------------------------------------------------------
    # Public: whole-request submission (used by /v1/tts and /v1/tts/stream)
    # ------------------------------------------------------------------

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
        await self._work_queue.put(_WorkItem(kind="whole", whole=item))
        return await fut

    # ------------------------------------------------------------------
    # Public: chunked streaming submission (used by /v1/tts/stream_chunks)
    # ------------------------------------------------------------------

    async def submit_chunked(self, payload: TTSRequest) -> _StreamingRequestState:
        """Split text, enqueue first chunk, return state with result_queue."""
        request_id = uuid.uuid4().hex
        chunks = split_text_for_streaming(
            payload.text,
            target_words=payload.chunk_target_words,
            max_words=payload.chunk_max_words,
        )
        if not chunks:
            chunks = [payload.text]

        chunk_cap = min(
            int(payload.auto_max_new_tokens_cap),
            int(payload.chunk_auto_max_new_tokens_cap),
        )

        result_queue: asyncio.Queue = asyncio.Queue()
        state = _StreamingRequestState(
            request_id=request_id,
            payload=payload,
            chunks=chunks,
            chunk_cap=chunk_cap,
            session=None,
            result_queue=result_queue,
            enqueued_at=time.perf_counter(),
        )

        first_job = _ChunkJob(
            request_id=request_id,
            chunk_text=chunks[0],
            chunk_index=0,
            chunk_count=len(chunks),
            is_final=(len(chunks) == 1),
            enqueued_at=time.perf_counter(),
            result_queue=result_queue,
            state=state,
        )
        await self._work_queue.put(_WorkItem(kind="chunk", chunk=first_job))
        return state

    # ------------------------------------------------------------------
    # Runner loop
    # ------------------------------------------------------------------

    async def _runner(self) -> None:
        while not self._stop_event.is_set():
            first = await self._work_queue.get()
            batch: list[_WorkItem] = [first]

            if self.batch_window_s > 0:
                await asyncio.sleep(self.batch_window_s)

            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self._work_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            started = time.perf_counter()
            batch_id = self._batch_seq
            self._batch_seq += 1
            queue_depth_after_pick = self._work_queue.qsize()
            try:
                infer_start = time.perf_counter()
                whole_results, chunk_results, infer_meta = await asyncio.to_thread(
                    self._infer_mixed_batch, batch, batch_id
                )
                infer_end = time.perf_counter()
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    if item.kind == "whole" and not item.whole.future.done():
                        item.whole.future.set_exception(exc)
                    elif item.kind == "chunk":
                        await item.chunk.result_queue.put({
                            "event": "error",
                            "request_id": item.chunk.request_id,
                            "chunk_index": item.chunk.chunk_index,
                            "detail": repr(exc),
                        })
                continue

            batch_trace = {
                "batch_id": batch_id,
                "batch_size": len(batch),
                "whole_count": int(infer_meta.get("whole_count", 0)),
                "chunk_count": int(infer_meta.get("chunk_count", 0)),
                "queue_depth_after_pick": queue_depth_after_pick,
                "batch_wait_window_s": self.batch_window_s,
                "infer_wall_s": infer_end - infer_start,
                "model_generate_many_s": float(infer_meta.get("model_generate_many_s", 0.0)),
                "session_create_whole_s": float(infer_meta.get("session_create_whole_s", 0.0)),
                "session_create_chunk_s": float(infer_meta.get("session_create_chunk_s", 0.0)),
            }
            if self._trace_enabled:
                self._recent_batches.append(batch_trace)
            if self._trace_stdout:
                LOGGER.info("trace_batch %s", json.dumps(batch_trace, sort_keys=True))

            # Resolve whole-request futures.
            whole_items = [b for b in batch if b.kind == "whole"]
            for w_item, result in zip(whole_items, whole_results):
                item = w_item.whole
                if item.future.done():
                    continue
                result["request_id"] = item.request_id
                result["queue_wait_s"] = started - item.enqueued_at
                result["trace"] = {
                    **batch_trace,
                    "request_elapsed_s": time.perf_counter() - item.enqueued_at,
                    "stage_timings": _extract_stage_timings(result.get("profile", {})),
                    "stage_meta": _extract_stage_meta(result.get("profile", {})),
                }
                item.future.set_result(result)

            # Push chunk events and enqueue next chunks.
            chunk_items = [b for b in batch if b.kind == "chunk"]
            for c_item, result in zip(chunk_items, chunk_results):
                chunk = c_item.chunk
                profile = result["profile"]
                emit_now = time.perf_counter()
                if chunk.state.first_chunk_emitted_at is None:
                    chunk.state.first_chunk_emitted_at = emit_now
                chunk.state.chunks_emitted += 1
                first_chunk_latency_s = chunk.state.first_chunk_emitted_at - chunk.state.enqueued_at

                event: dict[str, Any] = {
                    "event": "chunk",
                    "request_id": chunk.request_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.chunk_text,
                    "audio_wav_b64": result["audio_wav_b64"],
                    "sample_rate": result["sample_rate"],
                    "queue_wait_s": started - chunk.enqueued_at,
                    "t3_s": float(profile.get("t3_s", 0.0)),
                    "s3_s": float(profile.get("s3_s", 0.0)),
                    "chunk_total_s": float(profile.get("audio_ready_s", 0.0)) or result["batch_total_s"],
                    "is_final": chunk.is_final,
                    "trace": {
                        **batch_trace,
                        "request_elapsed_s": emit_now - chunk.state.enqueued_at,
                        "first_chunk_latency_s": first_chunk_latency_s,
                        "session_create_s": float(result.get("session_create_s", 0.0)),
                        "text_prep_s": float(profile.get("text_prep_s", 0.0)),
                        "t3_wait_s": float(profile.get("t3_wait_s", 0.0)),
                        "t3_active_s": float(profile.get("t3_active_s", 0.0)),
                        "audio_ready_s": float(profile.get("audio_ready_s", 0.0)),
                        "stage_timings": _extract_stage_timings(profile),
                        "stage_meta": _extract_stage_meta(profile),
                    },
                }
                await chunk.result_queue.put(event)

                if not chunk.is_final:
                    state = chunk.state
                    next_idx = chunk.chunk_index + 1
                    next_job = _ChunkJob(
                        request_id=chunk.request_id,
                        chunk_text=state.chunks[next_idx],
                        chunk_index=next_idx,
                        chunk_count=chunk.chunk_count,
                        is_final=(next_idx == chunk.chunk_count - 1),
                        enqueued_at=time.perf_counter(),
                        result_queue=chunk.result_queue,
                        state=state,
                    )
                    await self._work_queue.put(_WorkItem(kind="chunk", chunk=next_job))
                else:
                    await chunk.result_queue.put({
                        "event": "done",
                        "request_id": chunk.request_id,
                        "trace": {
                            "request_total_s": time.perf_counter() - chunk.state.enqueued_at,
                            "first_chunk_latency_s": first_chunk_latency_s,
                            "chunks_emitted": chunk.state.chunks_emitted,
                        },
                    })

    # ------------------------------------------------------------------
    # Inference — runs inside asyncio.to_thread()
    # ------------------------------------------------------------------

    def _infer_mixed_batch(
        self, batch: list[_WorkItem], batch_id: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """
        Handle a mixed batch of whole-request and chunk jobs.

        Both job types are collapsed into a single generate_many() call so the
        vLLM engine sees one batch regardless of source.
        """
        whole_items = [b for b in batch if b.kind == "whole"]
        chunk_items = [b for b in batch if b.kind == "chunk"]
        session_create_whole_s = 0.0
        session_create_chunk_s = 0.0

        all_sessions: list[Any] = []
        all_texts: list[str] = []

        # Sessions for whole requests (create fresh each time).
        for w in whole_items:
            req = w.whole.payload
            audio_prompt_path = req.audio_prompt_path or self.default_audio_prompt_path
            session_create_started = time.perf_counter()
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
            session_create_whole_s += time.perf_counter() - session_create_started
            all_sessions.append(session)
            all_texts.append(req.text)

        # Sessions for chunk jobs (create once on first chunk, reuse thereafter).
        chunk_session_create_map: dict[tuple[str, int], float] = {}
        for c in chunk_items:
            state = c.chunk.state
            if state.session is None:
                req = state.payload
                audio_prompt_path = req.audio_prompt_path or self.default_audio_prompt_path
                session_create_started = time.perf_counter()
                state.session = self.model.create_session(
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
                    auto_max_new_tokens_cap=state.chunk_cap,
                )
                created_s = time.perf_counter() - session_create_started
                session_create_chunk_s += created_s
                chunk_session_create_map[(c.chunk.request_id, c.chunk.chunk_index)] = created_s
            all_sessions.append(state.session)
            all_texts.append(c.chunk.chunk_text)

        # Single batched inference call via the underlying worker.
        batch_started = time.perf_counter()
        raw_results = self.model.worker.generate_many(
            sessions=all_sessions,
            texts=all_texts,
        )
        batch_total_s = time.perf_counter() - batch_started

        n_whole = len(whole_items)

        # Whole-request results.
        whole_results: list[dict[str, Any]] = []
        for r in raw_results[:n_whole]:
            wav_bytes = _tensor_to_wav_bytes(r["wav"], self.model.sr)
            whole_results.append({
                "audio_wav": wav_bytes,
                "profile": r.get("profile", {}),
                "batch_total_s": batch_total_s,
            })

        # Chunk results.
        chunk_results: list[dict[str, Any]] = []
        for r in raw_results[n_whole:]:
            wav_bytes = _tensor_to_wav_bytes(r["wav"], self.model.sr)
            chunk_results.append({
                "audio_wav_b64": base64.b64encode(wav_bytes).decode(),
                "sample_rate": self.model.sr,
                "profile": r.get("profile", {}),
                "batch_total_s": batch_total_s,
            })
        for chunk, chunk_result in zip(chunk_items, chunk_results):
            chunk_result["session_create_s"] = chunk_session_create_map.get(
                (chunk.chunk.request_id, chunk.chunk.chunk_index),
                0.0,
            )

        infer_meta = {
            "batch_id": batch_id,
            "whole_count": len(whole_items),
            "chunk_count": len(chunk_items),
            "model_generate_many_s": batch_total_s,
            "session_create_whole_s": session_create_whole_s,
            "session_create_chunk_s": session_create_chunk_s,
        }

        return whole_results, chunk_results, infer_meta


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Chatterbox vLLM Turbo S3 API", version="0.2.0")


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


# ---------------------------------------------------------------------------
# Existing endpoints — behaviour unchanged
# ---------------------------------------------------------------------------

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
    trace = out.get("trace", {})
    headers = {
        "X-Request-Id": out["request_id"],
        "X-Queue-Wait-S": f"{out.get('queue_wait_s', 0.0):.4f}",
        "X-Total-S": f"{total_s:.4f}",
        "X-T3-S": f"{float(profile.get('t3_s', 0.0)):.4f}",
        "X-S3-S": f"{float(profile.get('s3_s', 0.0)):.4f}",
        "X-Batch-Id": str(trace.get("batch_id", "")),
        "X-Batch-Size": str(trace.get("batch_size", "")),
        "X-Batch-Infer-S": f"{float(trace.get('infer_wall_s', 0.0)):.4f}",
        "X-Batch-Queue-Depth": str(trace.get("queue_depth_after_pick", "")),
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
    trace = out.get("trace", {})
    headers = {
        "X-Request-Id": out["request_id"],
        "X-Queue-Wait-S": f"{out.get('queue_wait_s', 0.0):.4f}",
        "X-Total-S": f"{total_s:.4f}",
        "X-T3-S": f"{float(profile.get('t3_s', 0.0)):.4f}",
        "X-S3-S": f"{float(profile.get('s3_s', 0.0)):.4f}",
        "X-Batch-Id": str(trace.get("batch_id", "")),
        "X-Batch-Size": str(trace.get("batch_size", "")),
        "X-Batch-Infer-S": f"{float(trace.get('infer_wall_s', 0.0)):.4f}",
        "X-Batch-Queue-Depth": str(trace.get("queue_depth_after_pick", "")),
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
    profile = dict(out.get("profile", {}))
    trace = out.get("trace", {})
    if trace:
        profile["_trace"] = trace
    return TTSResponseMeta(
        request_id=out["request_id"],
        queue_wait_s=float(out.get("queue_wait_s", 0.0)),
        total_s=total_s,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# New chunked-streaming endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/tts/stream_chunks")
async def synthesize_stream_chunks(req: TTSRequest) -> StreamingResponse:
    """
    True text-chunk streaming endpoint.

    Text is split into natural-boundary chunks (punctuation → word count) and
    each chunk is synthesised independently.  Chunk jobs from concurrent
    requests are batched together by the shared scheduler.

    Response: application/x-ndjson — one JSON object per line.

    Event shapes:
      {"event": "chunk", "request_id": "...", "chunk_index": 0,
       "text": "...", "audio_wav_b64": "<base64 WAV>", "sample_rate": 24000,
       "queue_wait_s": 0.0, "t3_s": 1.2, "s3_s": 0.3,
       "chunk_total_s": 1.6, "is_final": false}

      {"event": "done",  "request_id": "..."}

      {"event": "error", "request_id": "...", "chunk_index": 0, "detail": "..."}
    """
    scheduler: _BatchScheduler = app.state.scheduler

    async def ndjson_gen():
        try:
            state = await scheduler.submit_chunked(req)
            while True:
                try:
                    event = await asyncio.wait_for(
                        state.result_queue.get(), timeout=180.0
                    )
                except asyncio.TimeoutError:
                    yield json.dumps({
                        "event": "error",
                        "request_id": "unknown",
                        "detail": "timeout waiting for chunk",
                    }) + "\n"
                    return
                yield json.dumps(event) + "\n"
                if event.get("event") in ("done", "error"):
                    return
        except Exception as exc:  # noqa: BLE001
            yield json.dumps({"event": "error", "detail": repr(exc)}) + "\n"

    return StreamingResponse(ndjson_gen(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Debug / introspection
# ---------------------------------------------------------------------------

@app.post("/v1/tts/split_preview")
async def split_preview(req: TTSRequest) -> dict[str, Any]:
    """Return the chunk split that would be used for stream_chunks, without synthesising."""
    chunks = split_text_for_streaming(
        req.text,
        target_words=req.chunk_target_words,
        max_words=req.chunk_max_words,
    )
    return {
        "chunk_count": len(chunks),
        "chunks": chunks,
        "chunk_target_words": req.chunk_target_words,
        "chunk_max_words": req.chunk_max_words,
        "chunk_auto_max_new_tokens_cap": req.chunk_auto_max_new_tokens_cap,
    }


@app.get("/v1/tts/trace/recent")
async def trace_recent(limit: int = 50) -> dict[str, Any]:
    """Return recent scheduler batch traces to diagnose queueing and stage timing."""
    scheduler: _BatchScheduler = app.state.scheduler
    return {
        "limit": max(1, int(limit)),
        "queue_depth_now": scheduler._work_queue.qsize(),
        "batches": scheduler.get_recent_batch_traces(limit=limit),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("fastapi_vllm_tts_service:app", host=host, port=port, reload=False)
