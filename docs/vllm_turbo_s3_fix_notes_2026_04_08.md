# vLLM Turbo S3 Fix Notes (2026-04-08)

This note captures the debugging and fixes applied while validating the
FastAPI + vLLM multilingual TTS path.

## Summary

The main correctness bug in the experimental vLLM runtime was a decode-side
speech-position mismatch. Native T3 generates speech tokens with learned speech
positions relative to the speech segment (`0, 1, 2, ...` after the speech BOS).
The vLLM spike was using approximate absolute positions from the full prompt,
which caused degenerate loops and truncated or nonsensical audio.

After fixing speech-relative position handling in the vLLM model wrapper, the
Arabic warmup request began emitting diverse token IDs and stopping on EOS. The
generated audio sounded correct.

## Files Changed

### `src/chatterbox/vllm_t3_model.py`

- Fix decode-side speech-position semantics.
- Record the prompt's final absolute speech-BOS position during prefill.
- Map decode positions back to speech-relative positions during generation.
- Add targeted debug logging for prompt/decode position tracking.

Why this matters:
- Native T3 expects the first generated speech token to use speech position `1`.
- The broken vLLM path was effectively using prompt-absolute positions such as
  `67`, `68`, `69`, which point at very different learned embeddings.

### `src/chatterbox/vllm_t3_bridge.py`

- Add detailed shape tracing for:
  - normalized text
  - text token IDs
  - prompt-embed assembly
  - the extra BOS appended to match native T3 inference

Why this matters:
- It made the prompt contract visible and confirmed that prompt assembly was
  structurally correct before decode.

### `src/chatterbox/runtime/worker_vllm.py`

- Add detailed trace logging for:
  - raw vLLM token IDs
  - post-trim token IDs
  - post-filter speech tokens
  - S3 output shape
- Replace BOS/EOS slicing behavior with a valid speech-token range filter.
- Harden the repeated-suffix tail trim so it cannot erase nearly the whole
  sequence when vLLM hits a length cap.

Why this matters:
- The first visible failure mode was that a 96-token vLLM output was being
  trimmed down to 2 tokens before S3 even saw it.
- After this fix, S3 received the full token stream and the remaining bug was
  clearly isolated to decode semantics.

### `fastapi_vllm_tts_service.py`

- Add in-memory capture of `chatterbox.shape` logs.
- Add `GET /v1/tts/trace/shapes` so recent shape traces can be retrieved
  without tailing a logfile.
- Default FastAPI request temperature was updated away from greedy decoding.

Why this matters:
- Greedy decoding made the earlier vLLM failures much harsher and harder to
  diagnose.
- The trace endpoint made it practical to compare native and vLLM execution.

### `src/chatterbox/models/t3/inference/alignment_stream_analyzer_scheduled.py`

- Force eager attention when enabling `output_attentions=True` on newer
  Transformers builds where SDPA no longer supports that mode.

Why this matters:
- The alignment controller otherwise fails under the newer Transformers stack.

### `src/chatterbox/models/t3/inference/t3_hf_backend.py`

- Add compatibility helpers to convert KV caches to and from
  `DynamicCache` on newer Transformers versions.

Why this matters:
- It keeps the scheduled/native path working across the modern cache API
  boundary.

### `src/chatterbox/models/t3/inference/scheduled_decode.py`
### `src/chatterbox/runtime/t3_scheduler.py`

- Reduce trace noise to first-step-only in the most repetitive decode logs.

Why this matters:
- It made the traces readable enough to compare native and vLLM behavior.

### `stream_chunks_client.py`

- Restore `--use-language-patterns`.
- Add built-in language-specific sample texts for Arabic, English, and Chinese.
- Compute whole-request and per-chunk token caps from the input text and
  language instead of always sending flat defaults.
- Record the client-side budget decisions in `summary.json`.

Why this matters:
- Chunked streaming still uses a flat per-chunk cap chosen by the request.
- Different languages tokenize differently, so the client now picks a more
  appropriate cap from the actual request text instead of overshooting by
  default.

## Key Diagnostic Trace Progression

### Broken vLLM state

- Prompt assembly looked plausible.
- vLLM output hit the length limit.
- Repeated-suffix trimming reduced `raw_token_ids len=96` to
  `token_ids_after_trim len=2`.
- S3 received only 2 tokens and produced extremely short audio.

### After tail-trim hardening

- The same request no longer collapsed to 2 tokens.
- S3 received the full token stream.
- The token stream was still degenerate (mostly repeated `4137`), which showed
  the deeper bug was upstream in decode semantics.

### After speech-position fix

- vLLM output became diverse instead of repeating one token.
- Finish reason changed from `length` to `stop`.
- EOS (`6562`) appeared in the output.
- Audio quality became subjectively correct in the Arabic warmup test.

## Operational Notes

- The startup warnings about unknown `VLLM_*` environment variables are expected
  in this service layout. The FastAPI service reads these values itself and
  passes them programmatically into vLLM.
- `VLLM_ENABLE_PREFIX_CACHING` remains guarded off for this embed-only prompt
  path.
- The dynamic stream-client budgeting is heuristic by design. The server still
  makes the final decision about effective whole-request token count for the
  auto-tier path.

