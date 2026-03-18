import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from ..models.t3.inference.scheduled_decode import (
    ScheduledDecodeRequest,
    advance_scheduled_cohort,
    build_scheduled_runtime_components,
    prepare_scheduled_cohort,
)


shape_logger = logging.getLogger("chatterbox.shape")


def _trace_t3_enabled() -> bool:
    return bool(os.getenv("CHATTERBOX_TRACE_SHAPES"))


@dataclass
class _PendingScheduledRequest:
    decode_request: ScheduledDecodeRequest
    done: threading.Event = field(default_factory=threading.Event)
    result: torch.Tensor | None = None
    error: Exception | None = None
    submitted_at: float = field(default_factory=time.perf_counter)
    first_started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None
    metrics: dict = field(default_factory=dict)


@dataclass
class _ActiveScheduledCohort:
    cohort_state: object
    pending_by_session: dict[str, _PendingScheduledRequest]


class T3DecodeScheduler:
    """
    Dynamic scheduler for batched T3 decode.

    Requests are still grouped into same-shape cohorts, but cohorts no longer run
    to completion in one call. Instead, the scheduler advances one decode step at
    a time and rotates across active cohorts, which lets new requests enter while
    older cohorts are still decoding.
    """

    def __init__(
        self,
        t3,
        *,
        batching_window_ms: float = 5.0,
        enable_alignment_controller: bool = False,
        hydra_model=None,
        hydra_speculate_k: int = 3,
    ):
        self.t3 = t3
        self.batching_window_ms = batching_window_ms
        self.condition = threading.Condition()
        self.pending: list[_PendingScheduledRequest] = []
        self.active_cohorts: deque[_ActiveScheduledCohort] = deque()
        self.stopped = False
        self.hydra_model = hydra_model
        self.hydra_speculate_k = hydra_speculate_k
        if self.hydra_model is not None and enable_alignment_controller:
            raise ValueError("Hydra scheduled runtime does not support the alignment controller")
        self.patched_model, self.alignment_controller = build_scheduled_runtime_components(
            self.t3,
            enable_alignment_controller=enable_alignment_controller,
        )
        self.worker = threading.Thread(
            target=self._run_loop,
            name="chatterbox-t3-scheduler",
            daemon=True,
        )
        self.worker.start()

    def submit(self, decode_request: ScheduledDecodeRequest) -> tuple[torch.Tensor, dict]:
        pending = _PendingScheduledRequest(decode_request=decode_request)
        with self.condition:
            self.pending.append(pending)
            self.condition.notify()

        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        assert pending.result is not None
        return pending.result, pending.metrics

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.worker.join(timeout=1.0)
        if self.alignment_controller is not None:
            self.alignment_controller.close()

    def _drain_pending(self, *, block_if_idle: bool) -> list[_PendingScheduledRequest] | None:
        with self.condition:
            while block_if_idle and not self.pending and not self.stopped and not self.active_cohorts:
                self.condition.wait()

            if self.stopped and not self.pending and not self.active_cohorts:
                return None

            if not self.pending:
                return []

            deadline = time.perf_counter() + (self.batching_window_ms / 1000.0)
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)

            pending = self.pending
            self.pending = []
            return pending

    def _activate_pending(self, pending_items: list[_PendingScheduledRequest]):
        if not pending_items:
            return

        grouped: dict[tuple[int, int], list[_PendingScheduledRequest]] = {}
        for item in pending_items:
            grouped.setdefault(item.decode_request.batch_key(), []).append(item)

        for batch_key, items in grouped.items():
            decode_requests = [item.decode_request for item in items]
            cohort_state = prepare_scheduled_cohort(self.t3, decode_requests)
            pending_by_session = {item.decode_request.session_id: item for item in items}
            self.active_cohorts.append(
                _ActiveScheduledCohort(
                    cohort_state=cohort_state,
                    pending_by_session=pending_by_session,
                )
            )

            if _trace_t3_enabled():
                shape_logger.info("[runtime/t3_scheduler.py] run_cohort")
                shape_logger.info("  requests %s", len(decode_requests))
                shape_logger.info("  batch_key %s", batch_key)
                shape_logger.info("  sessions %s", [request.session_id for request in decode_requests])
                shape_logger.info("  active_cohorts %s", len(self.active_cohorts))

    def _process_one_step(self, cohort: _ActiveScheduledCohort):
        if _trace_t3_enabled():
            shape_logger.info("[runtime/t3_scheduler.py] step_cohort")
            shape_logger.info("  batch_key %s", cohort.cohort_state.batch_key)
            shape_logger.info("  active_requests %s", len(cohort.cohort_state.active_states))

        step_started_at = time.perf_counter()
        for state in cohort.cohort_state.active_states:
            item = cohort.pending_by_session[state.request.session_id]
            if item.first_started_at is None:
                item.first_started_at = step_started_at

        advance_result = advance_scheduled_cohort(
            self.t3,
            cohort.cohort_state,
            patched_model=self.patched_model,
            alignment_controller=self.alignment_controller,
            hydra_model=self.hydra_model,
            hydra_speculate_k=self.hydra_speculate_k,
        )
        first_token_recorded_at = time.perf_counter()
        for session_id in advance_result.first_token_session_ids:
            item = cohort.pending_by_session.get(session_id)
            if item is not None and item.first_token_at is None:
                item.first_token_at = first_token_recorded_at

        for finished in advance_result.finished_results:
            item = cohort.pending_by_session.pop(finished.session_id)
            item.completed_at = time.perf_counter()
            item.metrics = {
                "t3_wait_s": 0.0 if item.first_started_at is None else item.first_started_at - item.submitted_at,
                "t3_first_token_s": 0.0 if item.first_token_at is None else item.first_token_at - item.submitted_at,
                "t3_active_s": 0.0 if item.first_started_at is None or item.completed_at is None else item.completed_at - item.first_started_at,
                "t3_s": 0.0 if item.completed_at is None else item.completed_at - item.submitted_at,
            }
            item.metrics.update(finished.decode_metrics)
            item.result = finished.speech_tokens
            item.done.set()

        for successor in advance_result.successor_cohorts:
            successor_pending = {
                state.request.session_id: cohort.pending_by_session[state.request.session_id]
                for state in successor.active_states
            }
            self.active_cohorts.append(
                _ActiveScheduledCohort(
                    cohort_state=successor,
                    pending_by_session=successor_pending,
                )
            )

        if _trace_t3_enabled() and len(advance_result.successor_cohorts) > 1:
            shape_logger.info("[runtime/t3_scheduler.py] split_cohort")
            shape_logger.info("  batch_key %s", cohort.cohort_state.batch_key)
            shape_logger.info(
                "  successor_sizes %s",
                [len(successor.active_states) for successor in advance_result.successor_cohorts],
            )
            shape_logger.info(
                "  successor_decode_steps %s",
                [successor.active_states[0].decode_step for successor in advance_result.successor_cohorts],
            )

        if advance_result.successor_cohorts:
            return

        if _trace_t3_enabled():
            shape_logger.info("[runtime/t3_scheduler.py] complete_cohort")
            shape_logger.info("  batch_key %s", cohort.cohort_state.batch_key)
            shape_logger.info("  remaining_active_cohorts %s", len(self.active_cohorts))

    def _fail_cohort(self, cohort: _ActiveScheduledCohort, exc: Exception):
        for item in cohort.pending_by_session.values():
            item.error = exc
            item.done.set()

    def _run_loop(self):
        while True:
            pending_items = self._drain_pending(block_if_idle=not self.active_cohorts)
            if pending_items is None:
                return
            self._activate_pending(pending_items)

            if not self.active_cohorts:
                continue

            cohort = self.active_cohorts.popleft()
            try:
                self._process_one_step(cohort)
            except Exception as exc:  # noqa: BLE001
                self._fail_cohort(cohort, exc)
