import logging
import threading
import time
from dataclasses import dataclass, field

import torch

from ..models.t3.inference.scheduled_decode import ScheduledDecodeRequest, run_scheduled_t3_batch


shape_logger = logging.getLogger("chatterbox.shape")


@dataclass
class _PendingScheduledRequest:
    decode_request: ScheduledDecodeRequest
    done: threading.Event = field(default_factory=threading.Event)
    result: torch.Tensor | None = None
    error: Exception | None = None


class T3DecodeScheduler:
    """
    First-pass scheduler for batching T3 decode across multiple concurrent requests.

    This implementation groups pending requests with matching batch keys into a
    cohort, runs one batched T3 decode to completion, and then releases each
    request back to its caller for S3 inference.
    """

    def __init__(self, t3, *, batching_window_ms: float = 5.0):
        self.t3 = t3
        self.batching_window_ms = batching_window_ms
        self.condition = threading.Condition()
        self.pending: list[_PendingScheduledRequest] = []
        self.stopped = False
        self.worker = threading.Thread(
            target=self._run_loop,
            name="chatterbox-t3-scheduler",
            daemon=True,
        )
        self.worker.start()

    def submit(self, decode_request: ScheduledDecodeRequest) -> torch.Tensor:
        pending = _PendingScheduledRequest(decode_request=decode_request)
        with self.condition:
            self.pending.append(pending)
            self.condition.notify()

        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        assert pending.result is not None
        return pending.result

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.worker.join(timeout=1.0)

    def _pop_cohort(self) -> list[_PendingScheduledRequest] | None:
        with self.condition:
            while not self.pending and not self.stopped:
                self.condition.wait()

            if self.stopped and not self.pending:
                return None

            first = self.pending.pop(0)
            cohort = [first]
            batch_key = first.decode_request.batch_key()
            deadline = time.perf_counter() + (self.batching_window_ms / 1000.0)

            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                if not self.pending:
                    self.condition.wait(timeout=remaining)
                matched = []
                still_pending = []
                for pending in self.pending:
                    if pending.decode_request.batch_key() == batch_key:
                        matched.append(pending)
                    else:
                        still_pending.append(pending)
                self.pending = still_pending
                cohort.extend(matched)
                if not matched and not self.pending:
                    break

            return cohort

    def _run_loop(self):
        while True:
            cohort = self._pop_cohort()
            if cohort is None:
                return
            self._process_cohort(cohort)

    def _process_cohort(self, cohort: list[_PendingScheduledRequest]):
        decode_requests = [item.decode_request for item in cohort]
        if shape_logger.isEnabledFor(logging.INFO):
            shape_logger.info("[runtime/t3_scheduler.py] run_cohort")
            shape_logger.info("  requests %s", len(decode_requests))
            shape_logger.info("  batch_key %s", decode_requests[0].batch_key())
            shape_logger.info("  sessions %s", [request.session_id for request in decode_requests])
        try:
            results = run_scheduled_t3_batch(self.t3, decode_requests)
            for item, result in zip(cohort, results):
                item.result = result
                item.done.set()
        except Exception as exc:  # noqa: BLE001
            for item in cohort:
                item.error = exc
                item.done.set()
