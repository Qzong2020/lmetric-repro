"""Request-level TTFT/TPOT from observed content timing and true token counts."""
from dataclasses import dataclass


@dataclass
class StreamMetrics:
    sent_ms: float
    first_content_ms: float | None = None
    last_content_ms: float | None = None
    output_tokens: int | None = None
    last_event_ms: float | None = None
    finished: bool = False

    def observe(self, event, received_ms):
        if received_ms < self.sent_ms or (self.last_event_ms is not None and received_ms < self.last_event_ms):
            raise ValueError("timestamps must be monotonic")
        self.last_event_ms = received_ms
        if event == "[DONE]":
            self.finished = True
            return
        if "error" in event:
            raise ValueError("failed requests cannot contribute latency samples")
        usage = event.get("usage")
        if usage is not None and usage.get("completion_tokens") is not None:
            self.output_tokens = int(usage["completion_tokens"])
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("only single-choice streams are supported")
            if choice.get("delta", {}).get("content"):
                if self.first_content_ms is None:
                    self.first_content_ms = received_ms
                self.last_content_ms = received_ms

    def result(self, authoritative_output_tokens=None):
        if not self.finished:
            raise ValueError("stream did not complete")
        count = self.output_tokens if authoritative_output_tokens is None else authoritative_output_tokens
        if count is not None and (not isinstance(count, int) or count < 0):
            raise ValueError("invalid token count")
        ttft = None if self.first_content_ms is None else self.first_content_ms - self.sent_ms
        tpot = None
        if count is not None and count > 1 and self.first_content_ms is not None:
            tpot = (self.last_content_ms - self.first_content_ms) / (count - 1)
        return {"ttft_ms": ttft, "tpot_ms": tpot, "output_tokens": count,
                "e2e_ms": self.last_event_ms - self.sent_ms,
                "count_source": "usage" if authoritative_output_tokens is None and count is not None
                                else "external" if count is not None else "missing"}


def empirical_cdf(values):
    values = sorted(values)
    return [(value, (i + 1) / len(values)) for i, value in enumerate(values)]
