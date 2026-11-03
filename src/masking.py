"""Logits masking logic for constrained decoding."""

def mask_logits(logits: list[float], allowed_ids: set[int]) -> list[float]:
    """Mask logits by setting disallowed token IDs to negative infinity."""
    return [val if i in allowed_ids else float("-inf") for i, val in enumerate(logits)]