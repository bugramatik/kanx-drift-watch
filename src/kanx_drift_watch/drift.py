"""Drift monitor for KAN models.

A KAN layer only knows the range its knots were fitted to. Outside that
range the B-spline basis is zero and the layer falls back to its SiLU base
path, so predictions keep coming back while the spline part silently stops
contributing. ``DriftMonitor`` reads the per-feature grid of every KAN edge
bank in a model and reports, per request, which input features left the
trained range and by how much.

The monitor is read-only: it never touches weights, and it works on the
TensorFlow ``KANLinear``, the PyTorch ``KANLinear`` and ``MatrixKANLinear``
layers, because all three keep a ``grid`` of shape
``(in_features, grid_size + 1)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .viz import kan_layers


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class FeatureRange:
    """Trained knot range of one input feature, across every layer that reads it."""

    feature: int
    low: float
    high: float
    layers: tuple[int, ...]

    def excess(self, values: np.ndarray) -> np.ndarray:
        """Distance of each value outside ``[low, high]``; zero inside."""
        values = np.asarray(values, dtype=np.float64)
        return np.maximum(np.maximum(self.low - values, values - self.high), 0.0)


class DriftMonitor:
    """Compare predict requests against the trained grid of a KAN model.

    Args:
        model: a KAN model, or a single KAN edge bank. The layers are found
            with :func:`kanx_drift_watch.viz.kan_layers`.
        tolerance: fraction of each feature's trained span that is allowed
            outside the knots before the feature counts as drifted. ``0.0``
            flags any value outside the grid.

    Raises:
        TypeError: when the model holds no KAN layer.
        ValueError: when ``tolerance`` is negative or a layer is not built.
    """

    def __init__(self, model: Any, tolerance: float = 0.0) -> None:
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self.tolerance = float(tolerance)
        self.layers = kan_layers(model)
        self._ranges = self._read_ranges()
        self.in_features = len(self._ranges)

    # ---- construction ---------------------------------------------------
    def _read_ranges(self) -> list[FeatureRange]:
        lows: dict[int, float] = {}
        highs: dict[int, float] = {}
        owners: dict[int, list[int]] = {}
        for index, layer in enumerate(self.layers):
            try:
                grid = _to_numpy(layer.grid)
            except (AttributeError, ValueError) as err:
                raise ValueError(
                    f"layer {index} ({type(layer).__name__}) has no readable grid; "
                    "build the model before creating a DriftMonitor"
                ) from err
            if grid.ndim != 2 or grid.shape[1] < 2:
                raise ValueError(
                    f"layer {index} grid must be (in_features, grid_size + 1), "
                    f"got {tuple(grid.shape)}"
                )
            for feature in range(grid.shape[0]):
                low = float(np.min(grid[feature]))
                high = float(np.max(grid[feature]))
                lows[feature] = min(lows.get(feature, low), low)
                highs[feature] = max(highs.get(feature, high), high)
                owners.setdefault(feature, []).append(index)
        return [
            FeatureRange(
                feature=feature,
                low=lows[feature],
                high=highs[feature],
                layers=tuple(owners[feature]),
            )
            for feature in sorted(lows)
        ]

    # ---- inspection -----------------------------------------------------
    def feature_ranges(self) -> list[dict[str, Any]]:
        """Per-feature trained range, as plain dicts for JSON responses."""
        return [
            {
                "feature": r.feature,
                "low": round(r.low, 6),
                "high": round(r.high, 6),
                "layers": list(r.layers),
            }
            for r in self._ranges
        ]

    def check(self, x: Any) -> dict[str, Any]:
        """Score a batch of shape ``(batch, in_features)``.

        Returns a dict with ``drifted`` (whether any feature left its range),
        ``flagged`` (the features that did, in order), ``excess`` (per flagged
        feature, the largest distance outside the range), ``samples`` (how
        many rows of the batch were outside) and ``max_excess`` (the largest
        excess over the whole batch, zero when nothing drifted).
        """
        batch = np.asarray(x, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        if batch.ndim != 2:
            raise ValueError(f"x must be 1-D or 2-D, got shape {tuple(batch.shape)}")
        if batch.shape[1] != self.in_features:
            raise ValueError(
                f"expected {self.in_features} features per sample, got {batch.shape[1]}"
            )
        if batch.shape[0] == 0:
            raise ValueError("x must contain at least one sample")
        if not np.isfinite(batch).all():
            raise ValueError("x must be finite")

        flagged: list[int] = []
        excess: dict[int, float] = {}
        outside_rows = np.zeros(batch.shape[0], dtype=bool)
        for r in self._ranges:
            span = r.high - r.low
            allowed = self.tolerance * span if span > 0 else 0.0
            over = r.excess(batch[:, r.feature]) - allowed
            over = np.maximum(over, 0.0)
            if over.max() > 0:
                flagged.append(r.feature)
                excess[r.feature] = round(float(over.max()), 6)
                outside_rows |= over > 0

        return {
            "drifted": bool(flagged),
            "flagged": flagged,
            "excess": excess,
            "samples": int(outside_rows.sum()),
            "max_excess": round(max(excess.values()), 6) if excess else 0.0,
        }
