from __future__ import annotations

from typing import Any

from PIL import Image

from degradations.blur import apply_blur
from degradations.fog import apply_fog
from degradations.glare import apply_glare
from degradations.jpeg import apply_jpeg_compression
from degradations.lowlight import apply_lowlight
from degradations.noise import apply_noise
from degradations.resample import apply_resample
from degradations.rotation import apply_rotation


DEGRADATIONS = {
    "blur": apply_blur,
    "fog": apply_fog,
    "glare": apply_glare,
    "jpeg": apply_jpeg_compression,
    "lowlight": apply_lowlight,
    "noise": apply_noise,
    "resample": apply_resample,
    "rotation": apply_rotation,
}

DEFAULT_SEVERITY_PARAMETERS = {
    "blur": {
        "low": {"radius": 1.0},
        "mid": {"radius": 2.0},
        "high": {"radius": 4.0},
    },
    "jpeg": {
        "low": {"quality": 90},
        "mid": {"quality": 70},
        "high": {"quality": 40},
    },
    "lowlight": {
        "low": {"factor": 0.8},
        "mid": {"factor": 0.5},
        "high": {"factor": 0.3},
    },
    "glare": {
        "low": {"factor": 1.2},
        "mid": {"factor": 1.5},
        "high": {"factor": 2.0},
    },
    "resample": {
        "low": {"scale": 0.8},
        "mid": {"scale": 0.6},
        "high": {"scale": 0.4},
    },
    "rotation": {
        "low": {"angle": 5.0},
        "mid": {"angle": 15.0},
        "high": {"angle": 30.0},
    },
    "noise": {
        "low": {"amount": 0.02},
        "mid": {"amount": 0.05},
        "high": {"amount": 0.1},
    },
    "fog": {
        "low": {"opacity": 0.15},
        "mid": {"opacity": 0.30},
        "high": {"opacity": 0.45},
    },
}


def _resolve_operation_params(name: str, params: dict[str, Any] | str | None) -> dict[str, Any]:
    if params is None:
        return {}

    if isinstance(params, str):
        severity = params.lower()
        if name not in DEFAULT_SEVERITY_PARAMETERS:
            raise ValueError(f"Severity presets are not defined for degradation: {name}")
        if severity not in DEFAULT_SEVERITY_PARAMETERS[name]:
            raise ValueError(f"Unknown severity level '{severity}' for degradation '{name}'.")
        return DEFAULT_SEVERITY_PARAMETERS[name][severity]

    if isinstance(params, dict):
        if "severity" in params:
            severity = str(params["severity"]).lower()
            if name not in DEFAULT_SEVERITY_PARAMETERS or severity not in DEFAULT_SEVERITY_PARAMETERS[name]:
                raise ValueError(f"Unknown severity level '{severity}' for degradation '{name}'.")
            resolved = DEFAULT_SEVERITY_PARAMETERS[name][severity].copy()
            resolved.update({k: v for k, v in params.items() if k != "severity"})
            return resolved
        return params

    raise TypeError("Degradation parameters must be a dict, a severity string, or None.")


def apply_degradations(image: Image.Image, operations: list[tuple[str, dict[str, Any] | str]] | None = None) -> Image.Image:
    """Apply a sequence of degradations to an image in order."""
    current = image.convert("RGB")

    for name, params in operations or []:
        if name not in DEGRADATIONS:
            raise ValueError(f"Unknown degradation: {name}")
        current = DEGRADATIONS[name](current, **_resolve_operation_params(name, params))

    return current
