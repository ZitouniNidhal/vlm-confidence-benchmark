from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_reliability_curve(confidences, accuracies, title="Reliability Curve", output_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(confidences, accuracies, marker="o", linestyle="-", label="Model Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    plt.close(fig)
    return fig


def plot_confidence_histogram(confidences, bins=10, title="Confidence Histogram", output_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(confidences, bins=bins, range=(0, 1), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    plt.close(fig)
    return fig


def plot_degradation_effects(severity_levels, metric_values, metric_name="Metric", output_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in metric_values.items():
        ax.plot(severity_levels, values, marker="o", label=label)
    ax.set_xlabel("Severity")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs. Degradation Severity")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    plt.close(fig)
    return fig
