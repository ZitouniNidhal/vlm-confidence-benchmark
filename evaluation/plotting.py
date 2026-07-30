from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_curve(confidences, accuracies, title="Reliability Curve"):
    plt.figure(figsize=(6, 6))
    plt.plot(confidences, accuracies, marker="o", linestyle="-", label="Reliability")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


def plot_confidence_histogram(confidences, bins=10, title="Confidence Histogram"):
    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=bins, range=(0, 1), edgecolor="black")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    return plt


def plot_degradation_effects(severity_levels, metric_values, metric_name="Metric"):
    plt.figure(figsize=(8, 4))
    for label, values in metric_values.items():
        plt.plot(severity_levels, values, marker="o", label=label)
    plt.xlabel("Severity")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs. Degradation Severity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt
