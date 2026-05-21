"""Simplified proxy physics metrics for predicted airway fields.

These are not CFD results. They are lightweight indicators for dashboards and
reports, useful when we want quick feedback from the digital twin.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# NumPy computes geometry and pressure statistics.
import numpy as np


# SMALL_EPSILON prevents division by zero in the resistance proxy.
SMALL_EPSILON = 1e-12


def compute_geometry_metrics(points: np.ndarray | None) -> dict[str, float]:
    # Start with an empty metric dictionary.
    metrics: dict[str, float] = {}

    # If points are unavailable, return no geometry metrics.
    if points is None:
        return metrics

    # Compute the minimum x, y, z coordinates of the bounding box.
    bbox_min = np.min(points, axis=0)

    # Compute the maximum x, y, z coordinates of the bounding box.
    bbox_max = np.max(points, axis=0)

    # Compute the centroid, which is the average point location.
    centroid = np.mean(points, axis=0)

    # Compute the width in x direction.
    width_x = bbox_max[0] - bbox_min[0]

    # Compute the width in y direction.
    width_y = bbox_max[1] - bbox_min[1]

    # Compute the vertical extent in z direction.
    vertical_extent = bbox_max[2] - bbox_min[2]

    # Compute each point's radial distance from the centroid in the xy plane.
    radial_distances = np.sqrt((points[:, 0] - centroid[0]) ** 2 + (points[:, 1] - centroid[1]) ** 2)

    # Compute simplified radius proxies.
    mean_radius_proxy = float(np.mean(radial_distances))
    min_radius_proxy = float(np.min(radial_distances))
    max_radius_proxy = float(np.max(radial_distances))

    # Compute a constriction indicator; lower values suggest stronger narrowing.
    constriction_index = min_radius_proxy / mean_radius_proxy if mean_radius_proxy > 0 else np.nan

    # Store bounding box, extent, centroid, and radius proxy metrics.
    metrics["bbox_min_x"] = float(bbox_min[0])
    metrics["bbox_min_y"] = float(bbox_min[1])
    metrics["bbox_min_z"] = float(bbox_min[2])
    metrics["bbox_max_x"] = float(bbox_max[0])
    metrics["bbox_max_y"] = float(bbox_max[1])
    metrics["bbox_max_z"] = float(bbox_max[2])
    metrics["vertical_extent"] = float(vertical_extent)
    metrics["width_x"] = float(width_x)
    metrics["width_y"] = float(width_y)
    metrics["centroid_x"] = float(centroid[0])
    metrics["centroid_y"] = float(centroid[1])
    metrics["centroid_z"] = float(centroid[2])
    metrics["mean_radius_proxy"] = mean_radius_proxy
    metrics["min_radius_proxy"] = min_radius_proxy
    metrics["max_radius_proxy"] = max_radius_proxy
    metrics["constriction_index"] = float(constriction_index)

    # Return all geometry metrics.
    return metrics


def compute_pressure_metrics(pressure: np.ndarray | None) -> dict[str, float]:
    # Start with an empty metric dictionary.
    metrics: dict[str, float] = {}

    # If pressure is unavailable, return no pressure metrics.
    if pressure is None:
        return metrics

    # Flatten pressure so all values are treated as scalar samples.
    values = np.asarray(pressure).ravel()

    # If no pressure values exist, return no pressure metrics.
    if values.size == 0:
        return metrics

    # Compute minimum and maximum predicted pressure.
    pressure_min = float(np.min(values))
    pressure_max = float(np.max(values))

    # Store pressure metrics.
    metrics["pressure_min"] = pressure_min
    metrics["pressure_max"] = pressure_max
    metrics["pressure_mean"] = float(np.mean(values))
    metrics["pressure_std"] = float(np.std(values))
    metrics["pressure_range"] = float(pressure_max - pressure_min)

    # Return all pressure metrics.
    return metrics


def add_proxy_outputs(metrics: dict[str, float]) -> dict[str, float]:
    # Read the mean radius proxy if geometry metrics were available.
    mean_radius = metrics.get("mean_radius_proxy")

    # If mean radius exists, estimate flow capacity using a radius^4 relationship.
    if mean_radius is not None and np.isfinite(mean_radius):
        metrics["flow_capacity_proxy"] = float(mean_radius**4)

    # If mean radius is missing, flow capacity cannot be estimated.
    else:
        metrics["flow_capacity_proxy"] = np.nan

    # Read the flow capacity proxy.
    flow_capacity = metrics["flow_capacity_proxy"]

    # Compute resistance as the inverse of flow capacity, protected by epsilon.
    if np.isfinite(flow_capacity):
        metrics["resistance_proxy"] = float(1.0 / max(flow_capacity, SMALL_EPSILON))

    # If flow capacity is unavailable, resistance is unavailable.
    else:
        metrics["resistance_proxy"] = np.nan

    # If pressure range exists, use it as the pressure drop proxy.
    if "pressure_range" in metrics:
        metrics["pressure_drop_proxy"] = metrics["pressure_range"]

    # Otherwise use resistance as a fallback simplified pressure-drop proxy.
    else:
        metrics["pressure_drop_proxy"] = metrics["resistance_proxy"]

    # Return the updated metric dictionary.
    return metrics


def compute_proxy_metrics(points: np.ndarray | None, pressure: np.ndarray | None) -> dict[str, float]:
    # Compute geometry metrics first.
    metrics = compute_geometry_metrics(points)

    # Add pressure metrics.
    metrics.update(compute_pressure_metrics(pressure))

    # Add flow, resistance, and pressure-drop proxy values.
    return add_proxy_outputs(metrics)
