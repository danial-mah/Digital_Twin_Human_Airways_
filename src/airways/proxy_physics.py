"""Simplified proxy physics metrics for predicted airway fields.

Important:
These are not real CFD results.

CFD, or Computational Fluid Dynamics, solves detailed fluid equations. This
module does not do that. Instead, it computes simple numerical indicators from
the predicted geometry and pressure arrays. These indicators are useful for a
dashboard because they are fast, easy to understand, and help compare designs.
"""

# This line allows Python type hints to be handled in a modern, flexible way.
from __future__ import annotations

# NumPy is the main numerical library used for arrays and mathematical formulas.
import numpy as np


# SMALL_EPSILON is a very small number used to avoid division by zero.
# We use it when computing resistance_proxy, because dividing by zero would crash.
SMALL_EPSILON = 1e-12


def compute_geometry_metrics(points: np.ndarray | None) -> dict[str, float]:
    """Compute simple geometry metrics from x, y, z point coordinates.

    The expected input shape is:
        (number_of_points, 3)

    Each row is one point:
        [x, y, z]
    """

    # Create an empty dictionary where metric names and metric values will be stored.
    metrics: dict[str, float] = {}

    # If points is None, geometry was missing or could not be reshaped.
    # In that case, this function cannot calculate geometry metrics.
    if points is None:
        return metrics

    # np.min(points, axis=0) calculates the minimum value in each coordinate column.
    # Result:
    #   bbox_min[0] = minimum x
    #   bbox_min[1] = minimum y
    #   bbox_min[2] = minimum z
    bbox_min = np.min(points, axis=0)

    # np.max(points, axis=0) calculates the maximum value in each coordinate column.
    # Result:
    #   bbox_max[0] = maximum x
    #   bbox_max[1] = maximum y
    #   bbox_max[2] = maximum z
    bbox_max = np.max(points, axis=0)

    # np.mean(points, axis=0) calculates the average x, y, and z coordinate.
    # This gives the approximate center of the point cloud.
    centroid = np.mean(points, axis=0)

    # Width in x is the distance between the maximum x and minimum x.
    width_x = bbox_max[0] - bbox_min[0]

    # Width in y is the distance between the maximum y and minimum y.
    width_y = bbox_max[1] - bbox_min[1]

    # Vertical extent is the distance between the maximum z and minimum z.
    vertical_extent = bbox_max[2] - bbox_min[2]

    # This computes distance from each point to the centroid in the xy plane only.
    # We ignore z here because this is a simple radius-like proxy around the airway axis.
    radial_distances = np.sqrt((points[:, 0] - centroid[0]) ** 2 + (points[:, 1] - centroid[1]) ** 2)

    # The mean radius proxy is the average xy distance from the centroid.
    # It is not a medical airway radius; it is a simple geometry indicator.
    mean_radius_proxy = float(np.mean(radial_distances))

    # The minimum radius proxy is the smallest xy distance from the centroid.
    # A small value can indicate a narrow region or points close to the center.
    min_radius_proxy = float(np.min(radial_distances))

    # The maximum radius proxy is the largest xy distance from the centroid.
    max_radius_proxy = float(np.max(radial_distances))

    # The constriction index compares the minimum radius proxy to the average radius proxy.
    # A smaller value suggests stronger narrowing relative to the overall size.
    if mean_radius_proxy > 0:
        constriction_index = min_radius_proxy / mean_radius_proxy
    else:
        constriction_index = np.nan

    # Save bounding box minimum coordinates into the metrics dictionary.
    metrics["bbox_min_x"] = float(bbox_min[0])
    metrics["bbox_min_y"] = float(bbox_min[1])
    metrics["bbox_min_z"] = float(bbox_min[2])

    # Save bounding box maximum coordinates into the metrics dictionary.
    metrics["bbox_max_x"] = float(bbox_max[0])
    metrics["bbox_max_y"] = float(bbox_max[1])
    metrics["bbox_max_z"] = float(bbox_max[2])

    # Save overall size measurements.
    metrics["vertical_extent"] = float(vertical_extent)
    metrics["width_x"] = float(width_x)
    metrics["width_y"] = float(width_y)

    # Save centroid coordinates.
    metrics["centroid_x"] = float(centroid[0])
    metrics["centroid_y"] = float(centroid[1])
    metrics["centroid_z"] = float(centroid[2])

    # Save radius-related proxy metrics.
    metrics["mean_radius_proxy"] = mean_radius_proxy
    metrics["min_radius_proxy"] = min_radius_proxy
    metrics["max_radius_proxy"] = max_radius_proxy
    metrics["constriction_index"] = float(constriction_index)

    # Return all calculated geometry metrics.
    return metrics


def compute_pressure_metrics(pressure: np.ndarray | None) -> dict[str, float]:
    """Compute simple pressure statistics from a predicted pressure field."""

    # Create an empty dictionary where pressure metric values will be stored.
    metrics: dict[str, float] = {}

    # If pressure is None, there is no pressure prediction to summarize.
    if pressure is None:
        return metrics

    # np.asarray converts the input into a NumPy array if it is not already one.
    # ravel() flattens it into one long vector.
    # This is useful because pressure may be shaped as (N,) or (N, 1).
    values = np.asarray(pressure).ravel()

    # If the pressure array is empty, there are no values to summarize.
    if values.size == 0:
        return metrics

    # pressure_min is the lowest predicted pressure value.
    pressure_min = float(np.min(values))

    # pressure_max is the highest predicted pressure value.
    pressure_max = float(np.max(values))

    # pressure_mean is the average predicted pressure.
    pressure_mean = float(np.mean(values))

    # pressure_std is the standard deviation, showing how spread out values are.
    pressure_std = float(np.std(values))

    # pressure_range is max minus min.
    # It is used later as a simplified pressure-drop proxy.
    pressure_range = float(pressure_max - pressure_min)

    # Store pressure metrics in the dictionary.
    metrics["pressure_min"] = pressure_min
    metrics["pressure_max"] = pressure_max
    metrics["pressure_mean"] = pressure_mean
    metrics["pressure_std"] = pressure_std
    metrics["pressure_range"] = pressure_range

    # Return all calculated pressure metrics.
    return metrics


def add_proxy_outputs(metrics: dict[str, float]) -> dict[str, float]:
    """Add simplified flow, resistance, and pressure-drop proxy values."""

    # Read mean_radius_proxy from the metrics dictionary.
    # It may be missing if geometry metrics could not be computed.
    mean_radius = metrics.get("mean_radius_proxy")

    # A common simplified fluid-flow relationship says flow capacity scales
    # strongly with radius, often approximately radius^4 in idealized tubes.
    # This is only a proxy here, not a full physical calculation.
    if mean_radius is not None and np.isfinite(mean_radius):
        metrics["flow_capacity_proxy"] = float(mean_radius**4)
    else:
        metrics["flow_capacity_proxy"] = np.nan

    # Read the newly created flow capacity value.
    flow_capacity = metrics["flow_capacity_proxy"]

    # Resistance is treated as the inverse of flow capacity.
    # max(flow_capacity, SMALL_EPSILON) prevents division by zero.
    if np.isfinite(flow_capacity):
        metrics["resistance_proxy"] = float(1.0 / max(flow_capacity, SMALL_EPSILON))
    else:
        metrics["resistance_proxy"] = np.nan

    # If pressure data exists, pressure_range is the best simple pressure-drop proxy.
    if "pressure_range" in metrics:
        metrics["pressure_drop_proxy"] = metrics["pressure_range"]

    # If pressure does not exist, use resistance_proxy as a fallback indicator.
    else:
        metrics["pressure_drop_proxy"] = metrics["resistance_proxy"]

    # Return the same dictionary after adding the new proxy outputs.
    return metrics


def compute_proxy_metrics(points: np.ndarray | None, pressure: np.ndarray | None) -> dict[str, float]:
    """Compute all proxy metrics from optional geometry points and pressure values."""

    # First compute geometry metrics such as bounding box, centroid, and radius proxies.
    metrics = compute_geometry_metrics(points)

    # Then compute pressure metrics and merge them into the same dictionary.
    metrics.update(compute_pressure_metrics(pressure))

    # Finally add flow_capacity_proxy, resistance_proxy, and pressure_drop_proxy.
    return add_proxy_outputs(metrics)
