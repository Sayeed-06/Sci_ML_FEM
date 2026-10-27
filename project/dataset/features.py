"""Node-feature schema helpers for legacy and expanded graph caches."""

from __future__ import annotations

import torch


BASE_NODE_FEATURES = (
    "x_norm", "y_norm", "thickness", "E", "k_winkler", "q_local",
    "delta_T", "bc_flag", "D", "Lx", "Ly",
)
EXPANDED_NODE_FEATURES = BASE_NODE_FEATURES + ("alpha_T", "pressure_q")
EXPANDED_NODE_DIM = len(EXPANDED_NODE_FEATURES)


def ensure_expanded_features(data):
    """Append graph-level alpha_T and pressure to every node, if needed."""
    width = int(data.x.shape[1])
    if width == EXPANDED_NODE_DIM:
        return data
    if width != len(BASE_NODE_FEATURES):
        raise ValueError(
            f"Expected {len(BASE_NODE_FEATURES)} or {EXPANDED_NODE_DIM} node "
            f"features, received {width}."
        )
    if not hasattr(data, "alpha_T") or not hasattr(data, "pressure"):
        raise ValueError("Cached graph is missing alpha_T or pressure attributes.")

    n = data.x.shape[0]
    alpha = torch.as_tensor(data.alpha_T, dtype=data.x.dtype, device=data.x.device).reshape(-1)[0]
    pressure = torch.as_tensor(data.pressure, dtype=data.x.dtype, device=data.x.device).reshape(-1)[0]
    extra = torch.stack((alpha.expand(n), pressure.expand(n)), dim=1)
    data.x = torch.cat((data.x, extra), dim=1)
    return data


def ensure_expanded_dataset(data_list):
    """Upgrade each graph in an iterable and return a concrete list."""
    return [ensure_expanded_features(data) for data in data_list]
