"""Metric geometry shared by training, rolling priors, sampling and control."""
import torch
from torch.nn import functional as F


SPATIAL_SOURCE_INDICES = (0.0, 1.8, 3.6, 5.4, 7.2, 9.0)
FUTURE_TIMES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def interpolate_nodes(points, source_indices, destination_indices):
    """Resample ordered geometric nodes without changing their covered extent."""
    source = torch.as_tensor(source_indices, device=points.device, dtype=torch.float32)
    destination = torch.as_tensor(destination_indices, device=points.device, dtype=torch.float32)
    if points.shape[-2] != len(source) or source.ndim != 1 or destination.ndim != 1:
        raise ValueError('Geometric node coordinates and sampling indices differ')
    if not bool((source[1:] > source[:-1]).all()):
        raise ValueError('Source indices must be increasing')
    if bool((destination < source[0]).any() | (destination > source[-1]).any()):
        raise ValueError('Geometric interpolation must not extrapolate')
    right = torch.searchsorted(source, destination).clamp(1, len(source) - 1)
    left = right - 1
    weight = (destination - source[left]) / (source[right] - source[left])
    return points[..., left, :] * (1 - weight[..., None]) + points[..., right, :] * weight[..., None]


def spatial_ten_to_six(points):
    if points.shape[-2:] != (10, 2):
        raise ValueError('Expected the original ten geometric points')
    return interpolate_nodes(points, tuple(range(10)), SPATIAL_SOURCE_INDICES)


def spatial_six_to_controller(points):
    """The unchanged lateral controller consumes its original geometric lattice."""
    if points.shape[-2:] != (6, 2):
        raise ValueError('Expected six predicted geometric points')
    return interpolate_nodes(points, SPATIAL_SOURCE_INDICES, tuple(range(10)))


def align_history(history, source_to_current, frame_valid, extent=32.0):
    """Pull observed BEVs into the fixed current ego frame, with pixel masks."""
    b, n, c, h, w = history.shape
    if source_to_current.shape != (b, n, 3, 3) or frame_valid.shape != (b, n):
        raise ValueError('History transform or validity shape differs')
    with torch.autocast(device_type=history.device.type, enabled=False):
        inverse = torch.linalg.inv(source_to_current.float()).reshape(b * n, 3, 3)
        axis_x = (torch.arange(w, device=history.device).float() + .5) * (2 * extent / w) - extent
        axis_y = (torch.arange(h, device=history.device).float() + .5) * (2 * extent / h) - extent
        yy, xx = torch.meshgrid(axis_y, axis_x, indexing='ij')
        grid = torch.stack((xx, yy, torch.ones_like(xx)), -1)
        source = torch.einsum('bij,hwj->bhwi', inverse, grid)[..., :2]
        limits = source.new_tensor([extent - extent / w, extent - extent / h])
        valid = (source.abs() <= limits + 1e-5).all(-1)
        valid &= frame_valid.reshape(b * n, 1, 1)
        aligned = F.grid_sample(history.float().reshape(b * n, c, h, w), source / extent,
                                align_corners=False, padding_mode='zeros')
        aligned = aligned * valid[:, None]
    return aligned.reshape(b, n, c, h, w), valid.reshape(b, n, 1, h, w)


def cumulative_valid(valid):
    """An increment/recursive prediction is usable only with its full prefix."""
    return valid.to(torch.int64).cumprod(-1).bool()


def motion_features(history, valid, speed, prior_delta, prior_age):
    """Twenty observed/prior features; no future labels or future poses."""
    b = len(history)
    pooled = history.float().mean((-1, -2, -3))
    differences = history[:, 1:].float() - history[:, :-1].float()
    mean_change = differences.mean((-1, -2, -3))
    rms_change = differences.square().mean((-1, -2, -3)).sqrt()
    interval_speed = prior_delta.float().norm(dim=-1) / .5
    inputs = torch.cat((speed.float().reshape(b, 1) / 15, prior_age.float().reshape(b, 1),
                        valid.float().reshape(b, 3), pooled, mean_change, rms_change,
                        interval_speed / 15, prior_delta[:, 0].float() / 5), -1)
    if inputs.shape != (b, 20):
        raise ValueError('Observed motion descriptor must have twenty dimensions')
    return inputs
