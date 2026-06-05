from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProbeComponent:
    x: float
    y: float
    area: int = 1
    score: float = 1.0


@dataclass
class ProbeCountResult:
    red_count: int
    green_count: int
    red_components: list[ProbeComponent]
    green_components: list[ProbeComponent]
    backend: str = "cv"


def _spots(channel, threshold: float, min_area: int = 2, mask=None, nms_distance: int | None = None):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    ch = channel.astype("float32") / 255.0
    if mask is not None:
        ch = ch * mask.astype("float32")
    thr = float(max(0.0, min(1.0, threshold)))
    binary = (ch >= thr).astype("uint8")
    if mask is not None:
        binary = binary & mask.astype("uint8")
    # Group connected bright pixels first. The previous local-maximum method
    # could mark several peaks inside one physical probe spot.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    candidates: list[ProbeComponent] = []
    min_area = max(1, int(min_area))
    dist = int(nms_distance if nms_distance is not None else max(8, round(min_area ** 0.5 * 4)))
    split_area = max(min_area * 20, dist * dist * 4)

    def add_one_component(component_ys, component_xs, area: int) -> None:
        vals = ch[component_ys, component_xs]
        best = int(np.argmax(vals))
        candidates.append(ProbeComponent(x=float(component_xs[best]), y=float(component_ys[best]), area=area, score=float(vals[best])))

    def add_split_component(component_mask, area: int) -> None:
        vals = ch * component_mask.astype("float32")
        comp_max = float(vals.max())
        if comp_max <= 0:
            return
        # Large merged spots are split only by strong local peaks. This keeps a
        # single probe from becoming many labels, while allowing red-red or
        # green-green touching probes to become multiple counts.
        local = cv2.dilate(vals, np.ones((5, 5), dtype=np.float32))
        peak_mask = (vals >= local - 1e-6) & (vals >= max(thr, comp_max * 0.92)) & component_mask
        ys2, xs2 = np.where(peak_mask)
        if xs2.size <= 1:
            ys1, xs1 = np.where(component_mask)
            add_one_component(ys1, xs1, area)
            return
        order = np.argsort(vals[ys2, xs2])[::-1]
        max_peaks = max(1, min(2, int(round(area / max(min_area * 12, 1)))))
        local_kept: list[ProbeComponent] = []
        d2_local = float(max(1, dist) * 1.5) ** 2
        for oi in order:
            cand = ProbeComponent(x=float(xs2[oi]), y=float(ys2[oi]), area=area, score=float(vals[ys2[oi], xs2[oi]]))
            if all((cand.x - old.x) ** 2 + (cand.y - old.y) ** 2 >= d2_local for old in local_kept):
                local_kept.append(cand)
            if len(local_kept) >= max_peaks:
                break
        candidates.extend(local_kept or [])

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        if area >= split_area:
            add_split_component(labels == i, area)
        else:
            add_one_component(ys, xs, area)

    d2 = float(max(1, dist)) ** 2
    kept: list[ProbeComponent] = []
    for cand in sorted(candidates, key=lambda p: p.score, reverse=True):
        if all((cand.x - old.x) ** 2 + (cand.y - old.y) ** 2 >= d2 for old in kept):
            kept.append(cand)
    return kept


def _nucleus_mask(rgb, min_size: int = 200, dilation_radius: int = 3):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    blue = rgb[..., 2].astype("uint8")
    _, m = cv2.threshold(blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_size):
            keep[labels == i] = 255
    if dilation_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
        keep = cv2.dilate(keep, k)
    return keep > 0


def render_probe_overlay(rgb, red_components, green_components):
    import cv2  # type: ignore

    def radius_for(component: ProbeComponent) -> int:
        import math

        area = max(1, int(getattr(component, "area", 1)))
        return max(2, min(6, int(round(math.sqrt(area / math.pi) + 1))))

    out = rgb.copy()
    for p in red_components:
        cv2.circle(out, (int(round(p.x)), int(round(p.y))), radius_for(p), (255, 0, 0), 1, cv2.LINE_AA)
    for p in green_components:
        cv2.circle(out, (int(round(p.x)), int(round(p.y))), radius_for(p), (0, 255, 0), 1, cv2.LINE_AA)
    return out


class UFishCounter:
    def __init__(self, onnx_path: Path | None = None, pth_path: Path | None = None, device: str = "0", prefer_onnx: bool = True):
        self.backend = "cv"
        self.device = device

    def count_many(
        self,
        crops_rgb: list,
        *,
        red_threshold: float = 0.5,
        green_threshold: float = 0.5,
        min_area: int = 2,
        nms_distance: int | None = 5,
        nucleus_min_size: int = 200,
        nucleus_dilation_radius: int = 3,
        batch_size: int = 32,
        input_size: int = 256,
        cancel_cb=None,
    ):
        import numpy as np

        out = []
        for crop in crops_rgb:
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            arr = np.asarray(crop)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=-1)
            arr = arr[:, :, :3].astype("uint8", copy=False)
            mask = _nucleus_mask(arr, nucleus_min_size, nucleus_dilation_radius)
            red = _spots(arr[..., 0], red_threshold, min_area=int(min_area), mask=mask, nms_distance=nms_distance)
            green = _spots(arr[..., 1], green_threshold, min_area=int(min_area), mask=mask, nms_distance=nms_distance)
            overlay = render_probe_overlay(arr, red, green)
            out.append((ProbeCountResult(len(red), len(green), red, green, self.backend), overlay))
        return out
