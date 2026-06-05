from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_from_directory


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_UI_CELLS = 500


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H%M%S")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    return name or f"upload_{_timestamp()}.png"


def _safe_rel_parts(filename: str) -> list[str]:
    raw_parts = str(filename).replace("\\", "/").split("/")
    parts: list[str] = []
    for part in raw_parts:
        safe = _safe_name(part)
        if safe not in {"", ".", ".."}:
            parts.append(safe)
    return parts


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def _discover_best_yolo_weight(project_root: Path) -> Path:
    """Pick the default YOLO weight requested for the web UI."""

    default_goal = project_root / "wights" / "goal" / "best.pt"
    if default_goal.exists():
        return default_goal
    legacy_goal = project_root / "wights" / "goal" / "best_goal.pt"
    if legacy_goal.exists():
        return legacy_goal
    organized_root = project_root / "wights" / "yolo"
    preferred = organized_root / "03_map838_R765_20260422_143619_best.pt"
    if preferred.exists():
        return preferred
    candidates = sorted(organized_root.glob("*.pt"))
    if candidates:
        return candidates[0]
    return default_goal


def _discover_normal_weight(project_root: Path) -> Path:
    preferred = project_root / "wights" / "normal" / "best_nomal.pt"
    if preferred.exists():
        return preferred
    candidates = sorted((project_root / "wights" / "normal").glob("*.pt"))
    if candidates:
        return candidates[0]
    return preferred


def _url_for_output(rel_path: str) -> str:
    rel = str(rel_path).replace("\\", "/").lstrip("/")
    return f"/outputs/{rel}"


def _hydrate_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data = json.loads(json.dumps(cells, ensure_ascii=False))
    for cell in data:
        cell.pop("red_components", None)
        cell.pop("green_components", None)
        if cell.get("crop_path"):
            cell["crop_url"] = _url_for_output(cell["crop_path"])
        if cell.get("ufish_path"):
            cell["ufish_url"] = _url_for_output(cell["ufish_path"])
    return data


def _hydrate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(manifest, ensure_ascii=False))
    for item in data.get("yolo_images", []) or []:
        if item.get("path"):
            item["url"] = _url_for_output(item["path"])
    all_cells = data.get("cells", []) or []
    class_counts: dict[str, int] = {"CTC": 0, "CEC": 0, "CELL": 0}
    for cell in all_cells:
        cls = str(cell.get("class", "")).upper()
        if cls in class_counts:
            class_counts[cls] += 1
    data["ui_cells_total"] = len(all_cells)
    data["ui_class_counts"] = class_counts
    data["ui_cells_limited"] = len(all_cells) > MAX_UI_CELLS
    data["cells"] = _hydrate_cells(all_cells[:MAX_UI_CELLS])
    exports = data.get("exports") or {}
    if exports.get("csv"):
        exports["csv_url"] = _url_for_output(exports["csv"])
    if exports.get("json"):
        exports["json_url"] = _url_for_output(exports["json"])
    return data


def _image_manifest_from_dir(
    *,
    out_root: Path,
    image_dir: Path,
    mode: str,
    config: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    images = []
    outputs_root = None
    for parent in [out_root.resolve(), *out_root.resolve().parents]:
        if parent.name.lower() == "outputs":
            outputs_root = parent
            break
    if outputs_root is None:
        outputs_root = out_root.resolve()

    for p in sorted(image_dir.glob("*.png")):
        try:
            rel = str(p.resolve().relative_to(outputs_root)).replace("\\", "/")
        except ValueError:
            rel = str(p).replace("\\", "/")
        images.append({"name": p.name, "path": rel, "has_detection": True, "mode": mode})

    manifest = {
        "schema_version": 2,
        "output": str(out_root),
        "summary": summary or {"total_cells": 0, "ctc_total": 0, "cec_total": 0, "red_probe_total": 0, "green_probe_total": 0},
        "config": config,
        "yolo_images": images,
        "cells": [],
        "exports": {},
    }
    export_dir = out_root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "results.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def create_app(project_root: Path, data_root: Path | None = None) -> Flask:
    import sys

    data_root = Path(data_root or project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(data_root) not in sys.path:
        sys.path.insert(0, str(data_root))

    template_dir = data_root / "webapp" / "backend" / "templates"
    static_dir = data_root / "webapp" / "backend" / "static"
    if not template_dir.exists():
        template_dir = Path(__file__).parent / "templates"
    if not static_dir.exists():
        static_dir = Path(__file__).parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir.resolve()),
        static_folder=str(static_dir.resolve()),
    )

    default_yolo_weight = _discover_best_yolo_weight(data_root)
    default_normal_weight = _discover_normal_weight(data_root)
    default_ufish_onnx = data_root / "wights" / "ufish" / "01_ufish_c32.onnx"
    default_ufish_pth = data_root / "wights" / "ufish" / "01_ufish_c32.pth"
    if not default_ufish_onnx.exists():
        default_ufish_onnx = data_root / "v1.0-alldata-ufish_c32.onnx"
    if not default_ufish_pth.exists():
        default_ufish_pth = data_root / "v1.0-alldata-ufish_c32.pth"

    state: dict[str, Any] = {
        "running": False,
        "cancel": False,
        "last": {"status": "idle", "stage": "idle", "message": "等待任务"},
        "logs": [],
        "current_output": None,
    }

    def push(event: dict[str, Any]) -> None:
        state["last"] = event
        state["logs"].append(event)
        if len(state["logs"]) > 4000:
            state["logs"] = state["logs"][-4000:]
        try:
            stage = str(event.get("stage", ""))
            if stage in {"start", "config", "tiling", "predict", "two_stage_normal", "two_stage_goal", "two_stage", "normal_only", "ufish", "summary", "done", "error", "cancel", "cleanup"}:
                print("[EVENT]", json.dumps(event, ensure_ascii=False))
        except Exception:
            pass

    def find_latest_output(outputs_root: Path) -> Path | None:
        if not outputs_root.exists():
            return None
        dirs = [p for p in outputs_root.iterdir() if p.is_dir() and p.name != "uploads"]
        if not dirs:
            return None
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0]

    def load_results(out_dir: Path) -> dict[str, Any]:
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            try:
                return _hydrate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001
                return {"output": str(out_dir), "summary": {}, "cells": [], "yolo_images": [], "error": str(exc)}

        # Compatibility with old/running outputs: before manifest.json exists,
        # only expose main result images. Never scan cells/crops here, or the
        # YOLO result panel will show single-cell thumbnails as if they were
        # two-stage before/after comparison images.
        items: list[dict[str, str]] = []
        outputs_root = project_root / "outputs"
        candidate_dirs = [
            out_dir / "stage2" / "pairs",
            out_dir / "visualization" / "yolo_tiles",
            out_dir / "stage2" / "normal_only",
        ]
        for image_dir in candidate_dirs:
            if not image_dir.exists():
                continue
            for p in sorted(image_dir.glob("*.png")):
                try:
                    rel = str(p.relative_to(outputs_root)).replace("\\", "/")
                except ValueError:
                    continue
                items.append({"name": rel, "path": rel, "url": _url_for_output(rel), "has_detection": True})
            if items:
                break
        return {
            "schema_version": 1,
            "output": str(out_dir),
            "summary": {},
            "cells": [],
            "yolo_images": items,
            "exports": {},
        }

    def clean_output_dir(out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs_root = (project_root / "outputs").resolve()
        target = out_dir.resolve()
        if target != outputs_root and outputs_root not in target.parents:
            raise RuntimeError(f"为避免误删，clean_output 只允许清理 {outputs_root} 下的目录")
        for child in sorted(out_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def prepare_tiles(cfg: dict[str, Any], input_path: Path, out_dir: Path, progress_cb, cancel_cb) -> Path:
        tiles_dir = out_dir / "tiles"
        if cancel_cb():
            raise RuntimeError("cancelled")
        if input_path.is_dir():
            return input_path

        if input_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if cancel_cb():
                raise RuntimeError("cancelled")
            tiles_dir.mkdir(parents=True, exist_ok=True)
            dst = tiles_dir / input_path.name
            if input_path.resolve() != dst.resolve():
                shutil.copy2(str(input_path), str(dst))
            progress_cb({"stage": "tiling", "message": "输入为普通图片，按单 tile 处理", "progress": 100})
            return tiles_dir

        from code.main import tile_big_tiff  # type: ignore

        tile_size = int(cfg.get("tile_size", 1280))
        overlap = float(cfg.get("overlap", 0.0))
        workers = int(cfg.get("tile_workers", 0))
        tile_big_tiff(
            in_path=input_path,
            tiles_dir=tiles_dir,
            tile=tile_size,
            overlap=overlap,
            workers=workers,
            skip_black_tiles=True,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
        return tiles_dir

    def run_job(cfg: dict[str, Any]) -> None:
        state["running"] = True
        state["cancel"] = False
        t0 = time.time()

        try:
            input_raw = str(cfg.get("input_path") or "").strip()
            if not input_raw:
                raise RuntimeError("Please choose an input image or tile folder")
            input_path = _resolve_path(project_root, input_raw)
            if not input_path.exists():
                raise RuntimeError(f"Input path does not exist: {input_path}")

            out_dir = _resolve_path(project_root, cfg.get("output_dir") or (project_root / "outputs" / _timestamp()))
            if _as_bool(cfg.get("clean_output"), False):
                outputs_root = (project_root / "outputs").resolve()
                input_resolved = input_path.resolve()
                if input_resolved == outputs_root or outputs_root in input_resolved.parents:
                    raise RuntimeError("Input is inside outputs. Move the input image/folder outside outputs before cleaning history results.")
                out_resolved = out_dir.resolve()
                if out_resolved == outputs_root or outputs_root in out_resolved.parents:
                    clean_output_dir(outputs_root)
                else:
                    clean_output_dir(out_dir)
            elif out_dir.exists() and any(out_dir.iterdir()):
                out_dir = out_dir / _timestamp()
            out_dir.mkdir(parents=True, exist_ok=True)
            state["current_output"] = str(out_dir)

            def progress_cb(evt: dict[str, Any]) -> None:
                evt = dict(evt)
                evt.setdefault("status", "running")
                evt.setdefault("elapsed", time.time() - t0)
                progress = evt.get("progress")
                if progress is not None:
                    progress = float(progress)
                    if 0 < progress < 100:
                        evt["eta"] = ((time.time() - t0) / progress) * (100 - progress)
                    elif progress >= 100:
                        evt["eta"] = 0
                push(evt)

            def cancel_cb() -> bool:
                return bool(state.get("cancel"))

            push({"status": "running", "stage": "start", "message": "Task started", "progress": 0, "elapsed": 0.0})

            detection_mode = str(cfg.get("detection_mode") or "goal_only")
            weights_yolo = _resolve_path(project_root, cfg.get("weights_yolo") or default_yolo_weight)
            if detection_mode != "ufish_only" and not weights_yolo.exists():
                raise RuntimeError(f"YOLO weight does not exist: {weights_yolo}")
            weights_normal = _resolve_path(project_root, cfg.get("weights_normal") or default_normal_weight)
            yolo_conf = float(cfg.get("conf", 0.5))
            yolo_iou = float(cfg.get("iou", 0.45))
            normal_conf = float(cfg.get("normal_conf", yolo_conf))
            normal_iou = float(cfg.get("normal_iou", yolo_iou))
            device = str(cfg.get("device", "0"))
            half = _as_bool(cfg.get("half"), True)
            # 3060-friendly cap: avoid locking the desktop with oversized batches.
            requested_batch = max(1, int(cfg.get("batch_size", 24)))
            run_batch = min(requested_batch, 12 if detection_mode == "two_stage" else 24)

            if detection_mode == "ufish_only":
                progress_cb({"stage": "ufish", "message": "U-FISH only mode: no YOLO detection", "progress": 1})
                tiles_dir = input_path
            elif detection_mode == "goal_only" and not input_path.is_dir() and input_path.suffix.lower() in {".tif", ".tiff"}:
                progress_cb({"stage": "tiling", "message": "Single-stage goal memory streaming: no full tile prewrite", "progress": 1})
                tiles_dir = out_dir / "tiles"
            elif detection_mode == "two_stage" and not input_path.is_dir():
                progress_cb({"stage": "tiling", "message": "Two-stage memory streaming: no full tile prewrite", "progress": 1})
                tiles_dir = out_dir / "tiles"
            else:
                progress_cb({"stage": "tiling", "message": "Preparing image tiles", "progress": 1})
                tiles_dir = prepare_tiles(cfg, input_path, out_dir, progress_cb, cancel_cb)
            if cancel_cb():
                push({"status": "cancelled", "stage": "cancel", "message": "Task cancelled", "progress": 100})
                return

            # U-FISH counting is part of the fixed workflow; the UI no longer exposes a disable switch.
            enable_ufish = True
            ufish_model_path = _resolve_path(project_root, cfg.get("ufish_model_path") or default_ufish_onnx)
            ufish_onnx_path: Path | None = None
            ufish_pth_path: Path | None = None
            if ufish_model_path.suffix.lower() == ".onnx":
                ufish_onnx_path = ufish_model_path
                ufish_pth_path = default_ufish_pth if default_ufish_pth.exists() else None
            elif ufish_model_path.suffix.lower() == ".pth":
                ufish_pth_path = ufish_model_path
                ufish_onnx_path = default_ufish_onnx if default_ufish_onnx.exists() else None
            else:
                ufish_onnx_path = default_ufish_onnx if default_ufish_onnx.exists() else None
                ufish_pth_path = default_ufish_pth if default_ufish_pth.exists() else None

            common_config = {
                "detection_mode": detection_mode,
                "weights_yolo": str(weights_yolo),
                "weights_normal": str(weights_normal),
                "conf": yolo_conf,
                "iou": yolo_iou,
                "normal_conf": normal_conf,
                "normal_iou": normal_iou,
                "strict_nms_iou": float(cfg.get("strict_nms_iou", 0.25)),
                "strict_nms_overlap_min": float(cfg.get("strict_nms_overlap_min", 0.80)),
                "device": device,
                "batch": int(run_batch),
                "half": bool(half),
            }

            if detection_mode == "ufish_only":
                from code.ufish_only import run_ufish_only  # type: ignore

                manifest = run_ufish_only(
                    input_path=input_path,
                    out_root=out_dir,
                    ufish_onnx_path=ufish_onnx_path,
                    ufish_pth_path=ufish_pth_path,
                    ufish_red_threshold=float(cfg.get("ufish_red_threshold", 0.5)),
                    ufish_green_threshold=float(cfg.get("ufish_green_threshold", 0.5)),
                    ufish_min_area=int(cfg.get("ufish_min_area", 2)),
                    ufish_nms_distance=int(cfg.get("ufish_nms_distance", 5)),
                    ufish_nucleus_min_size=int(cfg.get("ufish_nucleus_min_size", 200)),
                    ufish_nucleus_dilation_radius=int(cfg.get("ufish_nucleus_dilation_radius", 3)),
                    ufish_batch_size=min(max(1, int(cfg.get("ufish_batch_size", 32))), 32),
                    device=device,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                )
            elif detection_mode == "two_stage":
                if not weights_normal.exists():
                    raise RuntimeError(f"normal weight does not exist: {weights_normal}")
                if input_path.is_dir():
                    from code.two_stage_normal_first import run_two_stage_normal_first_from_tiles  # type: ignore

                    pairs_dir = run_two_stage_normal_first_from_tiles(
                        tiles_dir=tiles_dir,
                        out_dir=out_dir,
                        weights_normal=weights_normal,
                        weights_goal=weights_yolo,
                        normal_conf=normal_conf,
                        normal_iou=normal_iou,
                        goal_conf=yolo_conf,
                        goal_iou=yolo_iou,
                        batch_size=run_batch,
                        device=device,
                        half=half,
                        pairs_gap=40,
                        strict_nms_iou=float(cfg.get("strict_nms_iou", 0.25)),
                        strict_nms_overlap_min=float(cfg.get("strict_nms_overlap_min", 0.80)),
                        progress_cb=progress_cb,
                        cancel_cb=cancel_cb,
                    )
                else:
                    from code.pipeline import run_two_stage_streaming_pairs  # type: ignore

                    pairs_dir = run_two_stage_streaming_pairs(
                        in_path=input_path,
                        out_dir=out_dir,
                        tile_size=int(cfg.get("tile_size", 1280)),
                        overlap=float(cfg.get("overlap", 0.0)),
                        tile_workers=int(cfg.get("tile_workers", 0)),
                        weights_normal=weights_normal,
                        weights_goal=weights_yolo,
                        stage1_conf=normal_conf,
                        stage1_iou=normal_iou,
                        stage2_conf=yolo_conf,
                        stage2_iou=yolo_iou,
                        batch_size=run_batch,
                        device=device,
                        half=half,
                        pairs_gap=40,
                        max_batch=run_batch,
                        strict_nms_iou=float(cfg.get("strict_nms_iou", 0.25)),
                        strict_nms_overlap_min=float(cfg.get("strict_nms_overlap_min", 0.80)),
                        progress_cb=progress_cb,
                        cancel_cb=cancel_cb,
                    )
                from code.two_stage_ufish_export import build_two_stage_ufish_manifest  # type: ignore

                manifest = build_two_stage_ufish_manifest(
                    out_root=out_dir,
                    pairs_dir=pairs_dir,
                    labels_goal_dir=out_dir / "labels" / "stage2_goal_base",
                    config=common_config,
                    # Two-stage U-FISH should always use the goal labels and
                    # the corresponding tile images. For big TIFF input the
                    # streaming pipeline also writes tiles to out_dir/tiles.
                    tiles_dir=tiles_dir,
                    source_image=input_path if not input_path.is_dir() else None,
                    tile_size=int(cfg.get("tile_size", 1280)),
                    enable_ufish=enable_ufish,
                    ufish_onnx_path=ufish_onnx_path,
                    ufish_pth_path=ufish_pth_path,
                    ufish_red_threshold=float(cfg.get("ufish_red_threshold", 0.5)),
                    ufish_green_threshold=float(cfg.get("ufish_green_threshold", 0.5)),
                    ufish_min_area=int(cfg.get("ufish_min_area", 2)),
                    ufish_nms_distance=int(cfg.get("ufish_nms_distance", 5)),
                    ufish_nucleus_min_size=int(cfg.get("ufish_nucleus_min_size", 200)),
                    ufish_nucleus_dilation_radius=int(cfg.get("ufish_nucleus_dilation_radius", 3)),
                    ufish_batch_size=min(max(1, int(cfg.get("ufish_batch_size", 32))), 32),
                    ufish_input_size=int(cfg.get("ufish_input_size", 256)),
                    crop_expand_ratio=float(cfg.get("crop_expand_ratio", 0.15)),
                    device=device,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                )
            elif detection_mode == "normal_only":
                if not weights_normal.exists():
                    raise RuntimeError(f"normal weight does not exist: {weights_normal}")
                from code.normal_only_fast import run_normal_only_fast  # type: ignore

                pairs_dir = run_normal_only_fast(
                    tiles_dir=tiles_dir,
                    out_root=out_dir,
                    weights_normal=weights_normal,
                    conf=normal_conf,
                    iou=normal_iou,
                    device=device,
                    batch=run_batch,
                    half=half,
                    pairs_subdir="normal_only",
                    gap=40,
                    png_compression=1,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                )
                manifest = _image_manifest_from_dir(
                    out_root=out_dir,
                    image_dir=pairs_dir,
                    mode="normal_only",
                    config=common_config,
                    summary={"total_cells": 0, "ctc_total": 0, "cec_total": 0, "red_probe_total": 0, "green_probe_total": 0, "normal_images": len(list(pairs_dir.glob("*.png")))},
                )
            else:
                if not input_path.is_dir() and input_path.suffix.lower() in {".tif", ".tiff"}:
                    from code.unified_detection import run_unified_yolo_ufish_streaming_tiff  # type: ignore

                    manifest = run_unified_yolo_ufish_streaming_tiff(
                        in_path=input_path,
                        out_root=out_dir,
                        weights_yolo=weights_yolo,
                        tile_size=int(cfg.get("tile_size", 1280)),
                        overlap=float(cfg.get("overlap", 0.0)),
                        conf=yolo_conf,
                        iou=yolo_iou,
                        device=device,
                        batch=run_batch,
                        half=half,
                        enable_ufish=enable_ufish,
                        ufish_onnx_path=ufish_onnx_path,
                        ufish_pth_path=ufish_pth_path,
                        ufish_red_threshold=float(cfg.get("ufish_red_threshold", 0.5)),
                        ufish_green_threshold=float(cfg.get("ufish_green_threshold", 0.5)),
                        ufish_min_area=int(cfg.get("ufish_min_area", 2)),
                        ufish_nms_distance=int(cfg.get("ufish_nms_distance", 5)),
                        ufish_nucleus_min_size=int(cfg.get("ufish_nucleus_min_size", 200)),
                        ufish_nucleus_dilation_radius=int(cfg.get("ufish_nucleus_dilation_radius", 3)),
                        ufish_batch_size=min(max(1, int(cfg.get("ufish_batch_size", 32))), 32),
                        crop_expand_ratio=float(cfg.get("crop_expand_ratio", 0.15)),
                        strict_nms_iou=float(cfg.get("strict_nms_iou", 0.25)),
                        strict_nms_overlap_min=float(cfg.get("strict_nms_overlap_min", 0.80)),
                        min_cell_box_side=float(cfg.get("min_cell_box_side", 20.0)),
                        max_cell_box_aspect=float(cfg.get("max_cell_box_aspect", 5.0)),
                        progress_cb=progress_cb,
                        cancel_cb=cancel_cb,
                    )
                else:
                    from code.unified_detection import run_unified_yolo_ufish  # type: ignore

                    manifest = run_unified_yolo_ufish(
                        tiles_dir=tiles_dir,
                        out_root=out_dir,
                        weights_yolo=weights_yolo,
                        conf=yolo_conf,
                        iou=yolo_iou,
                        device=device,
                        batch=run_batch,
                        half=half,
                        enable_ufish=enable_ufish,
                        ufish_onnx_path=ufish_onnx_path,
                        ufish_pth_path=ufish_pth_path,
                        ufish_red_threshold=float(cfg.get("ufish_red_threshold", 0.5)),
                        ufish_green_threshold=float(cfg.get("ufish_green_threshold", 0.5)),
                        ufish_min_area=int(cfg.get("ufish_min_area", 2)),
                        ufish_nms_distance=int(cfg.get("ufish_nms_distance", 5)),
                        ufish_nucleus_min_size=int(cfg.get("ufish_nucleus_min_size", 200)),
                        ufish_nucleus_dilation_radius=int(cfg.get("ufish_nucleus_dilation_radius", 3)),
                        ufish_batch_size=min(max(1, int(cfg.get("ufish_batch_size", 32))), 32),
                        ufish_input_size=int(cfg.get("ufish_input_size", 256)),
                        crop_expand_ratio=float(cfg.get("crop_expand_ratio", 0.15)),
                        strict_nms_iou=float(cfg.get("strict_nms_iou", 0.25)),
                        strict_nms_overlap_min=float(cfg.get("strict_nms_overlap_min", 0.80)),
                        min_cell_box_side=float(cfg.get("min_cell_box_side", 20.0)),
                        max_cell_box_aspect=float(cfg.get("max_cell_box_aspect", 5.0)),
                        progress_cb=progress_cb,
                        cancel_cb=cancel_cb,
                    )

            if cancel_cb():
                push({"status": "cancelled", "stage": "cancel", "message": "Task cancelled", "progress": 100})
                return

            push({
                "status": "done",
                "stage": "done",
                "message": "Detection complete",
                "progress": 100,
                "elapsed": time.time() - t0,
                "results": _hydrate_manifest(manifest),
            })
        except Exception as exc:  # noqa: BLE001
            if str(exc) == "cancelled" or bool(state.get("cancel")):
                push({"status": "cancelled", "stage": "cancel", "message": "Task cancelled", "progress": 100, "elapsed": time.time() - t0})
            else:
                push({"status": "error", "stage": "error", "message": str(exc), "progress": 100, "elapsed": time.time() - t0})
        finally:
            try:
                import gc

                gc.collect()
            except Exception:
                pass
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            state["running"] = False
            state["cancel"] = False
    _tk_lock = threading.Lock()
    container_mode = os.environ.get("APP_CONTAINER_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

    def pick_file_dialog(title: str, filetypes: list[tuple[str, str]] | None = None) -> str | None:
        if container_mode:
            return None
        with _tk_lock:
            try:
                import tkinter as tk
                from tkinter import filedialog
            except Exception:
                return None
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(title=title, filetypes=filetypes or [("All files", "*.*")])
            root.destroy()
            return path or None

    def pick_dir_dialog(title: str) -> str | None:
        if container_mode:
            return None
        with _tk_lock:
            try:
                import tkinter as tk
                from tkinter import filedialog
            except Exception:
                return None
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return path or None

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            project_root=str(project_root),
            default_yolo_weight=str(default_yolo_weight),
            default_normal_weight=str(default_normal_weight),
            default_ufish_model=str(default_ufish_onnx if default_ufish_onnx.exists() else default_ufish_pth),
            default_ufish_onnx=str(default_ufish_onnx),
            default_ufish_pth=str(default_ufish_pth),
            default_output=str(project_root / "outputs" / _timestamp()),
            container_mode=container_mode,
        )

    @app.get("/api/pick_file")
    def api_pick_file():
        kind = request.args.get("kind", "")
        if kind == "weights_yolo":
            p = pick_file_dialog("Choose YOLO weight", [("YOLO weight", "*.pt;*.onnx"), ("All files", "*.*")])
        elif kind == "weights_normal":
            p = pick_file_dialog("Choose normal weight", [("YOLO weight", "*.pt;*.onnx"), ("All files", "*.*")])
        elif kind == "ufish_model":
            p = pick_file_dialog("Choose U-FISH model", [("U-FISH model", "*.onnx;*.pth"), ("All files", "*.*")])
        else:
            p = pick_file_dialog("Choose input image", [("Image", "*.tif;*.tiff;*.png;*.jpg;*.jpeg"), ("All files", "*.*")])
        if not p:
            msg = "Docker 版不能弹出 Windows 文件选择框，请把文件放到启动包 data 文件夹，然后填写 /data/文件名.tif"
            return jsonify({"ok": False, "error": msg if container_mode else ""})
        return jsonify({"ok": True, "path": p})

    @app.get("/api/pick_dir")
    def api_pick_dir():
        p = pick_dir_dialog("Choose folder")
        if not p:
            msg = "Docker 版不能弹出 Windows 文件夹选择框，请把文件夹放到启动包 data 目录，然后填写 /data/文件夹名"
            return jsonify({"ok": False, "error": msg if container_mode else ""})
        return jsonify({"ok": True, "path": p})

    @app.post("/api/upload")
    def api_upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"ok": False, "error": "没有收到上传文件"}), 400
        suffix = Path(file.filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            return jsonify({"ok": False, "error": f"不支持的图像格式: {suffix}"}), 400
        upload_dir = project_root / "outputs" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dst = upload_dir / f"{_timestamp()}_{_safe_name(file.filename)}"
        file.save(str(dst))
        return jsonify({"ok": True, "path": str(dst)})

    @app.post("/api/upload_folder")
    def api_upload_folder():
        files = request.files.getlist("files")
        if not files:
            return jsonify({"ok": False, "error": "没有收到上传文件夹"}), 400
        upload_root = project_root / "outputs" / "uploads" / f"{_timestamp()}_folder"
        saved = 0
        for file in files:
            if file is None or not file.filename:
                continue
            suffix = Path(file.filename).suffix.lower()
            if suffix and suffix not in IMAGE_SUFFIXES:
                continue
            parts = _safe_rel_parts(file.filename)
            if not parts:
                continue
            if len(parts) == 1:
                dst = upload_root / parts[0]
            else:
                dst = upload_root.joinpath(*parts[1:])
            dst.parent.mkdir(parents=True, exist_ok=True)
            file.save(str(dst))
            saved += 1
        if saved <= 0:
            return jsonify({"ok": False, "error": "文件夹里没有支持的图像文件"}), 400
        return jsonify({"ok": True, "path": str(upload_root), "count": saved})

    @app.get("/api/list_weights")
    def api_list_weights():
        items: list[dict[str, str]] = []
        default_dir = data_root / "wights" / "goal"
        optional_dir = data_root / "wights" / "yolo"

        def add_weight(path: Path, prefix: str = "") -> None:
            if path.exists() and all(str(path) != it["path"] for it in items):
                name = f"{prefix}{path.name}" if prefix else path.name
                items.append({"name": name, "path": str(path)})

        add_weight(default_yolo_weight, "默认: ")
        for p in sorted(default_dir.glob("*.pt")):
            add_weight(p)
        for p in sorted(optional_dir.glob("*.pt")):
            add_weight(p)
        if default_yolo_weight.exists() and all(str(default_yolo_weight) != it["path"] for it in items):
            items.insert(0, {"name": default_yolo_weight.name, "path": str(default_yolo_weight)})
        return jsonify({
            "ok": True,
            "default": str(default_yolo_weight),
            "default_dir": str(default_dir),
            "optional_dir": str(optional_dir),
            "items": items,
        })

    @app.get("/api/list_ufish_models")
    def api_list_ufish_models():
        items = []
        for p in [default_ufish_onnx, default_ufish_pth]:
            if p.exists():
                items.append({"name": p.name, "path": str(p)})
        return jsonify({"ok": True, "items": items})

    @app.get("/api/results")
    def api_results():
        out = state.get("current_output")
        if out:
            out_dir = Path(str(out))
        else:
            latest = find_latest_output(project_root / "outputs")
            if latest is None:
                return jsonify({"output": None, "summary": {}, "cells": [], "yolo_images": [], "exports": {}})
            out_dir = latest
        return jsonify(load_results(out_dir))

    @app.get("/api/results_cells")
    def api_results_cells():
        out = state.get("current_output")
        if out:
            out_dir = Path(str(out))
        else:
            latest = find_latest_output(project_root / "outputs")
            if latest is None:
                return jsonify({"ok": False, "cells": [], "offset": 0, "limit": 0, "total": 0})
            out_dir = latest

        manifest_path = out_dir / "manifest.json"
        if not manifest_path.exists():
            return jsonify({"ok": False, "cells": [], "offset": 0, "limit": 0, "total": 0})
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc), "cells": [], "offset": 0, "limit": 0, "total": 0}), 500

        cells = manifest.get("cells", []) or []
        cls_filter = str(request.args.get("class", "all")).upper()
        if cls_filter in {"CTC", "CEC", "CELL"}:
            cells = [cell for cell in cells if str(cell.get("class", "")).upper() == cls_filter]
        try:
            offset = max(0, int(request.args.get("offset", 0)))
            limit = max(1, min(1000, int(request.args.get("limit", 500))))
        except ValueError:
            offset, limit = 0, 500
        chunk = cells[offset : offset + limit]
        return jsonify({
            "ok": True,
            "cells": _hydrate_cells(chunk),
            "offset": offset,
            "limit": limit,
            "total": len(cells),
            "next_offset": min(len(cells), offset + len(chunk)),
            "has_more": offset + len(chunk) < len(cells),
        })

    @app.post("/api/start")
    def api_start():
        if state["running"]:
            status = "正在终止，请等待当前 YOLO 批次结束" if state.get("cancel") else "任务正在运行"
            return jsonify({"ok": False, "error": status}), 409
        state["cancel"] = False
        cfg = request.json or {}
        th = threading.Thread(target=run_job, args=(cfg,), daemon=True)
        th.start()
        return jsonify({"ok": True})

    @app.post("/api/stop")
    def api_stop():
        if not state["running"]:
            state["cancel"] = False
            push({"status": "idle", "stage": "idle", "message": "当前没有正在运行的任务"})
            return jsonify({"ok": True, "running": False})
        state["cancel"] = True
        push({"status": "stopping", "stage": "cancel", "message": "已请求终止，等待当前 YOLO 批次结束"})
        return jsonify({"ok": True, "running": True})

    @app.get("/api/status")
    def api_status():
        return jsonify({"running": state["running"], "last": state["last"]})

    @app.get("/api/stream")
    def api_stream():
        def gen():
            last_sent = None
            while True:
                payload = json.dumps(state.get("last"), ensure_ascii=False)
                if payload != last_sent:
                    yield f"data: {payload}\n\n"
                    last_sent = payload
                time.sleep(0.2)

        return Response(gen(), mimetype="text/event-stream")

    @app.get("/outputs/<path:subpath>")
    def outputs(subpath: str):
        return send_from_directory(str(project_root / "outputs"), subpath)

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else project_root
    app = create_app(project_root, data_root=data_root)

    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

