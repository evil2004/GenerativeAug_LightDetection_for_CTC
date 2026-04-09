from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_from_directory


def create_app(project_root: Path) -> Flask:
    # Ensure project_root/code is importable as a package
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    app = Flask(
        __name__,
        template_folder=str((Path(__file__).parent / "templates").resolve()),
        static_folder=str((Path(__file__).parent / "static").resolve()),
    )

    state: dict[str, Any] = {
        "running": False,
        "cancel": False,
        "last": {"status": "idle"},
        "logs": [],
        "current_output": None,
    }

    default_weights_normal_path = project_root / "wights" / "normal" / "best_nomal.pt"
    default_weights_goal_two_stage = project_root / "wights" / "goal" / "best_goal.pt"
    default_weights_goal_only = project_root / "wights" / "goal" / "onlygoal.pt"

    def push(event: dict[str, Any]) -> None:
        state["last"] = event
        state["logs"].append(event)
        if len(state["logs"]) > 4000:
            state["logs"] = state["logs"][-4000:]
        try:
            stage = str(event.get("stage", ""))
            if stage in {
                "config",
                "stage1",
                "stage2",
                "stage2_goal_only",
                "stage2_pairs",
                "summary",
                "error",
                "params",
                "predict",
                "cleanup",
                "done",
                "cancel",
            }:
                import json as _json

                print("[EVENT]", _json.dumps(event, ensure_ascii=False))
        except Exception:
            pass

    def find_latest_output(outputs_root: Path) -> Path | None:
        if not outputs_root.exists():
            return None
        dirs = [p for p in outputs_root.iterdir() if p.is_dir()]
        if not dirs:
            return None
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0]

    def list_result_images(out_dir: Path) -> dict[str, Any]:
        candidates = [
            out_dir / "stage2" / "pairs",
            out_dir / "stage2" / "normal_only_pairs",
            out_dir / "stage2" / "goal_only",
        ]

        def _has_images(d: Path) -> bool:
            if not d.exists():
                return False
            for _p in d.rglob("*.png"):
                return True
            for _p in d.rglob("*.jpg"):
                return True
            for _p in d.rglob("*.jpeg"):
                return True
            return False

        picked = None
        for d in candidates:
            if _has_images(d):
                picked = d
                break

        items: list[dict[str, str]] = []
        if picked is not None:
            for p in sorted(list(picked.rglob("*.png")) + list(picked.rglob("*.jpg")) + list(picked.rglob("*.jpeg"))):
                try:
                    rel = p.relative_to(project_root / "outputs")
                except ValueError:
                    try:
                        rel = p.relative_to(project_root)
                    except ValueError:
                        rel = p.name
                rel_url = str(rel).replace("\\", "/")
                items.append({"name": str(rel), "url": f"/outputs/{rel_url}"})

        return {"output": str(out_dir), "items": items}

    def run_job(cfg: dict[str, Any]) -> None:
        state["running"] = True
        state["cancel"] = False
        t0 = time.time()

        out_dir = Path(str(cfg.get("output_dir") or "")).expanduser()
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()

        clean_output = bool(cfg.get("clean_output", False))

        try:
            if clean_output:
                out_dir.mkdir(parents=True, exist_ok=True)

                parent = out_dir.parent
                ts_re = __import__("re").compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{6}$")
                ts_re2 = __import__("re").compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}-?[0-9]{6}$")

                send2trash_fn = None
                try:
                    from send2trash import send2trash as _send2trash  # type: ignore

                    send2trash_fn = _send2trash
                except Exception:
                    send2trash_fn = None

                removed = 0
                skipped = 0
                considered = 0
                matched = 0
                sample: list[str] = []

                for d in sorted(parent.iterdir()) if parent.exists() else []:
                    considered += 1
                    if len(sample) < 25:
                        sample.append(f"{d.name} (dir={d.is_dir()})")

                    try:
                        if d.resolve() == out_dir.resolve():
                            skipped += 1
                            continue
                        if not d.is_dir():
                            continue

                        name = d.name
                        if not (ts_re.match(name) or ts_re2.match(name)):
                            continue

                        matched += 1

                        if send2trash_fn is not None:
                            try:
                                send2trash_fn(str(d))
                                removed += 1
                                continue
                            except Exception as _e:
                                print(f"[WARN] 清理失败(回收站，转硬删): {d} err={_e}")

                        import shutil

                        shutil.rmtree(d)
                        removed += 1
                    except Exception as _e:
                        print(f"[WARN] 清理失败(硬删也失败): {d} err={_e}")

                push({
                    "status": "running",
                    "stage": "cleanup",
                    "message": (
                        f"clean_output=True keep={out_dir} parent={parent} "
                        f"considered={considered} matched={matched} removed={removed} skipped={skipped} sample={sample}"
                    ),
                })
            else:
                if out_dir.exists() and any(out_dir.iterdir()):
                    out_dir = out_dir / time.strftime("%Y-%m-%d-%H%M%S")
        except Exception as _e:
            print(f"[WARN] 输出目录处理失败: out_dir={out_dir} err={_e}")

        state["current_output"] = str(out_dir)

        def progress_cb(evt: dict) -> None:
            evt = dict(evt)
            evt.setdefault("status", "running")
            elapsed = time.time() - t0
            evt.setdefault("elapsed", elapsed)

            if "progress" in evt:
                progress = float(evt["progress"])
                if 0 < progress < 100:
                    eta = (elapsed / progress) * (100 - progress) if progress > 0 else 0
                    evt["eta"] = eta
                elif progress >= 100:
                    evt["eta"] = 0

            push(evt)

        def cancel_cb() -> bool:
            return bool(state.get("cancel"))

        try:
            push({"status": "running", "stage": "start", "message": "任务开始", "progress": 0, "elapsed": 0.0})

            mode = str(cfg.get("mode") or "two_stage")

            raw_conf = cfg.get("conf")
            raw_iou = cfg.get("iou")
            raw_s1c = cfg.get("stage1_conf")
            raw_s1i = cfg.get("stage1_iou")
            raw_s2c = cfg.get("stage2_conf")
            raw_s2i = cfg.get("stage2_iou")

            conf = float(cfg.get("conf", 0.25))
            iou = float(cfg.get("iou", 0.45))
            stage1_conf = float(cfg.get("stage1_conf", 0.25))
            stage1_iou = float(cfg.get("stage1_iou", 0.45))
            stage2_conf = float(cfg.get("stage2_conf", 0.01))
            stage2_iou = float(cfg.get("stage2_iou", 0.45))

            device = str(cfg.get("device", "cpu"))
            batch_size = int(cfg.get("batch_size", 64))
            half = bool(cfg.get("half", False))

            if mode == "goal_only":
                eff_goal_conf, eff_goal_iou = conf, iou
                eff_normal_conf, eff_normal_iou = None, None
                eff_goal_src = "conf/iou"
            elif mode == "normal_only":
                eff_goal_conf, eff_goal_iou = None, None
                eff_normal_conf, eff_normal_iou = stage1_conf, stage1_iou
                eff_goal_src = None
            else:
                eff_goal_conf, eff_goal_iou = stage2_conf, stage2_iou
                eff_normal_conf, eff_normal_iou = stage1_conf, stage1_iou
                eff_goal_src = "stage2_conf/stage2_iou"

            push({
                "status": "running",
                "stage": "params",
                "message": (
                    f"raw(conf={raw_conf},iou={raw_iou},s1=({raw_s1c},{raw_s1i}),s2=({raw_s2c},{raw_s2i})) -> "
                    f"parsed(conf={conf},iou={iou},s1=({stage1_conf},{stage1_iou}),s2=({stage2_conf},{stage2_iou})) | "
                    f"effective goal={eff_goal_conf},{eff_goal_iou} src={eff_goal_src} normal={eff_normal_conf},{eff_normal_iou} | "
                    f"device={device} batch={batch_size} half={half}"
                ),
            })

            weights_normal = str(cfg.get("weights_normal") or str(default_weights_normal_path))
            default_goal_weight = default_weights_goal_only if mode == "goal_only" else default_weights_goal_two_stage
            weights_goal = str(cfg.get("weights_goal") or str(default_goal_weight))

            input_path = str(cfg.get("input_path") or "")
            if not input_path:
                raise RuntimeError("输入路径为空")

            in_path = Path(input_path)
            if not in_path.is_absolute():
                in_path = (project_root / in_path).resolve()

            if not in_path.exists():
                raise RuntimeError(f"输入不存在: {in_path}")

            from code.main import tile_big_tiff  # type: ignore

            out_dir.mkdir(parents=True, exist_ok=True)
            tiles_dir = out_dir / "tiles"

            if in_path.is_dir():
                tiles_dir = in_path
                push({"status": "running", "stage": "tiling", "message": "输入为文件夹，跳过切片", "progress": 5, "elapsed": time.time() - t0})
            else:
                push({"status": "running", "stage": "tiling", "message": "开始切片...", "progress": 1, "elapsed": time.time() - t0})

                def tile_progress_cb(evt: dict) -> None:
                    evt = dict(evt)
                    evt.setdefault("status", "running")
                    evt.setdefault("stage", "tiling")
                    progress_cb(evt)

                tile_big_tiff(in_path=in_path, tiles_dir=tiles_dir, progress_cb=tile_progress_cb)
                push({"status": "running", "stage": "tiling", "message": "切片完成", "progress": 20, "elapsed": time.time() - t0})

            if mode == "goal_only":
                push({
                    "status": "running",
                    "stage": "predict",
                    "message": f"goal_only predict: conf={eff_goal_conf} iou={eff_goal_iou} device={device} batch={batch_size} half={half}",
                })

                from code.goal_only_fast import run_goal_only_fast  # type: ignore

                run_goal_only_fast(
                    tiles_dir=tiles_dir,
                    out_root=out_dir,
                    weights_goal=Path(weights_goal),
                    conf=float(eff_goal_conf),
                    iou=float(eff_goal_iou),
                    device=device,
                    batch=batch_size,
                    half=half,
                    pred_subdir="goal_only",
                    labels_subdir="goal_only",
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                )
            elif mode == "normal_only":
                push({
                    "status": "running",
                    "stage": "predict",
                    "message": f"normal_only predict: conf={eff_normal_conf} iou={eff_normal_iou} device={device} batch={batch_size} half={half}",
                })

                from code.normal_only_fast import run_normal_only_fast  # type: ignore

                run_normal_only_fast(
                    tiles_dir=tiles_dir,
                    out_root=out_dir,
                    weights_normal=Path(weights_normal),
                    conf=float(eff_normal_conf),
                    iou=float(eff_normal_iou),
                    device=device,
                    batch=batch_size,
                    half=half,
                    pairs_subdir="normal_only_pairs",
                    gap=int(cfg.get("pairs_gap", 60)),
                    png_compression=0,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                )
            else:
                push({
                    "status": "running",
                    "stage": "predict",
                    "message": (
                        f"two_stage predict: goal(conf={eff_goal_conf},iou={eff_goal_iou}) normal(conf={eff_normal_conf},iou={eff_normal_iou}) "
                        f"device={device} batch={batch_size} half={half}"
                    ),
                })

                from code.two_stage_goal_first import run_two_stage_goal_first_pairs_from_tiles  # type: ignore

                run_two_stage_goal_first_pairs_from_tiles(
                    tiles_dir=tiles_dir,
                    out_dir=out_dir,
                    weights_goal=Path(weights_goal),
                    weights_normal=Path(weights_normal),
                    batch_size=batch_size,
                    device=device,
                    half=half,
                    pairs_gap=int(cfg.get("pairs_gap", 60)),
                    goal_conf=float(eff_goal_conf),
                    goal_iou=float(eff_goal_iou),
                    normal_conf=float(eff_normal_conf),
                    normal_iou=float(eff_normal_iou),
                    normal_keep_overlap_ratio=0.10,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                )

            if cancel_cb():
                push({"status": "cancelled", "stage": "cancel", "message": "已终止", "progress": 100, "elapsed": time.time() - t0})
                return

            push({"status": "running", "stage": "index", "message": "索引结果...", "progress": 95, "elapsed": time.time() - t0})
            res = list_result_images(out_dir)
            push({"status": "done", "stage": "done", "message": "完成", "progress": 100, "elapsed": time.time() - t0, "results": res})

        except Exception as e:  # noqa: BLE001
            push({"status": "error", "stage": "error", "message": str(e), "progress": 100, "elapsed": time.time() - t0})
        finally:
            try:
                import gc

                gc.collect()
            except Exception:
                pass
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            state["running"] = False

    _tk_lock = threading.Lock()

    def pick_file_dialog(title: str, filetypes: list[tuple[str, str]] | None = None) -> str | None:
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
            default_weights_normal=str(default_weights_normal_path),
            default_weights_goal=str(default_weights_goal_two_stage),
            default_weights_goal_two_stage=str(default_weights_goal_two_stage),
            default_weights_goal_only=str(default_weights_goal_only),
            default_weights_normal_dir=str(default_weights_normal_path.parent),
            default_weights_goal_dir=str(default_weights_goal_two_stage.parent),
            default_output=str(project_root / "outputs" / time.strftime("%Y-%m-%d-%H%M%S")),
        )

    @app.get("/api/pick_file")
    def api_pick_file():
        kind = request.args.get("kind", "")
        if kind in ("weights_normal", "weights_goal"):
            p = pick_file_dialog("选择模型权重文件", [("PyTorch 权重", "*.pt"), ("所有文件", "*.*")])
        else:
            p = pick_file_dialog("选择输入图像", [("TIFF", "*.tif;*.tiff"), ("所有文件", "*.*")])
        if not p:
            return jsonify({"ok": False})
        return jsonify({"ok": True, "path": p})

    @app.get("/api/pick_dir")
    def api_pick_dir():
        p = pick_dir_dialog("选择文件夹")
        if not p:
            return jsonify({"ok": False})
        return jsonify({"ok": True, "path": p})

    @app.get("/api/list_weights")
    def api_list_weights():
        kind = request.args.get("kind", "weights_goal")
        dir_map = {
            "weights_goal": default_weights_goal_two_stage.parent,
            "weights_normal": default_weights_normal_path.parent,
        }
        dir_path = dir_map.get(kind, default_weights_goal_two_stage.parent)
        items: list[dict[str, str]] = []
        try:
            if dir_path.exists():
                for p in sorted(dir_path.iterdir()):
                    if p.is_file() and p.suffix.lower() == ".pt":
                        items.append({"name": p.name, "path": str(p)})
        except Exception:
            items = []
        return jsonify({"ok": True, "dir": str(dir_path), "items": items})

    @app.get("/api/results")
    def results():
        out = state.get("current_output")
        if out:
            out_dir = Path(str(out))
        else:
            latest = find_latest_output(project_root / "outputs")
            if not latest:
                return jsonify({"output": None, "items": []})
            out_dir = latest
        return jsonify(list_result_images(out_dir))

    @app.post("/api/start")
    def start():
        if state["running"]:
            return jsonify({"ok": False, "error": "任务正在运行"}), 409
        cfg = request.json or {}
        th = threading.Thread(target=run_job, args=(cfg,), daemon=True)
        th.start()
        return jsonify({"ok": True})

    @app.post("/api/stop")
    def stop():
        state["cancel"] = True
        push({"status": "cancel_requested", "stage": "cancel", "message": "已请求终止"})
        return jsonify({"ok": True})

    @app.get("/api/status")
    def status():
        return jsonify({"running": state["running"], "last": state["last"]})

    @app.get("/api/stream")
    def stream():
        def gen():
            last_sent = None
            while True:
                cur = state.get("last")
                payload = json.dumps(cur, ensure_ascii=False)
                if payload != last_sent:
                    yield f"data: {payload}\n\n"
                    last_sent = payload
                time.sleep(0.2)

        return Response(gen(), mimetype="text/event-stream")

    @app.get("/outputs/<path:subpath>")
    def outputs(subpath: str):
        out_dir = project_root / "outputs"
        return send_from_directory(str(out_dir), subpath)

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    app = create_app(project_root)

    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
