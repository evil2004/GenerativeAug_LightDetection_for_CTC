from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import time
import warnings
from pathlib import Path


def ensure_project_folders(project_root: Path) -> None:
    (project_root / "wights" / "normal").mkdir(parents=True, exist_ok=True)
    (project_root / "wights" / "goal").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs").mkdir(parents=True, exist_ok=True)
    (project_root / "code").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTC/CEC 检测工具")
    p.add_argument("--input", type=str, default="", help="输入tif/tiff大图路径")
    p.add_argument(
        "--weights-normal",
        type=str,
        default=str(Path("wights/normal/best_nomal.pt")),
        help="normal权重",
    )
    p.add_argument(
        "--weights-goal",
        type=str,
        default=str(Path("wights/goal/best_goal.pt")),
        help="goal权重",
    )
    p.add_argument("--output", type=str, default="", help="输出目录（默认 outputs/<时间戳>）")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值(仅 normal_only/two_stage 生效)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IOU阈值(仅 normal_only/two_stage 生效)")

    p.add_argument(
        "--mode",
        type=str,
        default="two_stage",
        choices=["two_stage", "goal_only", "normal_only"],
        help="运行模式：two_stage=normal->goal；goal_only=直接goal模型；normal_only=直接normal模型输出左右pair",
    )

    p.add_argument("--tile-size", type=int, default=1280, help="切图尺寸（默认1280）")
    p.add_argument("--overlap", type=float, default=0.10, help="重叠比例（默认10%%）")
    p.add_argument(
        "--tile-workers",
        type=int,
        default=0,
        help="切图并发进程数（0表示自动）",
    )
    p.add_argument(
        "--only-tile",
        action="store_true",
        help="只执行切图（用于验证切图速度/结果）",
    )

    p.add_argument(
        "--clean-output",
        action="store_true",
        help="开始预测前清理输出目录全部内容（谨慎：会删除该目录下所有文件）",
    )

    return p.parse_args()


def timestamp_name() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")


def _prompt_str(title: str, default: str = "") -> str:
    tip = f"[{default}]" if default else ""
    v = input(f"{title}{tip}: ").strip()
    return v if v else default


def _prompt_float(title: str, default: float) -> float:
    while True:
        v = input(f"{title}[{default}]: ").strip()
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            print("请输入数字。")


def interactive_fill_args(args: argparse.Namespace, project_root: Path) -> argparse.Namespace:
    print("\n===== 交互式配置 =====")
    print("提示：直接回车使用默认值；路径可拖拽文件到终端自动填入。")

    default_input = args.input
    if not default_input:
        default_input = str(project_root / "Result-2025-11-07-160724_Fused.tif")

    args.input = _prompt_str("请输入输入tif/tiff大图路径", default_input)

    args.mode = _prompt_str("请选择模式(two_stage/goal_only/normal_only)", args.mode)

    args.weights_normal = _prompt_str(
        "请输入 normal 权重路径",
        args.weights_normal or str(project_root / "wights" / "normal" / "best_nomal.pt"),
    )
    # goal 权重默认值随模式变化：two_stage 用 best_goal.pt；goal_only 用 onlygoal.pt
    goal_default = (
        args.weights_goal
        or str(
            project_root
            / "wights"
            / "goal"
            / ("onlygoal.pt" if args.mode == "goal_only" else "best_goal.pt")
        )
    )
    args.weights_goal = _prompt_str("请输入 goal 权重路径", goal_default)
    args.conf = _prompt_float("请输入 conf", float(args.conf))
    args.iou = _prompt_float("请输入 iou", float(args.iou))

    default_output = args.output or str(project_root / "outputs" / timestamp_name())
    args.output = _prompt_str("请输入输出目录（将自动创建）", default_output)

    print("===== 配置完成 =====\n")
    return args


def get_tiff_info(path: Path) -> dict:
    import tifffile as tiff  # type: ignore

    with tiff.TiffFile(str(path)) as tf:
        page0 = tf.pages[0]
        series0 = tf.series[0]

        compression = None
        try:
            compression = page0.tags["Compression"].value
        except Exception:  # noqa: BLE001
            compression = None

        photometric = None
        try:
            photometric = page0.photometric
        except Exception:  # noqa: BLE001
            photometric = None

        dtype = None
        try:
            dtype = str(series0.dtype)
        except Exception:  # noqa: BLE001
            dtype = None

        shape = None
        try:
            shape = tuple(int(x) for x in series0.shape)
        except Exception:  # noqa: BLE001
            shape = None

        return {
            "pages": len(tf.pages),
            "series": len(tf.series),
            "shape": shape,
            "dtype": dtype,
            "compression": compression,
            "photometric": str(photometric) if photometric is not None else None,
            "is_bigtiff": bool(getattr(tf, "is_bigtiff", False)),
        }


def compute_tiles(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    if tile <= 0:
        raise ValueError("tile-size 必须 > 0")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap 必须在 [0, 1) 范围")

    step = max(1, int(round(tile * (1.0 - overlap))))

    xs: list[int] = []
    ys: list[int] = []

    x = 0
    while True:
        if x + tile >= width:
            x = max(0, width - tile)
            xs.append(x)
            break
        xs.append(x)
        x += step

    y = 0
    while True:
        if y + tile >= height:
            y = max(0, height - tile)
            ys.append(y)
            break
        ys.append(y)
        y += step

    tiles: list[tuple[int, int, int, int]] = []
    for yy in ys:
        for xx in xs:
            tiles.append((xx, yy, tile, tile))
    return tiles


def _format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m{int(s % 60):02d}s"


def tile_big_tiff(
    in_path: Path,
    tiles_dir: Path,
    tile: int = 1280,
    overlap: float = 0.10,
    workers: int = 0,
    progress_cb: callable | None = None,
    on_tile: callable | None = None,
) -> tuple[int, int, int, float]:
    import multiprocessing as mp

    tiles_dir.mkdir(parents=True, exist_ok=True)

    info = get_tiff_info(in_path)
    shape = info.get("shape")
    if not shape or len(shape) < 2:
        raise RuntimeError(f"不支持的tiff shape: {shape}")

    height = int(shape[0])
    width = int(shape[1])

    tiles = compute_tiles(width=width, height=height, tile=tile, overlap=overlap)

    if workers <= 0:
        workers = min(16, (os.cpu_count() or 8))

    jobs: list[tuple[str, int, int, int, int, str]] = []
    in_path_s = str(in_path)
    for (x, y, w, h) in tiles:
        out_name = f"x{x}_y{y}.png"
        out_path = str(tiles_dir / out_name)
        jobs.append((in_path_s, x, y, w, h, out_path))

    from code.pipeline import tile_worker  # type: ignore

    t0 = time.perf_counter()
    total = len(jobs)
    done = 0
    last_print_elapsed = 0.0
    last_cb_elapsed = 0.0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=30) as pool:
        try:
            for out_path_str in pool.imap_unordered(tile_worker, jobs, chunksize=2):
                done += 1
                if on_tile is not None:
                    try:
                        on_tile(Path(out_path_str))
                    except Exception:
                        pass
                now = time.perf_counter()
                elapsed = now - t0

                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                pct = (done / total) * 100 if total else 100.0

                if progress_cb is not None and (
                    done == 1 or done == total or (elapsed - last_cb_elapsed) >= 0.25
                ):
                    progress_cb(
                        {
                            "stage": "tiling",
                            "progress": float(pct),
                            "elapsed": float(elapsed),
                            "eta": float(eta),
                            "current": int(done),
                            "total": int(total),
                            "rate": float(rate),
                            "message": f"切片中: {Path(out_path_str).name}",
                        }
                    )
                    last_cb_elapsed = elapsed

                # 保留终端打印（不影响网页），但降低频率
                if done == 1 or done == total or (elapsed - last_print_elapsed) >= 0.5:
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    r = f"{rate:.1f} tile/s" if rate > 0 else "-"
                    print(
                        f"[{bar}] {pct:6.2f}% ({done}/{total}) 速率{r} 已用{_format_seconds(elapsed)} 预计剩余{_format_seconds(eta)}",
                        end="\r",
                        flush=True,
                    )
                    last_print_elapsed = elapsed
        except KeyboardInterrupt:
            pool.terminate()
            raise

    t1 = time.perf_counter()
    print(" " * 160, end="\r")

    # 最终强制推一次 100%
    if progress_cb is not None:
        seconds = t1 - t0
        progress_cb(
            {
                "stage": "tiling",
                "progress": 100.0,
                "elapsed": float(seconds),
                "eta": 0.0,
                "current": int(total),
                "total": int(total),
                "rate": float((total / seconds) if seconds > 0 else 0.0),
                "message": "切片完成",
            }
        )

    return width, height, len(tiles), (t1 - t0)


def parse_tile_xy(tile_path: Path) -> tuple[int, int]:
    m = re.match(r"x(\d+)_y(\d+)\.(?:tif{1,2}|png)$", tile_path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"tile文件名不符合 x<数字>_y<数字>.tif: {tile_path.name}")
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_project_folders(project_root)

    args = parse_args()
    if not args.input:
        args = interactive_fill_args(args, project_root)
    # 若用户未显式指定 goal 权重，则根据模式切换默认值
    if args.mode == "goal_only" and args.weights_goal == str(Path("wights/goal/best_goal.pt")):
        args.weights_goal = str(Path("wights/goal/onlygoal.pt"))

    in_path = (project_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    if not in_path.exists():
        print(f"输入文件不存在: {in_path}")
        return 2

    out_root = Path(args.output) if args.output else (project_root / "outputs" / timestamp_name())
    out_root.mkdir(parents=True, exist_ok=True)

    if getattr(args, "clean_output", False):
        # 清理输出目录（仅清理内容，不删除目录本身）
        # 策略：优先送回收站；若失败（或未安装 send2trash）则直接删除。
        send2trash_fn = None
        try:
            from send2trash import send2trash as _send2trash  # type: ignore

            send2trash_fn = _send2trash
        except Exception:
            send2trash_fn = None

        for p in sorted(out_root.glob("*")):
            try:
                if send2trash_fn is not None:
                    try:
                        send2trash_fn(str(p))
                        continue
                    except Exception as _e:
                        print(f"清理失败(回收站，转硬删): {p} err={_e}")

                if p.is_dir():
                    import shutil

                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception as _e:
                print(f"清理失败(硬删也失败): {p} err={_e}")

    (out_root / "pred_images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    tiles_dir = out_root / "tiles"

    info = get_tiff_info(in_path)
    print("TIFF信息:")
    print(f"  path: {in_path}")
    print(f"  shape: {info.get('shape')}")
    print(f"  dtype: {info.get('dtype')}")
    print(f"  pages: {info.get('pages')}  series: {info.get('series')}  bigtiff: {info.get('is_bigtiff')}")
    print(f"  compression: {info.get('compression')}  photometric: {info.get('photometric')}")

    # 兼容旧版本切片(tif/tiff)与新版本切片(png)
    existing_tiles = []
    if tiles_dir.exists():
        existing_tiles = (
            list(tiles_dir.glob("*.png"))
            + list(tiles_dir.glob("*.tif"))
            + list(tiles_dir.glob("*.tiff"))
        )

    if not tiles_dir.exists() or len(existing_tiles) == 0:
        workers_used = min(16, (os.cpu_count() or 8)) if int(args.tile_workers) <= 0 else int(args.tile_workers)
        print(f"输出目录: {out_root}")
        print(f"开始切图: tile={args.tile_size}, overlap={args.overlap}, workers={workers_used}")

        try:
            w, h, n, seconds = tile_big_tiff(
                in_path=in_path,
                tiles_dir=tiles_dir,
                tile=args.tile_size,
                overlap=args.overlap,
                workers=workers_used,
            )
        except Exception as e:  # noqa: BLE001
            print(f"切图失败: {e}")
            return 1

        print(f"切图完成: 原图={w}x{h}, tiles={n}, 用时={_format_seconds(seconds)}, tiles_dir={tiles_dir}")

        if args.only_tile:
            return 0

    from code.pipeline import (  # type: ignore
        RunContext,
        append_summary_row,
        run_normal_only_pairs,
    )
    from code.normal_only_fast import run_normal_only_fast  # type: ignore
    from code.goal_only_fast import run_goal_only_fast  # type: ignore

    ctx = RunContext(out_root=out_root, conf=float(args.conf), iou=float(args.iou))
    summary_csv = out_root / "summary.csv"

    weights_goal = Path(args.weights_goal)
    if not weights_goal.is_absolute():
        weights_goal = (project_root / weights_goal).resolve()
    if not weights_goal.exists():
        print(f"goal权重不存在: {weights_goal}")
        return 2

    if args.mode == "goal_only":
        print("模式：goal_only -> 直接使用 onlygoal.pt 预测目标细胞并输出带框tile")
        # goal_only：conf/iou 必须与 UI/命令行一致
        t0 = time.perf_counter()
        try:
            stats = run_goal_only_fast(
                tiles_dir=tiles_dir,
                out_root=out_root,
                weights_goal=weights_goal,
                conf=float(ctx.conf),
                iou=float(ctx.iou),
                device="0",
                batch=64,
                half=True,
                pred_subdir="goal_only",
                labels_subdir="goal_only",
            )
        except Exception as e:  # noqa: BLE001
            print(f"goal_only失败: {e}")
            return 1
        t1 = time.perf_counter()

        append_summary_row(
            summary_csv,
            {
                "time": timestamp_name(),
                "mode": "goal_only",
                "stage": "goal_tiles",
                "tiles": stats.get("tiles"),
                "boxes": stats.get("boxes"),
                "ctc": stats.get("ctc"),
                "cec": stats.get("cec"),
                "seconds": f"{t1 - t0:.2f}",
                "pred_dir": stats.get("pred_dir"),
                "labels_dir": stats.get("labels_dir"),
            },
        )

        print(f"goal_only完成，用时={_format_seconds(t1 - t0)}")
        print(f"带框tile输出目录: {stats.get('pred_dir')}")
        return 0

    if args.mode == "normal_only":
        weights_normal = Path(args.weights_normal)
        if not weights_normal.is_absolute():
            weights_normal = (project_root / weights_normal).resolve()
        if not weights_normal.exists():
            print(f"normal权重不存在: {weights_normal}")
            return 2

        print("模式：normal_only -> 直接使用 normal 模型输出左右pair")
        t0 = time.perf_counter()
        try:
            pairs_dir = run_normal_only_fast(
                tiles_dir=tiles_dir,
                out_root=out_root,
                weights_normal=weights_normal,
                conf=float(ctx.conf),
                iou=float(ctx.iou),
                device="0",
                batch=64,
                half=True,
                pairs_subdir="normal_only_pairs",
                gap=60,
                png_compression=0,
            )
        except Exception as e:  # noqa: BLE001
            print(f"normal_only失败: {e}")
            return 1
        t1 = time.perf_counter()

        append_summary_row(
            summary_csv,
            {
                "time": timestamp_name(),
                "mode": "normal_only",
                "stage": "normal_only_pairs",
                "tiles": len(list(tiles_dir.glob('*.png')))+len(list(tiles_dir.glob('*.tif')))+len(list(tiles_dir.glob('*.tiff'))),
                "seconds": f"{t1 - t0:.2f}",
                "pairs_dir": str(pairs_dir),
                "labels_dir": str(out_root / "labels" / "normal_only"),
                "conf": f"{ctx.conf}",
                "iou": f"{ctx.iou}",
            },
        )

        print(f"normal_only完成，用时={_format_seconds(t1 - t0)}")
        print(f"pair输出目录: {pairs_dir}")
        return 0

    if args.mode != "two_stage":
        print(f"未知模式: {args.mode}")
        return 2

    raise RuntimeError("two_stage 已删除旧实现，请使用 WebApp 的两阶段(goal-first)新实现。")

    weights_normal = Path(args.weights_normal)
    if not weights_normal.is_absolute():
        weights_normal = (project_root / weights_normal).resolve()
    if not weights_normal.exists():
        print(f"normal权重不存在: {weights_normal}")
        return 2

    print("开始阶段1：普通细胞检测(best_nomal.pt) -> 生成 normal_only / normal_removed")
    t1s = time.perf_counter()
    try:
        normal_only_dir, normal_removed_dir, normal_boxes = run_stage1_normal_tiles(
            tiles_dir=tiles_dir,
            ctx=ctx,
            weights_normal=weights_normal,
        )
    except Exception as e:  # noqa: BLE001
        print(f"阶段1失败: {e}")
        return 1
    t1e = time.perf_counter()

    append_summary_row(
        summary_csv,
        {
            "time": timestamp_name(),
            "mode": "two_stage",
            "stage": "stage1_normal",
            "tiles": len(list(tiles_dir.glob('*.tif'))),
            "boxes": normal_boxes,
            "seconds": f"{t1e - t1s:.2f}",
            "normal_only_dir": str(normal_only_dir),
            "normal_removed_dir": str(normal_removed_dir),
        },
    )

    print("开始阶段2：目标细胞检测(best_goal.pt) -> 输出带框tile(CTC红/CEC绿)")
    t2s = time.perf_counter()
    try:
        stats_removed = run_stage_goal_tile_images(
            tiles_dir=normal_removed_dir,
            ctx=ctx,
            weights_goal=weights_goal,
            labels_subdir="two_stage_goal_from_removed",
            pred_subdir="two_stage/goal_from_removed",
        )
        stats_only = run_stage_goal_tile_images(
            tiles_dir=normal_only_dir,
            ctx=ctx,
            weights_goal=weights_goal,
            labels_subdir="two_stage_goal_from_only",
            pred_subdir="two_stage/goal_from_only",
        )
    except Exception as e:  # noqa: BLE001
        print(f"阶段2失败: {e}")
        return 1
    t2e = time.perf_counter()

    append_summary_row(
        summary_csv,
        {
            "time": timestamp_name(),
            "mode": "two_stage",
            "stage": "stage2_goal_tiles",
            "tiles": stats_removed.get("tiles"),
            "ctc_removed": stats_removed.get("ctc"),
            "cec_removed": stats_removed.get("cec"),
            "pred_removed_dir": stats_removed.get("pred_dir"),
            "ctc_only": stats_only.get("ctc"),
            "cec_only": stats_only.get("cec"),
            "pred_only_dir": stats_only.get("pred_dir"),
            "seconds": f"{t2e - t2s:.2f}",
        },
    )

    print(f"阶段2完成，用时={_format_seconds(t2e - t2s)}")
    print(f"带框tile输出目录(removed): {stats_removed.get('pred_dir')}")
    print(f"带框tile输出目录(only):   {stats_only.get('pred_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
