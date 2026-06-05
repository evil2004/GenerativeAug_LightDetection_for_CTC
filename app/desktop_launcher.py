from __future__ import annotations

import importlib.util
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path


APP_NAME = "CTC_CEC_AI"


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    path = Path(base) / APP_NAME
    (path / "outputs").mkdir(parents=True, exist_ok=True)
    return path


def find_free_port(host: str = "127.0.0.1", preferred: int = 8000) -> int:
    for port in [preferred, *range(8010, 8050)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def load_backend(root: Path):
    app_py = root / "webapp" / "backend" / "app.py"
    if not app_py.exists():
        raise FileNotFoundError(f"Cannot find backend file: {app_py}")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("ctc_cec_external_backend", app_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load backend spec: {app_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = app_dir()
    data = user_data_dir()
    log_path = data / "software.log"
    host = "127.0.0.1"
    port = find_free_port(host)
    url = f"http://{host}:{port}/"

    try:
        backend = load_backend(root)
        app = backend.create_app(project_root=data, data_root=root)
    except Exception:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise

    def open_browser() -> None:
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    print(f"{APP_NAME} is running at {url}")
    print(f"Outputs: {data / 'outputs'}")
    print(f"Log: {log_path}")
    app.run(host=host, port=port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        input("Press Enter to exit...")
        raise
