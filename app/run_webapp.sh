#!/usr/bin/env sh
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cmd.exe /c start "CTC/CEC Web Server" cmd.exe /k "$SCRIPT_DIR\\run_webapp.bat"
