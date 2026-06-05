@echo off
setlocal EnableExtensions
chcp 65001 >nul

title CTC_CEC_AI_Web_Server

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "HOST=127.0.0.1"
set "PORT=8000"

echo ============================================================
echo CTC/CEC AI Web Server
echo Project: %PROJECT_ROOT%
echo URL:     http://%HOST%:%PORT%/
echo.
echo Keep this window open while using the web page.
echo Backend logs and error messages will be shown here.
echo ============================================================
echo.

set "PYTHON_EXE="
if exist "D:\anaconda\envs\yolo\python.exe" set "PYTHON_EXE=D:\anaconda\envs\yolo\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\yolo\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\yolo\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\miniconda3\envs\yolo\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\yolo\python.exe"

if defined PYTHON_EXE (
  echo [INFO] Python: %PYTHON_EXE%
  start "" cmd /c "timeout /t 2 /nobreak >nul && start http://%HOST%:%PORT%/"
  "%PYTHON_EXE%" -m webapp.backend.app --project-root "%PROJECT_ROOT%" --data-root "%PROJECT_ROOT%" --host %HOST% --port %PORT%
  set "EC=%ERRORLEVEL%"
  echo.
  echo [INFO] Server exited with code %EC%.
  pause
  exit /b %EC%
)

where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Could not find yolo python or conda.
  echo [HINT] Expected: D:\anaconda\envs\yolo\python.exe
  pause
  exit /b 1
)

echo [INFO] Fallback: conda run -n yolo
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://%HOST%:%PORT%/"
conda run --no-capture-output -n yolo python -m webapp.backend.app --project-root "%PROJECT_ROOT%" --data-root "%PROJECT_ROOT%" --host %HOST% --port %PORT%
set "EC=%ERRORLEVEL%"
echo.
echo [INFO] Server exited with code %EC%.
pause
exit /b %EC%
