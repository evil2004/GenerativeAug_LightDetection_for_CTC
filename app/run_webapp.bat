@echo off
setlocal EnableExtensions EnableDelayedExpansion

title CTC/CEC Web Launcher

REM Use UTF-8 codepage to avoid garbled output
chcp 65001 >nul

REM Project root = directory containing this .bat
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo [INFO] PROJECT_ROOT=%PROJECT_ROOT%

REM Try to locate conda.bat
set "CONDA_BAT="
set "CONDA_BASE="

REM If conda command works, use conda info --base to find base\condabin\conda.bat
where conda >nul 2>nul
if not errorlevel 1 (
  for /f "usebackq delims=" %%b in (`conda info --base 2^>nul`) do (
    set "CONDA_BASE=%%b"
  )
)

if defined CONDA_BASE if exist "%CONDA_BASE%\condabin\conda.bat" set "CONDA_BAT=%CONDA_BASE%\condabin\conda.bat"

REM Fallback: common locations
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\AppData\Local\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\AppData\Local\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\AppData\Local\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\AppData\Local\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\Anaconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\Anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\Miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\Miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\Anaconda3\condabin\conda.bat" set "CONDA_BAT=C:\Anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\Miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\Miniconda3\condabin\conda.bat"

REM If still not found, fallback to conda run (no need conda.bat)
if not defined CONDA_BAT (
  echo [WARN] conda.bat was not found, but conda might still be usable.
  echo [INFO] Trying: conda run -n yolo ...

  start "CTC/CEC Web" cmd /c "ping 127.0.0.1 -n 3 >nul && start http://127.0.0.1:8000"

  conda run -n yolo python -m webapp.backend.app --project-root "%PROJECT_ROOT%" --host 127.0.0.1 --port 8000
  set "EC=%ERRORLEVEL%"

  if not "%EC%"=="0" (
    echo [ERROR] Web server exited with code=%EC%
    echo [HINT] Please run `conda run -n yolo python -c "import flask"` to verify env.
  ) else (
    echo [INFO] Web server exited.
  )

  pause
  exit /b %EC%
)

echo [INFO] Using conda: %CONDA_BAT%

echo [INFO] If 127.0.0.1:8000 is already in use, please close the previous server.

start "CTC/CEC Web" cmd /c "ping 127.0.0.1 -n 3 >nul && start http://127.0.0.1:8000"

call "%CONDA_BAT%" activate yolo
if errorlevel 1 (
  echo [ERROR] Failed to activate conda env: yolo
  echo [HINT] Please verify: conda env list
  pause
  exit /b 1
)

python -m webapp.backend.app --project-root "%PROJECT_ROOT%" --host 127.0.0.1 --port 8000
set "EC=%ERRORLEVEL%"

if not "%EC%"=="0" (
  echo [ERROR] Web server exited with code=%EC%
) else (
  echo [INFO] Web server exited.
)

pause
endlocal
