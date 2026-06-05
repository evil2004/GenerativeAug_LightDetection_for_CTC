@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "APP_DIR=%~dp0"
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"
set "UPDATE_DIR=%~1"

if not defined UPDATE_DIR (
  echo Enter the update folder path.
  echo The folder may contain code, webapp, and wights subfolders.
  set /p "UPDATE_DIR=> "
)

if not exist "%UPDATE_DIR%" (
  echo [ERROR] Update folder does not exist: %UPDATE_DIR%
  pause
  exit /b 1
)

echo [INFO] App folder:    %APP_DIR%
echo [INFO] Update folder: %UPDATE_DIR%
echo.

for %%D in (code webapp wights) do (
  if exist "%UPDATE_DIR%\%%D" (
    echo [INFO] Updating %%D ...
    robocopy "%UPDATE_DIR%\%%D" "%APP_DIR%\%%D" /E /NFL /NDL /NJH /NJS /NP
    if errorlevel 8 (
      echo [ERROR] Failed to update %%D
      pause
      exit /b 1
    )
  )
)

echo.
echo [OK] Update applied. Restart CTC_CEC_AI.exe.
pause
exit /b 0
