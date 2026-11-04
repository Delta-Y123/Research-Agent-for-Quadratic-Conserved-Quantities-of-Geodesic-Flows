@echo off
rem Launch the research UI. Requires Python 3.10+ and sympy.
rem Pure ASCII on purpose: a .bat with non-ASCII paths is decoded with whatever
rem code page cmd happens to use, which mangles them. So we locate the agent
rem directory by searching for ui\server.py instead of hard-coding its name.
setlocal
set "HERE=%~dp0"
set "AGENT="
for /r /d %%D in (*) do (
  if not defined AGENT if exist "%%D\ui\server.py" set "AGENT=%%D"
)
if not defined AGENT (
  echo [ERROR] ui\server.py was not found under %HERE%
  echo         Run this file from the project root.
  pause
  exit /b 1
)
cd /d "%AGENT%"
if /i "%~1"=="--dry" (
  echo AGENT=%AGENT%
  exit /b 0
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title BH Research Agent UI
set "PY=python"
where python >nul 2>&1
if errorlevel 1 (
  where py >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Neither "python" nor "py" was found in PATH.
    echo         Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
  )
  set "PY=py"
)
%PY% ui\server.py --port 8765 --open
echo.
echo Server stopped.
pause