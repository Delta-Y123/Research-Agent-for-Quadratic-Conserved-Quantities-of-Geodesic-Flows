@echo off
rem Start the research UI. Requires Python 3.10+ and sympy.
rem Pure ASCII on purpose: non-ASCII paths in a .bat get decoded with whatever
rem code page cmd happens to use. %~dp0 is supplied by cmd itself, so it is safe.
cd /d "%~dp0"
if /i "%~1"=="--dry" (
  echo AGENT=%CD%
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