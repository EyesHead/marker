@echo off
setlocal
if not exist "%~dp0ScriptOllama.py" (
    echo Missing ScriptOllama.py. Download it beside this wrapper ^(see README^). 1>&2
    exit /b 1
)
where py >nul 2>nul
if errorlevel 1 goto python
py -3 "%~dp0ScriptOllama.py" %*
exit /b %errorlevel%
:python
where python >nul 2>nul
if errorlevel 1 goto missing
python "%~dp0ScriptOllama.py" %*
exit /b %errorlevel%
:missing
echo Python 3.10 or newer is required. Install Python and enable the Python launcher or PATH. 1>&2
exit /b 1
