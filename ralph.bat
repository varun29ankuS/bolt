@echo off
REM Ralph Wiggum Loop for Bolt (Windows)
REM Run overnight: autonomous development with context

cd /d "%~dp0"

echo ========================================
echo  Bolt Overnight Development (Ralph)
echo ========================================
echo.
echo Context files:
echo   - CLAUDE.md  (architecture)
echo   - TODO.md    (priorities)
echo   - RALPH.md   (instructions)
echo.
echo Press Ctrl+C to stop
echo.

set /a iteration=0

:loop
set /a iteration+=1
echo.
echo ############################################
echo # Iteration %iteration% - %date% %time%
echo ############################################

REM Show current error count
echo Current errors:
.\target\release\bolt.exe check . 2>&1 | find /c "Error:"

echo.
echo Starting Claude session...

type RALPH.md | claude --allowedTools "Bash,Read,Write,Edit,Grep,Glob"

echo.
echo Session complete. Waiting 10 seconds...
timeout /t 10 /nobreak >nul

goto loop
