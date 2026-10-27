@echo off
setlocal

for /D %%D in ("generated cases\*") do (
    echo Running %%D
    ..\islab2000.exe "%%D\input.in2"
)

echo.
echo All cases completed.
pause