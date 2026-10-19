@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist "node_modules\@remotion\cli\remotion-cli.js" (
  echo 还未安装视频制作工具，请先运行 install.bat。
  pause
  exit /b 1
)
if not exist "dist\index.html" (
  echo 正在准备中文界面……
  call npm run build
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py --open
) else (
  python app.py --open
)
pause
